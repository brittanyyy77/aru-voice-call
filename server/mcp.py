import json
import uuid
import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from server.session import SessionManager

logger = logging.getLogger(__name__)


def setup_mcp_routes(app: FastAPI, session_manager: SessionManager):
    """Expose MCP-compatible tool endpoints."""

    @app.post("/mcp")
    async def mcp_handler(request: Request):
        body = await request.json()
        method = body.get("method", "")
        params = body.get("params", {})
        req_id = body.get("id", str(uuid.uuid4()))

        if method == "initialize":
            return _jsonrpc_response(req_id, {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {
                    "name": "aru-voice-call",
                    "version": "0.1.0",
                },
            })

        elif method == "tools/list":
            return _jsonrpc_response(req_id, {
                "tools": [
                    {
                        "name": "start_call",
                        "description": "Start a voice call session. Returns session ID and WebSocket URL.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {},
                        },
                    },
                    {
                        "name": "end_call",
                        "description": "End an active voice call session.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "session_id": {
                                    "type": "string",
                                    "description": "Session ID to end",
                                }
                            },
                            "required": ["session_id"],
                        },
                    },
                    {
                        "name": "get_transcript",
                        "description": "Get the latest transcription and emotion from an active call.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "session_id": {
                                    "type": "string",
                                    "description": "Session ID",
                                }
                            },
                            "required": ["session_id"],
                        },
                    },
                    {
                        "name": "send_voice_reply",
                        "description": "Send text to be spoken back to the user via TTS.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "session_id": {
                                    "type": "string",
                                    "description": "Session ID",
                                },
                                "text": {
                                    "type": "string",
                                    "description": "Text to speak",
                                },
                            },
                            "required": ["session_id", "text"],
                        },
                    },
                    {
                        "name": "get_call_status",
                        "description": "Get status of all active calls.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {},
                        },
                    },
                ]
            })

        elif method == "tools/call":
            tool_name = params.get("name", "")
            args = params.get("arguments", {})

            if tool_name == "start_call":
                session_id = str(uuid.uuid4())[:8]
                session_manager.get_or_create(session_id)
                return _jsonrpc_response(req_id, {
                    "content": [{
                        "type": "text",
                        "text": json.dumps({
                            "session_id": session_id,
                            "ws_url": f"/ws/call/{session_id}",
                            "status": "ready",
                        }),
                    }],
                })

            elif tool_name == "end_call":
                sid = args.get("session_id", "")
                session_manager.end_session(sid)
                return _jsonrpc_response(req_id, {
                    "content": [{
                        "type": "text",
                        "text": json.dumps({"status": "ended", "session_id": sid}),
                    }],
                })

            elif tool_name == "get_transcript":
                sid = args.get("session_id", "")
                session = session_manager.get_session(sid)
                if not session:
                    return _jsonrpc_response(req_id, {
                        "content": [{"type": "text", "text": '{"error": "session not found"}'}],
                    })
                latest = session.get_latest_transcript()
                return _jsonrpc_response(req_id, {
                    "content": [{
                        "type": "text",
                        "text": json.dumps(latest or {"text": "", "emotion": {"emotion": "neutral"}}),
                    }],
                })

            elif tool_name == "send_voice_reply":
                sid = args.get("session_id", "")
                text = args.get("text", "")
                session = session_manager.get_session(sid)
                if not session:
                    return _jsonrpc_response(req_id, {
                        "content": [{"type": "text", "text": '{"error": "session not found"}'}],
                    })
                # TTS will be handled through WebSocket
                session.add_response(text)
                return _jsonrpc_response(req_id, {
                    "content": [{
                        "type": "text",
                        "text": json.dumps({"status": "queued", "text": text}),
                    }],
                })

            elif tool_name == "get_call_status":
                sessions = session_manager.list_sessions()
                return _jsonrpc_response(req_id, {
                    "content": [{
                        "type": "text",
                        "text": json.dumps({
                            "active_sessions": sessions,
                            "count": len(sessions),
                        }),
                    }],
                })

            return _jsonrpc_error(req_id, -32601, f"Unknown tool: {tool_name}")

        return _jsonrpc_error(req_id, -32601, f"Unknown method: {method}")


def _jsonrpc_response(req_id, result):
    return JSONResponse({
        "jsonrpc": "2.0",
        "id": req_id,
        "result": result,
    })


def _jsonrpc_error(req_id, code, message):
    return JSONResponse({
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": code, "message": message},
    })
