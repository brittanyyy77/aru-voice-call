import json
import uuid
import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from server.session import SessionManager

logger = logging.getLogger(__name__)


def setup_mcp_routes(app: FastAPI, session_manager: SessionManager):
    """MCP endpoints for Aru integration.
    
    MCP is NOT in the real-time call loop.
    It provides two things:
    1. Post-call: retrieve call transcript & emotion summary to sync into Aru memory
    2. Push: send a message into an active call (e.g. Aru wants to say something)
    """

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
                    "version": "0.2.0",
                },
            })

        elif method == "tools/list":
            return _jsonrpc_response(req_id, {
                "tools": [
                    {
                        "name": "get_call_status",
                        "description": "Check if there's an active voice call, and list completed calls available for sync.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {},
                        },
                    },
                    {
                        "name": "get_call_transcript",
                        "description": "Get the full transcript and emotion summary from a completed call. Use this after a call ends to sync the conversation into Aru memory.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "session_id": {
                                    "type": "string",
                                    "description": "Session ID of the completed call",
                                }
                            },
                            "required": ["session_id"],
                        },
                    },
                    {
                        "name": "push_message",
                        "description": "Push a text message into an active voice call. The message will be synthesized as voice and played to the user. Use this when you want to proactively say something during a call.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "session_id": {
                                    "type": "string",
                                    "description": "Active session ID",
                                },
                                "text": {
                                    "type": "string",
                                    "description": "Text to speak to the user",
                                },
                            },
                            "required": ["session_id", "text"],
                        },
                    },
                    {
                        "name": "get_live_emotion",
                        "description": "Get the current emotion state from an active call. Use this to check how the user is feeling during a call.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "session_id": {
                                    "type": "string",
                                    "description": "Active session ID",
                                }
                            },
                            "required": ["session_id"],
                        },
                    },
                ]
            })

        elif method == "tools/call":
            tool_name = params.get("name", "")
            args = params.get("arguments", {})

            if tool_name == "get_call_status":
                active = session_manager.list_sessions()
                completed = session_manager.list_completed_sessions()
                return _jsonrpc_response(req_id, {
                    "content": [{
                        "type": "text",
                        "text": json.dumps({
                            "active_calls": active,
                            "completed_calls_available": completed,
                            "active_count": len(active),
                        }),
                    }],
                })

            elif tool_name == "get_call_transcript":
                sid = args.get("session_id", "")
                # Check completed calls first
                summary = session_manager.get_completed_session(sid)
                if summary:
                    return _jsonrpc_response(req_id, {
                        "content": [{
                            "type": "text",
                            "text": json.dumps(summary, ensure_ascii=False),
                        }],
                    })
                # Maybe still active
                session = session_manager.get_session(sid)
                if session:
                    return _jsonrpc_response(req_id, {
                        "content": [{
                            "type": "text",
                            "text": json.dumps({
                                "status": "call_still_active",
                                "turns_so_far": len(session.history),
                            }),
                        }],
                    })
                return _jsonrpc_response(req_id, {
                    "content": [{
                        "type": "text",
                        "text": '{"error": "session not found"}',
                    }],
                })

            elif tool_name == "push_message":
                sid = args.get("session_id", "")
                text = args.get("text", "")
                session = session_manager.get_session(sid)
                if not session or not session.websocket:
                    return _jsonrpc_response(req_id, {
                        "content": [{
                            "type": "text",
                            "text": '{"error": "no active call with this session_id"}',
                        }],
                    })
                try:
                    # Send text to frontend
                    await session.websocket.send_json({
                        "type": "response",
                        "text": text,
                        "source": "push",
                    })
                    # Synthesize and stream audio
                    async for audio_chunk in session.tts_engine.synthesize_stream(text):
                        await session.websocket.send_bytes(audio_chunk)
                    await session.websocket.send_json({"type": "audio_end"})
                    session.add_response(text)
                    return _jsonrpc_response(req_id, {
                        "content": [{
                            "type": "text",
                            "text": json.dumps({"status": "delivered", "text": text}),
                        }],
                    })
                except Exception as e:
                    return _jsonrpc_response(req_id, {
                        "content": [{
                            "type": "text",
                            "text": json.dumps({"error": str(e)}),
                        }],
                    })

            elif tool_name == "get_live_emotion":
                sid = args.get("session_id", "")
                session = session_manager.get_session(sid)
                if not session:
                    return _jsonrpc_response(req_id, {
                        "content": [{
                            "type": "text",
                            "text": '{"error": "no active call"}',
                        }],
                    })
                latest = session.get_latest_transcript()
                emotion = latest.get("emotion", {}) if latest else {}
                return _jsonrpc_response(req_id, {
                    "content": [{
                        "type": "text",
                        "text": json.dumps({
                            "emotion": emotion,
                            "last_text": latest.get("text", "") if latest else "",
                        }, ensure_ascii=False),
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
