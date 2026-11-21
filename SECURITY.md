# Security Policy

## Reporting a Vulnerability

If you find a security issue, please **do not** open a public issue.

Instead, report it privately by emailing the maintainer or opening a private security advisory on GitHub.

## What Counts as a Security Issue

- API key exposure
- WebSocket authentication bypass
- Unauthorized access to call sessions
- Audio data leaks

## Best Practices for Deployment

- Always use HTTPS (SSL certs required for mic access anyway)
- Never commit `.env` files with real keys
- Use strong, unique API keys
- Restrict VPS firewall to only expose the service port
- Rotate keys periodically
