# Security Policy

## Reporting a vulnerability

Please do not open a public issue containing credentials, private conversation data, or an exploitable vulnerability.

Until a dedicated security contact is published, contact the repository owner through GitHub's private contact options. Include a concise description, impact, and reproduction steps without including real secrets or other people's private data.

## Deployment guidance

- Set a strong `RON_API_KEY` in the host's secret manager.
- Use HTTPS and restrict network access to the API.
- Never commit credentials, tokens, private conversation logs, or model weights.
- Give any release-download token only the minimum read permissions needed.
- Rotate a credential immediately if it is exposed.
