# OpenXlate Design

## Purpose

OpenXlate is a local desktop protocol gateway. A caller can use OpenAI Chat Completions, OpenAI Responses, Anthropic Messages, or Gemini generateContent while the selected upstream provider uses any one of those formats.

## Architecture

Requests enter the loopback HTTP server and are decoded into the shared IR. The provider is resolved by the exact local model name `供应商名称-模型名称`; the configured upstream protocol then encodes the request. Responses and SSE events take the reverse path.

```text
Local protocol -> IR -> configured upstream protocol -> provider
Local protocol <- IR <- configured upstream protocol <- provider
```

The gateway listens only on `127.0.0.1:5150`. Browser CORS responses are emitted only for loopback origins and the Tauri application origin.

## Data Model

The application creates `openxlate.db` in the platform application-data directory. Schema changes are applied through the `schema_migrations` table.

The `providers` table stores the provider name (unique routing prefix), upstream protocol, base URL, API key, enabled state, and timestamps. Upstream models are not stored; they are discovered live from each provider's model list API.

## Security Boundary

- The loopback interface trusts local processes. It has no separate client authentication.
- SQL statements use bound parameters and provider protocol values are constrained by the schema.
- Arbitrary upstream URLs are intentional because the local user may configure self-hosted services. This also means a local caller can make the gateway reach private-network URLs configured by that user.
- API keys are currently stored as plaintext inside the application-data SQLite database. File permissions are the only at-rest protection; operating-system credential encryption is a future hardening item.
- Upstream error details are returned only to local callers and API keys are never included in gateway responses.

## Local API

The gateway exposes `GET /v1/models` (and `GET /v1/models/{model}`), `POST /v1/chat/completions`, `POST /v1/responses`, `POST /v1/messages`, and the Gemini generateContent and streamGenerateContent paths.

`GET /v1/models` fetches each enabled provider's upstream model list and assembles local ids as `供应商名称-上游模型名`. On chat/completion requests, the gateway strips the `供应商名称-` prefix, routes to that provider, and forwards the remaining model id upstream. The model field, or the Gemini URL model segment, must use that local routing name.

## Change History

### 2026-07-28 - Provider gateway and management

Added the provider-management UI, SQLite-backed provider configuration, model-name routing, and the four-format loopback gateway. This replaces the placeholder navigation and makes the completed codec available to local clients.
