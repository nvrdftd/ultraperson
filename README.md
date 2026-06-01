# ultraperson

An async CLI chat app that streams responses from OpenAI's Responses API and
calls two upstream tools (`get_weather`, `research_topic`) on demand.

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) for dependency management
- An OpenAI API key
- An Elyos interview API key (for the weather/research endpoints)

## Setup

```bash
uv sync
```

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-...
MODEL=gpt-5.4-mini
ELYOS_API_BASE=https://elyos-interview-907656039105.europe-west2.run.app
ELYOS_API_KEY=...
```

## Run

```bash
uv run python main.py
```

You'll get a `You: ` prompt. Type a question and hit Enter. A spinner shows
while the model thinks and while a tool call is in flight; the response
streams back token-by-token as soon as the first one arrives.

## Interaction

| Action               | Effect                                            |
| -------------------- | ------------------------------------------------- |
| Type a prompt        | Sends a turn to the model                         |
| `quit` / `exit` / `q`| Exits cleanly                                     |
| Ctrl+D               | Exits cleanly (EOF)                               |
| Ctrl+C during a turn | Cancels the in-flight turn, REPL stays alive      |

## Failure handling

Upstream failures (timeouts, 4xx, 5xx, malformed JSON) are folded into an
`{"ok": false, "error": "..."}` envelope and fed back to the model so it can
explain the failure to the user instead of the app crashing. LLM stream errors
and unexpected tool exceptions are also caught — a single bad turn does not
end the session.
