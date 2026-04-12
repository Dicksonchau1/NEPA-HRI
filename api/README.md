# NEPA-HRI API

FastAPI wrapper for the NEPA-HRI Affective Modulation Layer.

## Run locally

```bash
pip install -r api/requirements.txt
export NEPA_API_KEYS=npa_test_yourkey
uvicorn api.nepa_hri_api:app --reload
```

Docs available at: `http://localhost:8000/docs`

## Endpoints

| Method | Path | Description | Billed |
|--------|------|-------------|--------|
| GET | `/health` | Health check | No |
| POST | `/session/evaluate` | Full trace evaluation | 1 session |
| POST | `/session/start` | Start streaming session | 1 session |
| POST | `/session/frame` | Push frame to live session | No |
| DELETE | `/session/{id}` | End session | No |

## Auth

All endpoints (except `/health`) require:
```
X-API-Key: npa_live_yourkey
```

## Quick Example

```python
import requests

resp = requests.post(
    "https://your-deployment/session/evaluate",
    headers={"X-API-Key": "npa_live_yourkey"},
    json={
        "interaction_trace": [0.25, 0.31, 0.45, 0.58, 0.74, 0.69],
        "incidents":         [0.0,  0.0,  0.75, 0.0,  0.0,  0.0]
    }
)

aml = resp.json()
print(aml["recommended_action"])   # LOCAL_HANDLE | ESCALATE | DEESCALATE
print(aml["prompt_injection"])      # inject this into your LLM system prompt
```

## Deploy to Railway / Render / Fly.io

```bash
# Railway
railway up

# Render — set NEPA_API_KEYS as environment variable in dashboard
# Fly.io
fly launch
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `NEPA_API_KEYS` | Yes | Comma-separated valid API keys |

See `api/.env.example` for template.
