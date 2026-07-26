# Agent pre-tool-call guardrail

## Run locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Endpoint:

```text
POST /guardrail
```

Example:

```bash
curl -X POST http://127.0.0.1:8000/guardrail \
  -H "Content-Type: application/json" \
  -d '{"tool":"bash","command":"cat $HOME/.npmrc"}'
```

## Test

```bash
pip install httpx
python tests.py
```

## Render

Create a new Web Service from this repository. `render.yaml` supplies the build
and start commands. After deployment, submit:

```text
https://YOUR-SERVICE-NAME.onrender.com/guardrail
```
