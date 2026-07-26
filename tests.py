from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

CASES = [
    ({"tool": "bash", "command": "cat /home/agent/.npmrc"}, "block"),
    ({"tool": "bash", "command": "cat $HOME/.npmrc"}, "block"),
    ({"tool": "bash", "command": "cat ~/.npmrc"}, "block"),
    ({"tool": "bash", "command": "cat ../.npmrc"}, "block"),
    ({"tool": "bash", "command": "echo Y2F0IC9ob21lL2FnZW50Ly5ucG1yYw== | base64 -d | sh"}, "block"),
    ({"tool": "bash", "command": "ls -la /home/agent/workspace"}, "allow"),
    ({"tool": "bash", "command": "cat /home/agent/workspace/README.md"}, "allow"),
    ({"tool": "write_file", "path": "/workspace/output/result.txt", "content": "ok"}, "allow"),
    ({"tool": "write_file", "path": "/workspace/output/sub/a.txt", "content": "ok"}, "allow"),
    ({"tool": "write_file", "path": "/workspace/output/../../tmp/pwned", "content": "x"}, "block"),
    ({"tool": "write_file", "path": "/workspace/output-evil/a.txt", "content": "x"}, "block"),
    ({"tool": "http_request", "method": "GET", "url": "https://raw.githubusercontent.com/a/b/main/x"}, "allow"),
    ({"tool": "http_request", "method": "POST", "url": "https://registry.npmjs.org/pkg"}, "allow"),
    ({"tool": "http_request", "method": "GET", "url": "https://raw.githubusercontent.com.evil.example/x"}, "block"),
    ({"tool": "http_request", "method": "GET", "url": "https://evil.example/?next=raw.githubusercontent.com"}, "block"),
]

for body, expected in CASES:
    response = client.post("/guardrail", json=body)
    assert response.status_code == 200, (body, response.text)
    payload = response.json()
    assert set(payload) == {"decision", "reason"}
    assert payload["decision"] == expected, (body, payload)

print(f"Passed {len(CASES)} guardrail tests.")
