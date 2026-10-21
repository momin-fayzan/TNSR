"""End-to-end: real weights from a tiny training run served by a real uvicorn process."""

import json
import os
import socket
import subprocess
import sys
import time

import httpx
import pytest


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def server(trained, trained_home, tmp_path):
    port = free_port()
    env = {
        **os.environ,
        "TNSR_HOME": str(trained_home),
        "TNSR_DB": str(tmp_path / "e2e.db"),
        "TNSR_DEVICE": "cpu",
    }
    proc = subprocess.Popen(
        [sys.executable, "-m", "tnsr_core.cli", "serve", "--port", str(port)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    base = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            if httpx.get(f"{base}/api/health", timeout=1).status_code == 200:
                break
        except httpx.HTTPError:
            time.sleep(0.2)
    else:
        proc.kill()
        pytest.fail(f"server did not start: {proc.stdout.read().decode() if proc.stdout else ''}")
    yield base
    proc.terminate()
    proc.wait(timeout=10)


def read_events(response: httpx.Response):
    buffer = ""
    for chunk in response.iter_text():
        buffer += chunk
        while "\n\n" in buffer:
            block, buffer = buffer.split("\n\n", 1)
            lines = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
            yield lines["event"], json.loads(lines["data"])


def test_real_server_streams_generated_text(server):
    with httpx.Client(base_url=server, timeout=60) as client:
        models = client.get("/api/models").json()["models"]
        assert models, "trained checkpoint should be discovered"
        conv = client.post(
            "/api/conversations", json={"settings": {"max_new_tokens": 20, "deterministic": True}}
        ).json()
        token_times = []
        with client.stream(
            "POST", f"/api/conversations/{conv['id']}/messages", json={"content": "Hello!"}
        ) as res:
            assert res.status_code == 200
            events = []
            for event, data in read_events(res):
                events.append((event, data))
                if event == "token":
                    token_times.append(time.perf_counter())
        assert events[0][0] == "start" and events[-1][0] == "done"
        text = "".join(d["text"] for e, d in events if e == "token")
        final = events[-1][1]["message"]
        assert final["content"] == text.rstrip() and final["status"] == "complete"
        assert final["model_id"] == models[0]["id"] or final["model_id"].startswith("experiments/")
        # Persisted and reloadable
        stored = client.get(f"/api/conversations/{conv['id']}").json()["messages"]
        assert [m["role"] for m in stored] == ["user", "assistant"]
        assert stored[1]["content"] == final["content"]


def test_client_disconnect_cancels_generation(server):
    with httpx.Client(base_url=server, timeout=60) as client:
        conv = client.post(
            "/api/conversations",
            json={
                "mode": "completion",
                "settings": {"max_new_tokens": 4000, "deterministic": True},
            },
        ).json()
        with client.stream(
            "POST", f"/api/conversations/{conv['id']}/messages", json={"content": "Once"}
        ) as res:
            seen = 0
            for event, _ in read_events(res):
                if event == "token":
                    seen += 1
                    if seen >= 3:
                        break  # leaving the context closes the connection mid-stream
        for _ in range(50):
            msgs = client.get(f"/api/conversations/{conv['id']}").json()["messages"]
            if msgs[-1]["status"] != "streaming":
                break
            time.sleep(0.1)
        assert msgs[-1]["status"] == "cancelled"
        assert msgs[-1]["content"]
        # The model is free again for the next request
        conv2 = client.post(
            "/api/conversations", json={"settings": {"max_new_tokens": 5, "deterministic": True}}
        ).json()
        res = client.post(f"/api/conversations/{conv2['id']}/messages", json={"content": "Hi"})
        assert "event: done" in res.text
