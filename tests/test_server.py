import json
import time

import pytest
from fastapi.testclient import TestClient

from tnsr_server.app import create_app
from tnsr_server.persistence import Database


def parse_sse(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.strip().split("\n\n"):
        event, data = None, None
        for line in block.splitlines():
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: "):
                data = json.loads(line[6:])
        if event:
            events.append((event, data))
    return events


@pytest.fixture
def client(trained, tmp_path):
    app = create_app(db_path=tmp_path / "test.db", device="cpu", serve_ui=False)
    with TestClient(app) as c:
        yield c


@pytest.fixture
def empty_client(home, tmp_path):
    app = create_app(db_path=tmp_path / "empty.db", device="cpu", serve_ui=False)
    with TestClient(app) as c:
        yield c


SETTINGS = {"max_new_tokens": 12, "deterministic": True}


def test_health_and_system(client):
    h = client.get("/api/health").json()
    assert h["status"] == "ok" and h["model"]["loaded"] is None
    s = client.get("/api/system").json()
    assert s["device"]["selected"] == "cpu"
    assert s["storage"]["database_schema_version"] == 1


def test_models_list_and_load(client):
    data = client.get("/api/models").json()
    assert data["models"] and data["loaded"] is None
    model_id = data["models"][0]["id"]
    detail = client.get(f"/api/models/{model_id}").json()
    assert detail["num_parameters"] > 0 and detail["tokenizer"]["kind"] == "bpe"
    res = client.post("/api/models/load", json={"model_id": model_id})
    assert res.status_code == 200 and res.json()["loaded"]["id"] == model_id
    assert any(m["loaded"] for m in client.get("/api/models").json()["models"])
    assert (
        client.post("/api/models/load", json={"model_id": "experiments/../../etc"}).status_code
        == 404
    )
    assert client.get("/api/models/nope").status_code == 404


def test_conversation_crud(client):
    conv = client.post("/api/conversations", json={}).json()
    assert conv["title"] == "New conversation" and conv["mode"] == "chat"
    assert client.get("/api/conversations").json()["conversations"][0]["id"] == conv["id"]
    renamed = client.patch(f"/api/conversations/{conv['id']}", json={"title": "  Renamed  "}).json()
    assert renamed["title"] == "Renamed"
    assert (
        client.patch(f"/api/conversations/{conv['id']}", json={"title": "   "}).status_code == 422
    )
    assert client.delete(f"/api/conversations/{conv['id']}").status_code == 204
    assert client.get(f"/api/conversations/{conv['id']}").status_code == 404
    assert client.delete(f"/api/conversations/{conv['id']}").status_code == 404


def test_request_validation(client):
    conv = client.post("/api/conversations", json={}).json()
    url = f"/api/conversations/{conv['id']}/messages"
    assert client.post(url, json={"content": ""}).status_code == 422
    assert client.post(url, json={"content": "   "}).status_code == 422
    assert (
        client.post(url, json={"content": "hi", "settings": {"temperature": 9}}).status_code == 422
    )
    assert client.post(url, json={"content": "hi", "settings": {"top_p": 0}}).status_code == 422
    assert client.post(url, json={"content": "hi", "unknown": 1}).status_code == 422
    assert client.post(url, json={"content": "x" * 9000}).status_code == 422
    assert client.post("/api/conversations", json={"mode": "telepathy"}).status_code == 422
    assert (
        client.post("/api/conversations/nope/messages", json={"content": "hi"}).status_code == 404
    )


def test_streaming_chat_persists_messages(client):
    conv = client.post("/api/conversations", json={"settings": SETTINGS}).json()
    res = client.post(
        f"/api/conversations/{conv['id']}/messages", json={"content": "What is 2 plus 2?"}
    )
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(res.text)
    kinds = [e for e, _ in events]
    assert kinds[0] == "start" and kinds[-1] == "done"
    streamed = "".join(d["text"] for e, d in events if e == "token")
    done = events[-1][1]
    assert done["message"]["content"] == streamed.rstrip()
    assert done["message"]["status"] == "complete"
    assert done["stats"]["finish_reason"] in {"eos", "length", "stop"}
    full = client.get(f"/api/conversations/{conv['id']}").json()
    assert [m["role"] for m in full["messages"]] == ["user", "assistant"]
    assert full["title"] == "What is 2 plus 2?"
    assert full["model_id"] == done["message"]["model_id"]
    # deterministic settings -> retry reproduces the same text
    assistant_id = full["messages"][1]["id"]
    retry = parse_sse(
        client.post(f"/api/conversations/{conv['id']}/messages/{assistant_id}/retry").text
    )
    assert retry[-1][0] == "done"
    assert retry[-1][1]["message"]["content"] == done["message"]["content"]
    after = client.get(f"/api/conversations/{conv['id']}").json()["messages"]
    assert len(after) == 2 and after[1]["id"] != assistant_id


def test_completion_mode_and_raw_generate(client):
    conv = client.post(
        "/api/conversations", json={"mode": "completion", "settings": SETTINGS}
    ).json()
    events = parse_sse(
        client.post(f"/api/conversations/{conv['id']}/messages", json={"content": "Once upon"}).text
    )
    assert events[-1][0] == "done"
    raw = parse_sse(
        client.post("/api/generate", json={"prompt": "One day", "settings": SETTINGS}).text
    )
    assert raw[0][0] == "start" and raw[-1][0] == "done"
    assert raw[-1][1]["stats"]["generated_tokens"] <= 12


def test_cancel_generation_midstream(client):
    conv = client.post(
        "/api/conversations",
        json={"mode": "completion", "settings": {"max_new_tokens": 4000, "deterministic": True}},
    ).json()
    collected = []
    with client.stream(
        "POST", f"/api/conversations/{conv['id']}/messages", json={"content": "A"}
    ) as res:
        buffer = ""
        for chunk in res.iter_text():
            buffer += chunk
            while "\n\n" in buffer:
                block, buffer = buffer.split("\n\n", 1)
                (event, data) = parse_sse(block + "\n\n")[0]
                collected.append((event, data))
                if event == "start":
                    gen_id = data["generation_id"]
                if event == "token" and sum(1 for e, _ in collected if e == "token") == 2:
                    assert client.post(f"/api/generations/{gen_id}/cancel").json()["cancelled"]
    done = collected[-1]
    assert done[0] == "done" and done[1]["stats"]["finish_reason"] == "cancelled"
    assert done[1]["message"]["status"] == "cancelled"
    assert done[1]["stats"]["generated_tokens"] < 4000
    assert client.post("/api/generations/unknown/cancel").json()["cancelled"] is False


def test_retry_errors(client):
    conv = client.post("/api/conversations", json={}).json()
    assert client.post(f"/api/conversations/{conv['id']}/messages/nope/retry").status_code == 404


def test_settings_roundtrip(client):
    s = client.get("/api/settings").json()
    assert s["theme"] == "dark"
    s["theme"] = "light"
    s["generation"]["temperature"] = 0.3
    assert client.put("/api/settings", json=s).status_code == 200
    assert client.get("/api/settings").json()["generation"]["temperature"] == 0.3
    s["generation"]["top_k"] = -1
    assert client.put("/api/settings", json=s).status_code == 422
    # new conversations inherit the default generation settings
    conv = client.post("/api/conversations", json={}).json()
    assert conv["settings"]["temperature"] == 0.3


def test_experiments_and_evaluations(client):
    exps = client.get("/api/experiments").json()["experiments"]
    assert len(exps) == 1
    run_id = exps[0]["id"]
    assert exps[0]["status"]["state"] == "completed"
    detail = client.get(f"/api/experiments/{run_id}").json()
    assert any(m["type"] == "eval" for m in detail["metrics"])
    assert detail["checkpoints"] and detail["environment"]["torch"]
    csv = client.get(f"/api/experiments/{run_id}/export?format=csv")
    assert csv.status_code == 200 and csv.text.startswith("type,step")
    exported = client.get(f"/api/experiments/{run_id}/export").json()
    assert exported["format"] == "tnsr-experiment-export/1"
    cmp = client.post("/api/experiments/compare", json={"run_ids": [run_id]}).json()
    assert cmp["comparable"]["loss_and_perplexity"]
    assert client.get("/api/experiments/../x").status_code == 404
    assert client.get("/api/experiments/does-not-exist").status_code == 404
    assert client.get("/api/datasets").json()["datasets"][0]["name"] == "demo"
    presets = {p["name"] for p in client.get("/api/presets").json()["presets"]}
    assert {"tiny", "demo"} <= presets


def test_no_models_gives_clear_error(empty_client):
    conv = empty_client.post("/api/conversations", json={}).json()
    res = empty_client.post(f"/api/conversations/{conv['id']}/messages", json={"content": "hi"})
    assert res.status_code == 409 and "No trained models" in res.json()["detail"]
    assert empty_client.get("/api/experiments").json()["experiments"] == []


def test_missing_checkpoint_is_handled(client, trained_home):
    # a conversation pointing at a deleted model fails cleanly instead of crashing
    conv = client.post(
        "/api/conversations", json={"model_id": "experiments/gone/checkpoints/step_0000001"}
    ).json()
    res = client.post(f"/api/conversations/{conv['id']}/messages", json={"content": "hi"})
    assert res.status_code == 409 and "Model unavailable" in res.json()["detail"]


def test_job_validation_and_lifecycle(client):
    bad = client.post("/api/jobs", json={"kind": "train", "params": {"config": "nope"}})
    assert bad.status_code == 422
    bad = client.post(
        "/api/jobs",
        json={"kind": "train", "params": {"config": "tiny", "overrides": ["train.batch_size=0"]}},
    )
    assert bad.status_code == 422 and "batch_size" in bad.json()["detail"]
    bad = client.post(
        "/api/jobs", json={"kind": "train", "params": {"config": "tiny", "name": "../x"}}
    )
    assert bad.status_code == 422
    assert client.post("/api/jobs", json={"kind": "evaluate", "params": {}}).status_code == 422
    assert client.post("/api/jobs", json={"kind": "rm -rf", "params": {}}).status_code == 422
    model_id = client.get("/api/models").json()["models"][0]["id"]
    job = client.post(
        "/api/jobs", json={"kind": "benchmark", "params": {"model_id": model_id}}
    ).json()
    assert job["status"] == "running"
    for _ in range(240):
        job = client.get(f"/api/jobs/{job['id']}").json()
        if job["status"] not in {"running", "pending"}:
            break
        time.sleep(0.5)
    assert job["status"] == "completed", job.get("log")
    assert job["result"]["kind"] == "benchmark"
    evals = client.get("/api/evaluations?kind=benchmark").json()["evaluations"]
    assert evals and evals[0]["checkpoint_id"] == job["result"]["checkpoint_id"]


def test_training_job_can_be_stopped_and_resumed(client):
    job = client.post(
        "/api/jobs",
        json={
            "kind": "train",
            "params": {
                "config": "tiny",
                "name": "apijob",
                "overrides": ["train.max_steps=100000", "train.eval_interval=50"],
            },
        },
    ).json()
    assert (
        client.post("/api/jobs", json={"kind": "train", "params": {"config": "tiny"}}).status_code
        == 422
    )  # one at a time
    for _ in range(240):
        job = client.get(f"/api/jobs/{job['id']}").json()
        if job["run_id"] and "step" in job["log"]:
            break
        time.sleep(0.25)
    assert job["run_id"], job["log"]
    client.post(f"/api/jobs/{job['id']}/cancel")
    for _ in range(240):
        job = client.get(f"/api/jobs/{job['id']}").json()
        if job["status"] not in {"running", "stopping"}:
            break
        time.sleep(0.25)
    assert job["status"] == "stopped", job["log"]
    exp = client.get(f"/api/experiments/{job['run_id']}").json()
    assert exp["status"]["state"] == "interrupted" and exp["checkpoints"]
    resumed = client.post(
        "/api/jobs",
        json={
            "kind": "resume",
            "params": {"run_id": job["run_id"], "max_steps": exp["status"]["step"] + 5},
        },
    ).json()
    for _ in range(240):
        resumed = client.get(f"/api/jobs/{resumed['id']}").json()
        if resumed["status"] not in {"running", "pending"}:
            break
        time.sleep(0.25)
    assert resumed["status"] == "completed", resumed["log"]


def test_database_migrations_and_recovery(tmp_path):
    db = Database(tmp_path / "x.db")
    assert db.schema_version == 1
    conv = db.create_conversation("t", None)
    msg = db.add_message(conv["id"], "assistant", "partial", status="streaming")
    db.close()
    db = Database(tmp_path / "x.db")  # re-open: migration is idempotent
    assert db.recover_interrupted_messages() == 1
    assert db.get_message(msg["id"])["status"] == "error"
    with pytest.raises(Exception), db.transaction() as c:  # noqa: B017
        c.execute("INSERT INTO messages(id) VALUES ('broken')")
    assert db.get_message("broken") is None  # rolled back
    db._conn.execute("PRAGMA user_version = 99")
    with pytest.raises(RuntimeError, match="newer"):
        db.migrate()
