import chess
import pytest
from fastapi.testclient import TestClient

from tnsr_server.app import create_app


@pytest.fixture
def client(chess_home, tmp_path, monkeypatch):
    monkeypatch.setenv("TNSR_HOME", str(chess_home))
    app = create_app(db_path=tmp_path / "chess.db", device="cpu", serve_ui=False)
    with TestClient(app) as c:
        yield c


def test_chess_models_listed(client):
    models = client.get("/api/chess/models").json()["models"]
    assert len(models) >= 1 and all(m["tokenizer"]["kind"] == "chess" for m in models)
    assert sum(m["recommended"] for m in models) == 1


def test_play_a_game_as_white(client):
    game = client.post("/api/chess/games", json={"human_color": "white"}).json()
    assert game["state"]["fen"] == chess.STARTING_FEN and game["engine_move"] is None
    assert len(game["state"]["legal"]) == 20
    assert game["analysis"]["legal_count"] == 20 and game["analysis"]["moves"]
    after = client.post(f"/api/chess/games/{game['id']}/move", json={"move": "e2e4"}).json()
    assert after["moves"][0] == "e2e4" and len(after["moves"]) == 2  # engine replied
    board = chess.Board()
    board.push_uci("e2e4")
    assert chess.Move.from_uci(after["engine_move"]["uci"]) in board.legal_moves
    assert after["state"]["turn"] == "white" and after["state"]["san"][0] == "e4"
    # persisted
    again = client.get(f"/api/chess/games/{game['id']}").json()
    assert again["moves"] == after["moves"]
    assert client.get("/api/chess/games").json()["games"][0]["id"] == game["id"]
    # undo returns to the human's turn before their move
    undone = client.post(f"/api/chess/games/{game['id']}/undo").json()
    assert undone["moves"] == []
    assert client.post(f"/api/chess/games/{game['id']}/undo").status_code == 409


def test_engine_moves_first_when_human_is_black(client):
    game = client.post("/api/chess/games", json={"human_color": "black"}).json()
    assert len(game["moves"]) == 1 and game["engine_move"] is not None
    assert game["state"]["turn"] == "black"


def test_illegal_and_out_of_turn_moves_are_rejected(client):
    gid = client.post("/api/chess/games", json={"human_color": "white"}).json()["id"]
    assert client.post(f"/api/chess/games/{gid}/move", json={"move": "e2e5"}).status_code == 409
    assert client.post(f"/api/chess/games/{gid}/move", json={"move": "zz99"}).status_code == 422
    assert client.post("/api/chess/games/nope/move", json={"move": "e2e4"}).status_code == 404
    black = client.post("/api/chess/games", json={"human_color": "black"}).json()
    # it is black's (human's) turn; a white move is illegal
    assert (
        client.post(f"/api/chess/games/{black['id']}/move", json={"move": "d2d4"}).status_code
        == 409
    )


def test_game_over_detection_and_custom_start(client):
    # White to move and mate in one: Ra8#
    fen = "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"
    game = client.post("/api/chess/games", json={"human_color": "white", "start_fen": fen}).json()
    done = client.post(f"/api/chess/games/{game['id']}/move", json={"move": "a1a8"}).json()
    assert done["state"]["outcome"] == {"result": "1-0", "termination": "checkmate"}
    assert done["result"] == "1-0" and done["engine_move"] is None and done["state"]["legal"] == []
    assert (
        client.post(f"/api/chess/games/{game['id']}/move", json={"move": "g1f1"}).status_code == 409
    )
    assert client.post("/api/chess/games", json={"start_fen": "not a fen"}).status_code == 409


def test_analyse_rename_delete(client):
    res = client.post("/api/chess/analyse", json={"fen": chess.STARTING_FEN}).json()
    assert len(res["analysis"]["moves"]) == 8 and res["state"]["turn"] == "white"
    probs = [m["win_prob"] for m in res["analysis"]["moves"]]
    assert probs == sorted(probs, reverse=True)
    assert client.post("/api/chess/analyse", json={"fen": "bad"}).status_code == 422
    gid = client.post("/api/chess/games", json={}).json()["id"]
    assert (
        client.patch(f"/api/chess/games/{gid}", json={"title": "Blitz"}).json()["title"] == "Blitz"
    )
    assert client.delete(f"/api/chess/games/{gid}").status_code == 204
    assert client.get(f"/api/chess/games/{gid}").status_code == 404


def test_missing_model_is_a_clear_error(client):
    res = client.post(
        "/api/chess/games", json={"model_id": "experiments/gone/checkpoints/step_0000001"}
    )
    assert res.status_code == 409 and "Model unavailable" in res.json()["detail"]
