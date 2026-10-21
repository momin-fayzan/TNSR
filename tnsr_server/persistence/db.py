"""SQLite persistence with versioned migrations.

The database stores conversations, messages, settings, job metadata and an
index of experiments/evaluations. Model weights and metrics stay on disk; the
experiment rows are an index that is re-synced from the filesystem, so a
deleted or moved run is marked ``missing`` instead of silently vanishing.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

MIGRATIONS: list[str] = [
    # v1: initial schema
    """
    CREATE TABLE conversations (
        id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        model_id TEXT,
        mode TEXT NOT NULL DEFAULT 'chat',
        settings TEXT NOT NULL DEFAULT '{}',
        created_at REAL NOT NULL,
        updated_at REAL NOT NULL
    );
    CREATE TABLE messages (
        id TEXT PRIMARY KEY,
        conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
        seq INTEGER NOT NULL,
        role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
        content TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'complete',
        model_id TEXT,
        settings TEXT,
        stats TEXT,
        error TEXT,
        created_at REAL NOT NULL,
        UNIQUE (conversation_id, seq)
    );
    CREATE INDEX idx_messages_conversation ON messages(conversation_id, seq);
    CREATE TABLE settings (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    );
    CREATE TABLE jobs (
        id TEXT PRIMARY KEY,
        kind TEXT NOT NULL,
        params TEXT NOT NULL,
        status TEXT NOT NULL,
        pid INTEGER,
        run_id TEXT,
        log_path TEXT,
        exit_code INTEGER,
        error TEXT,
        result TEXT,
        created_at REAL NOT NULL,
        started_at REAL,
        finished_at REAL
    );
    CREATE TABLE experiments (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        path TEXT NOT NULL,
        state TEXT,
        config TEXT NOT NULL,
        environment TEXT,
        dataset TEXT,
        tokenizer TEXT,
        num_parameters INTEGER,
        best_val_loss REAL,
        step INTEGER,
        max_steps INTEGER,
        created_at TEXT,
        missing INTEGER NOT NULL DEFAULT 0,
        synced_at REAL NOT NULL
    );
    CREATE TABLE evaluations (
        id TEXT PRIMARY KEY,
        kind TEXT NOT NULL CHECK (kind IN ('evaluation', 'benchmark')),
        experiment_id TEXT,
        checkpoint_id TEXT NOT NULL,
        step INTEGER,
        summary TEXT NOT NULL,
        file_path TEXT NOT NULL,
        created_at TEXT,
        missing INTEGER NOT NULL DEFAULT 0
    );
    CREATE INDEX idx_evaluations_checkpoint ON evaluations(checkpoint_id);
    """,
]


def now() -> float:
    return time.time()


def new_id() -> str:
    return uuid.uuid4().hex[:16]


def evaluation_id(file_path: str) -> str:
    return hashlib.sha1(file_path.encode()).hexdigest()[:16]


class Database:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA busy_timeout = 5000")
        if str(path) != ":memory:":
            self._conn.execute("PRAGMA journal_mode = WAL")
        self.migrate()

    # -- infrastructure --------------------------------------------------------------

    @property
    def schema_version(self) -> int:
        return int(self._conn.execute("PRAGMA user_version").fetchone()[0])

    def migrate(self) -> None:
        with self._lock:
            current = self.schema_version
            if current > len(MIGRATIONS):
                raise RuntimeError(
                    f"Database {self.path} has schema v{current}, newer than this TNSR build "
                    f"(v{len(MIGRATIONS)}). Upgrade TNSR instead of downgrading the database."
                )
            for version in range(current + 1, len(MIGRATIONS) + 1):
                with self.transaction() as conn:
                    for statement in MIGRATIONS[version - 1].split(";"):
                        if statement.strip():
                            conn.execute(statement)
                    conn.execute(f"PRAGMA user_version = {version}")

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            else:
                self._conn.execute("COMMIT")

    def query(self, sql: str, params: tuple | dict = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- settings --------------------------------------------------------------------

    def get_setting(self, key: str, default: Any = None) -> Any:
        rows = self.query("SELECT value FROM settings WHERE key = ?", (key,))
        return json.loads(rows[0]["value"]) if rows else default

    def set_setting(self, key: str, value: Any) -> None:
        with self.transaction() as c:
            c.execute(
                "INSERT INTO settings(key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, json.dumps(value)),
            )

    # -- conversations ---------------------------------------------------------------

    @staticmethod
    def _conversation(row: sqlite3.Row, message_count: int | None = None) -> dict[str, Any]:
        out = {
            "id": row["id"],
            "title": row["title"],
            "model_id": row["model_id"],
            "mode": row["mode"],
            "settings": json.loads(row["settings"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }
        if message_count is not None:
            out["message_count"] = message_count
        return out

    @staticmethod
    def _message(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "conversation_id": row["conversation_id"],
            "seq": row["seq"],
            "role": row["role"],
            "content": row["content"],
            "status": row["status"],
            "model_id": row["model_id"],
            "settings": json.loads(row["settings"]) if row["settings"] else None,
            "stats": json.loads(row["stats"]) if row["stats"] else None,
            "error": row["error"],
            "created_at": row["created_at"],
        }

    def create_conversation(
        self, title: str, model_id: str | None, mode: str = "chat", settings: dict | None = None
    ) -> dict[str, Any]:
        cid = new_id()
        ts = now()
        with self.transaction() as c:
            c.execute(
                "INSERT INTO conversations(id, title, model_id, mode, settings, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (cid, title, model_id, mode, json.dumps(settings or {}), ts, ts),
            )
        conv = self.get_conversation(cid)
        assert conv is not None
        return conv

    def list_conversations(self) -> list[dict[str, Any]]:
        rows = self.query(
            "SELECT c.*, (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) AS n "
            "FROM conversations c ORDER BY c.updated_at DESC"
        )
        return [self._conversation(r, r["n"]) for r in rows]

    def get_conversation(self, cid: str, with_messages: bool = False) -> dict[str, Any] | None:
        rows = self.query("SELECT * FROM conversations WHERE id = ?", (cid,))
        if not rows:
            return None
        conv = self._conversation(rows[0])
        msgs = self.list_messages(cid)
        conv["message_count"] = len(msgs)
        if with_messages:
            conv["messages"] = msgs
        return conv

    def update_conversation(self, cid: str, **fields: Any) -> dict[str, Any] | None:
        allowed = {"title", "model_id", "mode", "settings"}
        sets, params = [], []
        for key, value in fields.items():
            if key not in allowed or value is None:
                continue
            sets.append(f"{key} = ?")
            params.append(json.dumps(value) if key == "settings" else value)
        if sets:
            sets.append("updated_at = ?")
            params.append(now())
            with self.transaction() as c:
                cur = c.execute(
                    f"UPDATE conversations SET {', '.join(sets)} WHERE id = ?", (*params, cid)
                )
                if cur.rowcount == 0:
                    return None
        return self.get_conversation(cid)

    def delete_conversation(self, cid: str) -> bool:
        with self.transaction() as c:
            return c.execute("DELETE FROM conversations WHERE id = ?", (cid,)).rowcount > 0

    def list_messages(self, cid: str) -> list[dict[str, Any]]:
        rows = self.query("SELECT * FROM messages WHERE conversation_id = ? ORDER BY seq", (cid,))
        return [self._message(r) for r in rows]

    def get_message(self, mid: str) -> dict[str, Any] | None:
        rows = self.query("SELECT * FROM messages WHERE id = ?", (mid,))
        return self._message(rows[0]) if rows else None

    def add_message(
        self,
        cid: str,
        role: str,
        content: str,
        *,
        status: str = "complete",
        model_id: str | None = None,
        settings: dict | None = None,
    ) -> dict[str, Any]:
        mid = new_id()
        ts = now()
        with self.transaction() as c:
            seq = c.execute(
                "SELECT COALESCE(MAX(seq), 0) + 1 FROM messages WHERE conversation_id = ?", (cid,)
            ).fetchone()[0]
            c.execute(
                "INSERT INTO messages(id, conversation_id, seq, role, content, status, model_id, settings, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    mid,
                    cid,
                    seq,
                    role,
                    content,
                    status,
                    model_id,
                    json.dumps(settings) if settings else None,
                    ts,
                ),
            )
            c.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (ts, cid))
        msg = self.get_message(mid)
        assert msg is not None
        return msg

    def finish_message(
        self,
        mid: str,
        content: str,
        status: str,
        stats: dict | None = None,
        error: str | None = None,
    ) -> dict[str, Any] | None:
        with self.transaction() as c:
            c.execute(
                "UPDATE messages SET content = ?, status = ?, stats = ?, error = ? WHERE id = ?",
                (content, status, json.dumps(stats) if stats else None, error, mid),
            )
        return self.get_message(mid)

    def delete_messages_from(self, cid: str, seq: int) -> int:
        with self.transaction() as c:
            return c.execute(
                "DELETE FROM messages WHERE conversation_id = ? AND seq >= ?", (cid, seq)
            ).rowcount

    def recover_interrupted_messages(self) -> int:
        """Messages left 'streaming' by a crash are marked as interrupted on startup."""
        with self.transaction() as c:
            return c.execute(
                "UPDATE messages SET status = 'error', error = 'Server stopped during generation' "
                "WHERE status = 'streaming'"
            ).rowcount

    # -- jobs ------------------------------------------------------------------------

    @staticmethod
    def _job(row: sqlite3.Row) -> dict[str, Any]:
        out = dict(row)
        out["params"] = json.loads(row["params"])
        out["result"] = json.loads(row["result"]) if row["result"] else None
        return out

    def create_job(self, kind: str, params: dict[str, Any]) -> dict[str, Any]:
        jid = new_id()
        with self.transaction() as c:
            c.execute(
                "INSERT INTO jobs(id, kind, params, status, created_at) VALUES (?, ?, ?, 'pending', ?)",
                (jid, kind, json.dumps(params), now()),
            )
        job = self.get_job(jid)
        assert job is not None
        return job

    def update_job(self, jid: str, **fields: Any) -> None:
        allowed = {
            "status",
            "pid",
            "run_id",
            "log_path",
            "exit_code",
            "error",
            "result",
            "started_at",
            "finished_at",
        }
        sets, params = [], []
        for key, value in fields.items():
            if key not in allowed:
                raise ValueError(f"Unknown job field {key}")
            sets.append(f"{key} = ?")
            params.append(json.dumps(value) if key == "result" and value is not None else value)
        with self.transaction() as c:
            c.execute(f"UPDATE jobs SET {', '.join(sets)} WHERE id = ?", (*params, jid))

    def get_job(self, jid: str) -> dict[str, Any] | None:
        rows = self.query("SELECT * FROM jobs WHERE id = ?", (jid,))
        return self._job(rows[0]) if rows else None

    def list_jobs(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = self.query("SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,))
        return [self._job(r) for r in rows]

    # -- experiment index --------------------------------------------------------------

    def upsert_experiment(self, run: dict[str, Any], path: str) -> None:
        status = run.get("status") or {}
        with self.transaction() as c:
            c.execute(
                """
                INSERT INTO experiments(id, name, path, state, config, environment, dataset, tokenizer,
                    num_parameters, best_val_loss, step, max_steps, created_at, missing, synced_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
                ON CONFLICT(id) DO UPDATE SET
                    path = excluded.path, state = excluded.state, config = excluded.config,
                    best_val_loss = excluded.best_val_loss, step = excluded.step,
                    max_steps = excluded.max_steps, missing = 0, synced_at = excluded.synced_at
                """,
                (
                    run["id"], run["name"], path, status.get("state"), json.dumps(run["config"]),
                    json.dumps(run.get("environment")), json.dumps(run.get("dataset")),
                    json.dumps(run.get("tokenizer")), run.get("num_parameters"),
                    status.get("best_val_loss"), status.get("step"), status.get("max_steps"),
                    run.get("created_at"), now(),
                ),
            )  # fmt: skip

    def mark_missing_experiments(self, present_ids: set[str]) -> None:
        with self.transaction() as c:
            for row in c.execute("SELECT id FROM experiments WHERE missing = 0").fetchall():
                if row["id"] not in present_ids:
                    c.execute("UPDATE experiments SET missing = 1 WHERE id = ?", (row["id"],))

    def list_experiments(self, include_missing: bool = True) -> list[dict[str, Any]]:
        sql = "SELECT * FROM experiments" + ("" if include_missing else " WHERE missing = 0")
        rows = self.query(sql + " ORDER BY created_at DESC")
        out = []
        for r in rows:
            d = dict(r)
            for key in ("config", "environment", "dataset", "tokenizer"):
                d[key] = json.loads(d[key]) if d[key] else None
            d["missing"] = bool(d["missing"])
            out.append(d)
        return out

    def upsert_evaluation(
        self, result: dict[str, Any], experiment_id: str | None, file_path: str
    ) -> None:
        summary = {
            k: result.get(k)
            for k in (
                "metrics",
                "dataset",
                "tokenizer",
                "results",
                "memory",
                "device",
                "num_parameters",
                "threads",
            )
            if k in result
        }
        with self.transaction() as c:
            c.execute(
                """
                INSERT INTO evaluations(id, kind, experiment_id, checkpoint_id, step, summary, file_path, created_at, missing)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)
                ON CONFLICT(id) DO UPDATE SET summary = excluded.summary, missing = 0
                """,
                (
                    evaluation_id(file_path), result["kind"], experiment_id, result["checkpoint_id"], result.get("step"),
                    json.dumps(summary), file_path, result.get("created_at"),
                ),
            )  # fmt: skip

    def mark_missing_evaluations(self, present_files: set[str]) -> None:
        present = {evaluation_id(f) for f in present_files}
        with self.transaction() as c:
            for row in c.execute("SELECT id FROM evaluations WHERE missing = 0").fetchall():
                if row["id"] not in present:
                    c.execute("UPDATE evaluations SET missing = 1 WHERE id = ?", (row["id"],))

    def list_evaluations(
        self, experiment_id: str | None = None, kind: str | None = None
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM evaluations WHERE missing = 0"
        params: list[Any] = []
        if experiment_id:
            sql += " AND experiment_id = ?"
            params.append(experiment_id)
        if kind:
            sql += " AND kind = ?"
            params.append(kind)
        rows = self.query(sql + " ORDER BY created_at DESC", tuple(params))
        out = []
        for r in rows:
            d = dict(r)
            d["summary"] = json.loads(d["summary"])
            d.pop("missing", None)
            d.pop("file_path", None)
            out.append(d)
        return out
