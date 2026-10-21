"""Experiment index: syncs run directories and result files into SQLite and builds views."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

from tnsr_core.checkpoints import CheckpointError, describe_checkpoint, run_checkpoint_index
from tnsr_core.evaluation import load_results
from tnsr_core.paths import checkpoints_dir, datasets_dir, experiments_dir
from tnsr_core.training import list_runs, read_metrics, read_run, resolve_run
from tnsr_server.persistence import Database


class ExperimentService:
    def __init__(self, db: Database):
        self.db = db

    def sync(self) -> None:
        """Re-index run folders and result files; mark vanished ones as missing."""
        present: set[str] = set()
        result_files: set[str] = set()
        for run in list_runs():
            run_dir = experiments_dir() / run["id"]
            present.add(run["id"])
            self.db.upsert_experiment(run, str(run_dir))
            for kind in ("evals", "benchmarks"):
                folder = run_dir / kind
                for f in sorted(folder.glob("*.json")) if folder.is_dir() else []:
                    self._index_result(f, run["id"], result_files)
        if checkpoints_dir().is_dir():
            for ck in checkpoints_dir().iterdir():
                for kind in ("evals", "benchmarks"):
                    for f in sorted((ck / kind).glob("*.json")) if (ck / kind).is_dir() else []:
                        self._index_result(f, None, result_files)
        self.db.mark_missing_experiments(present)
        self.db.mark_missing_evaluations(result_files)

    def _index_result(self, f: Path, run_id: str | None, seen: set[str]) -> None:
        try:
            data = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError):
            return
        if data.get("kind") not in {"evaluation", "benchmark"} or "checkpoint_id" not in data:
            return
        self.db.upsert_evaluation(data, run_id, str(f))
        seen.add(str(f))

    def list_all(self) -> list[dict[str, Any]]:
        self.sync()
        out = []
        for exp in self.db.list_experiments():
            if not exp["missing"]:
                try:
                    exp["status"] = read_run(Path(exp["path"])).get("status", {})
                except (OSError, json.JSONDecodeError):
                    exp["status"] = {}
            else:
                exp["status"] = {"state": "missing"}
            exp.pop("path", None)
            out.append(exp)
        return out

    def detail(self, run_id: str) -> dict[str, Any]:
        run_dir = resolve_run(run_id)
        run = read_run(run_dir)
        index = run_checkpoint_index(run_dir)
        checkpoints = []
        for entry in index.get("checkpoints", []):
            path = run_dir / "checkpoints" / entry["name"]
            try:
                info = describe_checkpoint(path)
            except CheckpointError:
                info = {"name": entry["name"], "step": entry["step"], "missing": True}
            checkpoints.append(info)
        metrics = read_metrics(run_dir)
        return {
            **run,
            "checkpoints": checkpoints,
            "best_checkpoint": index.get("best"),
            "latest_checkpoint": index.get("latest"),
            "metrics": metrics,
            "evaluations": load_results(run_dir, "evals"),
            "benchmarks": load_results(run_dir, "benchmarks"),
        }

    def export_json(self, run_id: str) -> dict[str, Any]:
        detail = self.detail(run_id)
        return {"format": "tnsr-experiment-export/1", **detail}

    def export_metrics_csv(self, run_id: str) -> str:
        metrics = read_metrics(resolve_run(run_id))
        columns = [
            "type",
            "step",
            "loss",
            "val_loss",
            "train_loss",
            "val_perplexity",
            "lr",
            "grad_norm",
            "tokens_per_sec",
            "elapsed",
            "rss_mb",
        ]
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in metrics:
            writer.writerow(row)
        return buf.getvalue()

    def comparison(self, run_ids: list[str]) -> dict[str, Any]:
        """Side-by-side view that flags metrics that are not comparable."""
        rows = []
        for rid in run_ids:
            d = self.detail(rid)
            latest_eval = d["evaluations"][-1] if d["evaluations"] else None
            latest_bench = d["benchmarks"][-1] if d["benchmarks"] else None
            rows.append(
                {
                    "id": d["id"],
                    "name": d["name"],
                    "num_parameters": d["num_parameters"],
                    "model_config": d["config"]["model"],
                    "train_config": d["config"]["train"],
                    "dataset": d["dataset"],
                    "tokenizer": d["tokenizer"],
                    "status": d["status"],
                    "best_val_loss": d["status"].get("best_val_loss"),
                    "evaluation": latest_eval and latest_eval["metrics"],
                    "benchmark": latest_bench
                    and {
                        "results": latest_bench["results"],
                        "memory": latest_bench["memory"],
                        "device": latest_bench["device"],
                    },
                }
            )
        tokenizers = {r["tokenizer"]["identity"] for r in rows}
        datasets = {r["dataset"]["identity"] for r in rows}
        val_texts = {r["dataset"].get("val_text_sha256") for r in rows}
        warnings = []
        if len(tokenizers) > 1:
            warnings.append(
                "Runs use different tokenizers: loss and perplexity are not comparable. Compare bits-per-byte instead."
            )
        if len(val_texts) > 1:
            warnings.append(
                "Runs were evaluated on different validation text: no metric is directly comparable."
            )
        devices = {r["benchmark"]["device"] for r in rows if r["benchmark"]}
        if len(devices) > 1:
            warnings.append(
                "Benchmarks ran on different devices: throughput numbers are not comparable."
            )
        return {
            "runs": rows,
            "comparable": {
                "loss_and_perplexity": len(tokenizers) == 1 and len(datasets) == 1,
                "bits_per_byte": len(val_texts) == 1,
                "benchmarks": len(devices) <= 1,
            },
            "warnings": warnings,
        }


def list_datasets() -> list[dict[str, Any]]:
    root = datasets_dir()
    out = []
    for meta_path in sorted(root.glob("*/meta.json")) if root.is_dir() else []:
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        size = sum(f.stat().st_size for f in meta_path.parent.iterdir() if f.is_file())
        out.append({**meta, "size_bytes": size})
    return out
