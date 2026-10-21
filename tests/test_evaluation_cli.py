import json
import math

from typer.testing import CliRunner

from tnsr_core.checkpoints import describe_checkpoint
from tnsr_core.cli import app
from tnsr_core.evaluation import benchmark_checkpoint, evaluate_checkpoint

runner = CliRunner()


def test_evaluate_reports_consistent_metrics(trained):
    result = evaluate_checkpoint(trained, device="cpu", sample_prompts=["Once"], sample_tokens=5)
    m = result["metrics"]
    assert math.isclose(m["perplexity"], math.exp(m["val_loss"]), rel_tol=1e-3)
    assert 0 < m["bits_per_byte"] < 10
    assert m["covered_full_split"]
    assert result["dataset"]["split"] == "val"
    assert len(result["samples"]) == 1
    # stored next to the run and surfaced in the checkpoint description
    assert describe_checkpoint(trained)["evaluation"]["metrics"] == m


def test_benchmark_measures_ttft_and_throughput(trained):
    result = benchmark_checkpoint(
        trained, device="cpu", prompt_tokens=(8,), new_tokens=8, repeats=1
    )
    rows = result["results"]
    assert {r["kv_cache"] for r in rows} == {True, False}
    for r in rows:
        assert r["ttft_ms_median"] > 0 and r["decode_tokens_per_sec_median"] > 0
    assert result["memory"]["process_peak_rss_mb"] > 0
    assert describe_checkpoint(trained)["benchmark"]["results"] == rows


def test_cli_doctor_json():
    res = runner.invoke(app, ["doctor", "--json", "--quick"])
    data = json.loads(res.stdout)
    assert {c["name"] for c in data["checks"]} >= {"python", "torch", "device"}


def test_cli_end_to_end(home):
    res = runner.invoke(app, ["data", "prepare", "--docs", "300", "--vocab-size", "300"])
    assert res.exit_code == 0, res.output
    res = runner.invoke(app, ["data", "inspect", "demo", "--json"])
    assert json.loads(res.stdout)["stats"]["documents"] == 300
    res = runner.invoke(app, ["train", "--config", "tiny", "--smoke"])
    assert res.exit_code == 0, res.output
    res = runner.invoke(app, ["models", "list", "--json"])
    models = json.loads(res.stdout)
    assert models and models[0]["step"] == 20
    res = runner.invoke(
        app, ["generate", "-p", "Hello", "--max-new-tokens", "5", "--deterministic", "--no-stats"]
    )
    assert res.exit_code == 0 and res.stdout.startswith("Hello")
    res = runner.invoke(app, ["runs", "list", "--json"])
    assert json.loads(res.stdout)[0]["status"]["state"] == "completed"


def test_cli_errors_are_friendly(home):
    res = runner.invoke(app, ["generate", "-p", "x"])
    assert res.exit_code == 1 and "No checkpoints found" in res.output
    res = runner.invoke(app, ["train", "--config", "tiny"])
    assert res.exit_code == 1 and "tnsr data prepare" in res.output
    res = runner.invoke(app, ["train", "--config", "tiny", "--set", "train.batch_size=0"])
    assert res.exit_code == 1 and "batch_size" in res.output
