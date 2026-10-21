"""``tnsr`` command-line interface."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="tnsr",
    help="TNSR: train, evaluate and run a from-scratch Transformer language model.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
    add_completion=False,
)
data_app = typer.Typer(help="Prepare and inspect datasets.", no_args_is_help=True)
models_app = typer.Typer(help="List, inspect, export and import checkpoints.", no_args_is_help=True)
runs_app = typer.Typer(help="List training runs (experiments).", no_args_is_help=True)
app.add_typer(data_app, name="data")
app.add_typer(models_app, name="models")
app.add_typer(runs_app, name="runs")

console = Console()
err = Console(stderr=True)


def fail(message: str, code: int = 1) -> None:
    err.print(f"[bold red]error:[/] {message}")
    raise typer.Exit(code)


def emit_json(data: Any) -> None:
    sys.stdout.write(json.dumps(data, indent=2, default=str) + "\n")


def _resolve_model(ref: str | None) -> Path:
    from tnsr_core.checkpoints import (
        CheckpointError,
        discover_checkpoints,
        preferred_checkpoint,
        resolve_checkpoint,
    )

    if ref:
        try:
            return resolve_checkpoint(ref)
        except (CheckpointError, ValueError) as exc:
            fail(str(exc))
    choice = preferred_checkpoint(discover_checkpoints())
    if choice is None:
        fail("No checkpoints found. Train one first: tnsr train --config demo")
        raise AssertionError  # unreachable; fail() exits
    from tnsr_core.paths import tnsr_home

    return tnsr_home() / choice["id"]


# -- doctor --------------------------------------------------------------------------------


@app.command()
def doctor(
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable output")] = False,
    quick: Annotated[bool, typer.Option(help="Skip the model smoke test")] = False,
) -> None:
    """Check Python, PyTorch, accelerators, memory, disk and project state."""
    from tnsr_core.doctor import run_diagnostics

    report = run_diagnostics(quick=quick)
    if as_json:
        emit_json(report)
    else:
        table = Table(title=f"TNSR {report['tnsr_version']} diagnostics", show_lines=False)
        table.add_column("check")
        table.add_column("status")
        table.add_column("detail")
        colors = {"ok": "green", "warning": "yellow", "error": "red"}
        for c in report["checks"]:
            table.add_row(c["name"], f"[{colors[c['status']]}]{c['status']}[/]", c["detail"])
        console.print(table)
        console.print(
            f"Selected device: [bold]{report['selected_device']}[/]  (TNSR_HOME={report['tnsr_home']})"
        )
    if not report["ok"]:
        raise typer.Exit(1)


# -- data ----------------------------------------------------------------------------------


@data_app.command("prepare")
def data_prepare(
    name: Annotated[str, typer.Option(help="Dataset name (folder under datasets/)")] = "demo",
    source: Annotated[
        str, typer.Option(help="'synthetic' or a path to a .txt/.md file or directory")
    ] = "synthetic",
    tokenizer: Annotated[str, typer.Option(help="bpe or char")] = "bpe",
    vocab_size: Annotated[int, typer.Option(help="Target BPE vocabulary size")] = 2048,
    val_fraction: Annotated[float, typer.Option(help="Fraction of documents held out")] = 0.05,
    seed: Annotated[int, typer.Option(help="Split / generator seed")] = 0,
    docs: Annotated[int, typer.Option(help="Synthetic documents to generate")] = 20000,
    split_mode: Annotated[
        str, typer.Option(help="For files: paragraph, file or line")
    ] = "paragraph",
    overwrite: Annotated[
        bool, typer.Option(help="Replace an existing dataset of the same name")
    ] = False,
) -> None:
    """Build a dataset: split documents, fit the tokenizer on train only, write token files."""
    from tnsr_core.data import DatasetError, prepare_dataset, read_text_documents
    from tnsr_core.data.synthetic import PROVENANCE, generate_documents

    try:
        if source == "synthetic":
            documents = list(generate_documents(docs, seed))
            provenance = {**PROVENANCE, "n_docs": docs, "generator_seed": seed}
        else:
            documents, provenance = read_text_documents(source, split_mode)
        result = prepare_dataset(
            name,
            documents,
            tokenizer_kind=tokenizer,
            vocab_size=vocab_size,
            val_fraction=val_fraction,
            seed=seed,
            provenance=provenance,
            overwrite=overwrite,
            log=lambda m: console.print(f"[dim]{m}[/]"),
        )
    except (DatasetError, ValueError) as exc:
        fail(str(exc))
        return
    _print_dataset(result.meta)


def _print_dataset(meta: dict[str, Any]) -> None:
    s = meta["stats"]
    table = Table(title=f"dataset {meta['name']}", show_header=False)
    rows = [
        ("identity", meta["identity"]),
        ("source", meta.get("provenance", {}).get("source", "?")),
        ("license", meta.get("provenance", {}).get("license", "?")),
        (
            "tokenizer",
            f"{meta['tokenizer']['kind']} vocab={meta['tokenizer']['vocab_size']} ({meta['tokenizer']['identity']})",
        ),
        (
            "documents",
            f"{s['documents']:,} ({s['train_documents']:,} train / {s['val_documents']:,} val)",
        ),
        ("tokens", f"{s['train_tokens']:,} train / {s['val_tokens']:,} val"),
        ("bytes", f"{s['train_bytes']:,} train / {s['val_bytes']:,} val"),
        ("bytes/token", str(s["bytes_per_token"])),
        (
            "split",
            f"{meta['split']['method']} seed={meta['split']['seed']} val_fraction={meta['split']['val_fraction']}",
        ),
    ]
    for k, v in rows:
        table.add_row(k, v)
    console.print(table)


@data_app.command("inspect")
def data_inspect(
    target: Annotated[
        str, typer.Argument(help="Dataset name, or a text file/directory to analyse")
    ] = "demo",
    as_json: Annotated[bool, typer.Option("--json")] = False,
    sample: Annotated[
        int, typer.Option(help="Characters of decoded validation text to show")
    ] = 300,
) -> None:
    """Show statistics for a prepared dataset or a raw corpus."""
    from tnsr_core.data import DatasetError, TokenDataset, read_text_documents

    path = Path(target)
    if path.exists() and not (path / "meta.json").exists():
        try:
            documents, info = read_text_documents(path)
        except DatasetError as exc:
            fail(str(exc))
            return
        chars = sum(len(d) for d in documents)
        report = {
            **info,
            "documents": len(documents),
            "characters": chars,
            "unique_characters": len(set("".join(documents))),
            "mean_doc_chars": round(chars / max(1, len(documents)), 1),
        }
        emit_json(report) if as_json else console.print_json(json.dumps(report))
        return
    try:
        ds = TokenDataset(target)
    except DatasetError as exc:
        fail(str(exc))
        return
    if as_json:
        emit_json(ds.meta)
        return
    _print_dataset(ds.meta)
    if sample:
        text = ds.tokenizer.decode(ds.splits["val"][:sample].tolist())
        console.print("[bold]validation sample:[/]")
        console.print(text[:sample], markup=False)


@data_app.command("list")
def data_list() -> None:
    """List prepared datasets."""
    from tnsr_core.paths import datasets_dir

    table = Table("name", "identity", "tokenizer", "train tokens", "val tokens")
    for meta_path in sorted(datasets_dir().glob("*/meta.json")) if datasets_dir().is_dir() else []:
        meta = json.loads(meta_path.read_text())
        s = meta["stats"]
        table.add_row(
            meta["name"],
            meta["identity"],
            f"{meta['tokenizer']['kind']}/{meta['tokenizer']['vocab_size']}",
            f"{s['train_tokens']:,}",
            f"{s['val_tokens']:,}",
        )
    console.print(table)


# -- training ------------------------------------------------------------------------------


@app.command()
def train(
    config: Annotated[
        str, typer.Option("--config", "-c", help="Preset name or TOML path")
    ] = "demo",
    name: Annotated[str | None, typer.Option(help="Run name (defaults to the config name)")] = None,
    override: Annotated[
        list[str] | None, typer.Option("--set", help="Override, e.g. --set train.max_steps=500")
    ] = None,
    resume: Annotated[
        str | None, typer.Option(help="Resume an existing run id from its latest checkpoint")
    ] = None,
    max_steps: Annotated[
        int | None, typer.Option(help="Override max_steps (also extends a resumed run)")
    ] = None,
    device: Annotated[str | None, typer.Option(help="auto, cpu, cuda or mps")] = None,
    smoke: Annotated[bool, typer.Option(help="Fast smoke test: tiny step budget")] = False,
    run_id_file: Annotated[
        Path | None, typer.Option(hidden=True, help="Write the run id here (used by the server)")
    ] = None,
) -> None:
    """Train a model from scratch, or resume an interrupted run."""
    from tnsr_core.config import ConfigError, load_config, parse_override
    from tnsr_core.data import DatasetError
    from tnsr_core.training import Trainer, TrainingError, resolve_run

    log = console.print
    try:
        if resume:
            run_dir = resolve_run(resume)
            trainer = Trainer.resume_run(run_dir, log=log, max_steps=max_steps, device=device)
        else:
            overrides = dict(parse_override(o) for o in (override or []))
            if name:
                overrides["name"] = name
            if max_steps is not None:
                overrides["train.max_steps"] = max_steps
            if device:
                overrides["train.device"] = device
            if smoke:
                overrides.setdefault("train.max_steps", 20)
                overrides.update(
                    {"train.eval_interval": 10, "train.eval_iters": 2, "train.log_interval": 5}
                )
                overrides.setdefault("name", "smoke")
            cfg = load_config(config, overrides)
            trainer = Trainer.create(cfg, log=log)
            log(f"Run directory: {trainer.run_dir}")
        if run_id_file is not None:
            run_id_file.write_text(trainer.run_dir.name)
        trainer.install_signal_handlers()
        result = trainer.train()
    except (ConfigError, DatasetError, TrainingError, FileNotFoundError, ValueError) as exc:
        fail(str(exc))
        return
    if result.status == "interrupted":
        raise typer.Exit(130)


# -- inference -----------------------------------------------------------------------------


@app.command()
def generate(
    model: Annotated[
        str | None, typer.Argument(help="Checkpoint id, run id or path (default: newest best)")
    ] = None,
    prompt: Annotated[str, typer.Option("--prompt", "-p")] = "Once upon a time",
    max_new_tokens: Annotated[int, typer.Option()] = 200,
    temperature: Annotated[float, typer.Option()] = 0.8,
    top_k: Annotated[int, typer.Option()] = 40,
    top_p: Annotated[float, typer.Option()] = 0.95,
    repetition_penalty: Annotated[float, typer.Option()] = 1.0,
    deterministic: Annotated[bool, typer.Option(help="Greedy decoding")] = False,
    seed: Annotated[int | None, typer.Option(help="Seed for reproducible sampling")] = None,
    stop: Annotated[list[str] | None, typer.Option(help="Stop string (repeatable)")] = None,
    chat: Annotated[
        bool, typer.Option(help="Wrap the prompt as 'User: ...\\nTNSR:' and stop at the next turn")
    ] = False,
    device: Annotated[str, typer.Option()] = "auto",
    stats: Annotated[bool, typer.Option(help="Print timing statistics")] = True,
) -> None:
    """Generate text with a trained checkpoint (streams tokens as they are produced)."""
    from tnsr_core.checkpoints import CheckpointError
    from tnsr_core.inference import GenerationConfig, GenerationConfigError, InferenceEngine

    path = _resolve_model(model)
    try:
        engine = InferenceEngine.from_checkpoint(path, device)
        stops = list(stop or [])
        text = prompt
        if chat:
            text = f"User: {prompt}\nTNSR:"
            stops.append("\nUser:")
        cfg = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            deterministic=deterministic,
            seed=seed,
            stop=stops,
        ).validate(engine.tokenizer.vocab_size)
    except (CheckpointError, GenerationConfigError) as exc:
        fail(str(exc))
        return
    err.print(f"[dim]model {engine.model_id} on {engine.device}[/]")
    if not chat:
        sys.stdout.write(prompt)
    final = None
    for event in engine.stream(text, cfg):
        if event.kind == "token":
            sys.stdout.write(event.text)
            sys.stdout.flush()
        else:
            final = event.stats
    sys.stdout.write("\n")
    if stats and final:
        err.print(
            f"[dim]{final.generated_tokens} tokens | finish={final.finish_reason} | "
            f"TTFT {final.time_to_first_token_ms} ms | {final.decode_tokens_per_sec} tok/s[/]"
        )


@app.command()
def evaluate(
    model: Annotated[str | None, typer.Argument(help="Checkpoint id, run id or path")] = None,
    dataset: Annotated[
        str | None, typer.Option(help="Dataset (default: the checkpoint's own)")
    ] = None,
    max_tokens: Annotated[int, typer.Option(help="Cap on validation tokens scored")] = 200_000,
    probes: Annotated[
        bool, typer.Option(help="Also score question/answer probes (exact match)")
    ] = True,
    device: Annotated[str, typer.Option()] = "auto",
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Compute validation loss, perplexity, bits-per-byte, probe accuracy and sample generations."""
    from tnsr_core.checkpoints import CheckpointError
    from tnsr_core.data import DatasetError
    from tnsr_core.evaluation import evaluate_checkpoint

    path = _resolve_model(model)
    try:
        result = evaluate_checkpoint(
            path, dataset=dataset, device=device, max_tokens=max_tokens, probes=probes
        )
    except (CheckpointError, DatasetError, ValueError) as exc:
        fail(str(exc))
        return
    if as_json:
        emit_json(result)
        return
    m = result["metrics"]
    console.print(
        f"[bold]{result['checkpoint_id']}[/] on {result['dataset']['identity']} (val split)"
    )
    console.print(
        f"val loss {m['val_loss']:.4f} | perplexity {m['perplexity']:.3f} | bits/byte {m['bits_per_byte']:.4f} | "
        f"{m['tokens_evaluated']:,} tokens | {result['seconds']}s"
    )
    if result.get("probes"):
        p = result["probes"]
        cats = ", ".join(f"{c} {v['correct']}/{v['total']}" for c, v in p["by_category"].items())
        console.print(f"probe accuracy {p['accuracy']:.1%} ({p['total']} questions): {cats}")
    for s in result["samples"]:
        console.print(
            f"[cyan]{s['prompt']!r}[/] -> {s['completion']!r}", markup=True, highlight=False
        )
    console.print(f"[dim]saved {result.get('path')}[/]")


@app.command()
def benchmark(
    model: Annotated[str | None, typer.Argument(help="Checkpoint id, run id or path")] = None,
    device: Annotated[str, typer.Option()] = "auto",
    new_tokens: Annotated[int, typer.Option()] = 64,
    repeats: Annotated[int, typer.Option()] = 3,
    no_cache_compare: Annotated[
        bool, typer.Option(help="Also benchmark without the KV cache")
    ] = True,
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Measure time-to-first-token, decode throughput and peak memory."""
    from tnsr_core.checkpoints import CheckpointError
    from tnsr_core.evaluation import benchmark_checkpoint

    path = _resolve_model(model)
    try:
        result = benchmark_checkpoint(
            path,
            device=device,
            new_tokens=new_tokens,
            repeats=repeats,
            compare_no_cache=no_cache_compare,
        )
    except CheckpointError as exc:
        fail(str(exc))
        return
    if as_json:
        emit_json(result)
        return
    table = Table(
        title=f"benchmark {result['checkpoint_id']} on {result['device']} ({result['threads']} threads)"
    )
    for col in ("prompt tok", "new tok", "KV cache", "TTFT ms", "decode tok/s", "total ms"):
        table.add_column(col)
    for r in result["results"]:
        table.add_row(
            str(r["prompt_tokens"]),
            str(r["new_tokens"]),
            "yes" if r["kv_cache"] else "no",
            str(r["ttft_ms_median"]),
            str(r["decode_tokens_per_sec_median"]),
            str(r["total_ms_median"]),
        )
    console.print(table)
    console.print(f"memory: {result['memory']}")
    console.print(f"[dim]saved {result.get('path')}[/]")


# -- models / runs -------------------------------------------------------------------------


@models_app.command("list")
def models_list(as_json: Annotated[bool, typer.Option("--json")] = False) -> None:
    """List discovered checkpoints with real metadata."""
    from tnsr_core.checkpoints import discover_checkpoints

    found = discover_checkpoints()
    if as_json:
        emit_json(found)
        return
    if not found:
        console.print("No checkpoints yet. Train one with: tnsr train --config demo")
        return
    table = Table(
        "id", "step", "params", "layers/heads/dim", "ctx", "tokenizer", "val loss", "flags"
    )
    for c in found:
        mc = c["model_config"] or {}
        val = (c["metrics"] or {}).get("val_loss")
        flags = ",".join(f for f, on in (("best", c["is_best"]), ("latest", c["is_latest"])) if on)
        table.add_row(
            c["id"],
            str(c["step"]),
            f"{(c['num_parameters'] or 0) / 1e6:.2f}M",
            f"{mc.get('n_layer')}/{mc.get('n_head')}/{mc.get('n_embd')}",
            str(mc.get("block_size")),
            f"{c['tokenizer']['kind']}/{c['tokenizer']['vocab_size']}",
            f"{val:.4f}" if val is not None else "-",
            flags,
        )
    console.print(table)


@models_app.command("show")
def models_show(model: Annotated[str | None, typer.Argument()] = None) -> None:
    """Show full metadata for one checkpoint."""
    from tnsr_core.checkpoints import describe_checkpoint

    emit_json(describe_checkpoint(_resolve_model(model)))


@models_app.command("export")
def models_export(
    model: Annotated[str, typer.Argument(help="Checkpoint id, run id or path")],
    output: Annotated[Path, typer.Argument(help="Output .tar.gz path")],
    include_train_state: Annotated[
        bool, typer.Option(help="Include optimizer state for resuming")
    ] = False,
) -> None:
    """Package a checkpoint as a portable archive."""
    from tnsr_core.checkpoints import CheckpointError, export_checkpoint

    try:
        out = export_checkpoint(_resolve_model(model), output, include_train_state)
    except CheckpointError as exc:
        fail(str(exc))
        return
    console.print(f"Exported to {out} ({out.stat().st_size / 1e6:.1f} MB)")


@models_app.command("import")
def models_import(
    archive: Annotated[Path, typer.Argument(help="Archive created by tnsr models export")],
    name: Annotated[str, typer.Option(help="Name under checkpoints/")],
) -> None:
    """Import a checkpoint archive into checkpoints/<name> (validated before use)."""
    from tnsr_core.checkpoints import CheckpointError, import_checkpoint

    try:
        dest = import_checkpoint(archive, name)
    except (CheckpointError, ValueError, OSError) as exc:
        fail(str(exc))
        return
    console.print(f"Imported to {dest}")


@runs_app.command("list")
def runs_list(as_json: Annotated[bool, typer.Option("--json")] = False) -> None:
    """List training runs with their status."""
    from tnsr_core.training import list_runs

    runs = list_runs()
    if as_json:
        emit_json(runs)
        return
    table = Table("id", "state", "step", "params", "best val loss", "dataset", "created")
    for r in runs:
        st = r.get("status", {})
        best = st.get("best_val_loss")
        table.add_row(
            r["id"],
            st.get("state", "?"),
            f"{st.get('step', 0)}/{st.get('max_steps', '?')}",
            f"{r['num_parameters'] / 1e6:.2f}M",
            f"{best:.4f}" if best is not None else "-",
            r["dataset"]["name"],
            r["created_at"],
        )
    console.print(table)


# -- app -------------------------------------------------------------------------------------


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="Bind address (localhost by default)")] = "127.0.0.1",
    port: Annotated[int, typer.Option()] = 8000,
    reload: Annotated[bool, typer.Option(help="Auto-reload on code changes (development)")] = False,
) -> None:
    """Start the TNSR backend API (also serves the built Studio UI if present)."""
    import uvicorn

    if host not in {"127.0.0.1", "localhost", "::1"}:
        err.print(
            f"[yellow]warning:[/] binding to {host} exposes TNSR beyond this machine; it has no authentication."
        )
    if not _port_free(host, port):
        alt = next((p for p in range(port + 10, port + 100, 10) if _port_free(host, p)), None)
        hint = f" Try: tnsr serve --port {alt}" if alt else ""
        fail(
            f"Port {port} on {host} is already in use (another TNSR server or app may be "
            f"running; `lsof -nP -iTCP:{port} -sTCP:LISTEN` shows which).{hint}"
        )
    uvicorn.run("tnsr_server.app:create_app", factory=True, host=host, port=port, reload=reload)


def _port_free(host: str, port: int) -> bool:
    import socket

    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as sock:
        # Same option uvicorn uses, so a port in TIME_WAIT isn't misreported as busy.
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


@app.command()
def version() -> None:
    """Print the TNSR version."""
    from tnsr_core import __version__

    console.print(__version__)


if __name__ == "__main__":
    app()
