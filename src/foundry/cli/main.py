import typer
import yaml
import json
from pathlib import Path
from typing import Optional

from .. import __version__
from ..models import (
    ModelSpec, HardwareProfile, WorkloadProfile, Scenario,
    DeploymentCandidate, RankBy,
)
from ..intake import IntakeOrchestrator
from ..candidates import get_provider
from ..benchmarking import get_runner
from ..economics import evaluate
from ..qualification import QualificationEngine
from ..deployment import generate_manifest, manifest_to_yaml
from ..reports import generate_markdown_report
from ..schemas import export_performance_profile
from ..jobs import JobRunner
from decimal import Decimal

app = typer.Typer(name="foundry", help="Tesseract Foundry CLI")
intake = IntakeOrchestrator()
qualification = QualificationEngine()

def _load_yaml(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        typer.echo(f"Error: file not found: {path}", err=True)
        raise typer.Exit(1)
    with open(p) as f:
        return yaml.safe_load(f)

@app.command()
def version():
    typer.echo(f"Tesseract Foundry v{__version__}")

@app.command()
def validate(
    model_file: str = typer.Argument(..., help="Path to model YAML"),
):
    data = _load_yaml(model_file)
    model = ModelSpec(**data.get("model", data))
    hardware_data = data.get("hardware", [])
    hardware = [HardwareProfile(**h) if isinstance(h, dict) else h for h in hardware_data]
    results = intake.model_validator.validate(model, hardware)
    for r in [results]:
        if r.errors:
            typer.echo("VALIDATION FAILED:")
            for e in r.errors:
                typer.echo(f"  ? {e}")
        if r.warnings:
            typer.echo("Warnings:")
            for w in r.warnings:
                typer.echo(f"  ? {w}")
    if results.valid:
        typer.echo("? Model specification is valid")

@app.command()
def profile(
    scenario_file: str = typer.Argument(..., help="Path to scenario YAML"),
    mode: str = typer.Option("fixture", help="Profiling mode: fixture | conservative"),
    output: Optional[str] = typer.Option(None, help="Output file for candidates"),
):
    data = _load_yaml(scenario_file)
    model = ModelSpec(**data["model"])
    hardware = [HardwareProfile(**h) for h in data["hardware"]]
    workload = WorkloadProfile(**data["workload"])

    provider = get_provider(mode)
    candidates = provider.generate_candidates(model, hardware, workload)
    typer.echo(f"Generated {len(candidates)} candidates:")
    for c in candidates:
        typer.echo(f"  {c.candidate_id}: {c.serving_engine.value} TP={c.tensor_parallelism} GPU={c.gpu_count} VRAM={c.expected_vram_gb:.1f}GB")

    if output:
        with open(output, "w") as f:
            json.dump([c.model_dump() for c in candidates], f, indent=2, default=str)
        typer.echo(f"Candidates written to {output}")

@app.command()
def recommend(
    scenario_file: str = typer.Argument(..., help="Path to scenario YAML"),
    rank_by: str = typer.Option("lowest_cost", help="Ranking criterion"),
    output: Optional[str] = typer.Option(None, help="Output file for recommendation"),
):
    data = _load_yaml(scenario_file)
    model = ModelSpec(**data["model"])
    hardware = [HardwareProfile(**h) for h in data["hardware"]]
    workload = WorkloadProfile(**data["workload"])
    min_margin = float(data.get("min_contribution_margin_pct", 20.0))
    rank = RankBy(rank_by)

    val_results = intake.validate_scenario(model, hardware, workload)
    for vr in val_results:
        if not vr.valid:
            typer.echo("Validation failed:")
            for e in vr.errors:
                typer.echo(f"  ? {e}")
            raise typer.Exit(1)

    provider = get_provider("fixture")
    candidates = provider.generate_candidates(model, hardware, workload)
    runner = get_runner("fixture")

    evidence_map = {}
    economics_map = {}
    for c in candidates:
        evidence_map[c.candidate_id] = runner.run(c, workload)
        inp = Decimal(str(data.get("input_price_per_million"))) if data.get("input_price_per_million") else None
        outp = Decimal(str(data.get("output_price_per_million"))) if data.get("output_price_per_million") else None
        economics_map[c.candidate_id] = evaluate(c, workload, inp, outp, min_margin)

    rec = qualification.recommend(candidates, economics_map, workload, evidence_map, rank, min_margin)
    candidates_by_id = {c.candidate_id: c for c in candidates}

    report = generate_markdown_report(rec, candidates_by_id)
    typer.echo(report)

    if output:
        with open(output, "w") as f:
            f.write(report)
        typer.echo(f"\nReport written to {output}")

@app.command()
def economics(
    scenario_file: str = typer.Argument(..., help="Path to scenario YAML"),
):
    data = _load_yaml(scenario_file)
    model = ModelSpec(**data["model"])
    hardware = [HardwareProfile(**h) for h in data["hardware"]]
    workload = WorkloadProfile(**data["workload"])

    provider = get_provider("fixture")
    candidates = provider.generate_candidates(model, hardware, workload)

    for c in candidates:
        inp = Decimal(str(data.get("input_price_per_million", 0.15)))
        outp = Decimal(str(data.get("output_price_per_million", 0.60)))
        econ = evaluate(c, workload, inp, outp)
        typer.echo(f"\n--- {c.candidate_id} ---")
        typer.echo(f"  GPU Cost/Month:  ")
        typer.echo(f"  Total Cost/Month: ")
        typer.echo(f"  Revenue/Month:    ")
        typer.echo(f"  Margin:            ({econ.margin_pct:.1f}%)")
        typer.echo(f"  Cost/1M Input:    ")
        typer.echo(f"  Cost/1M Output:   ")

@app.command()
def report(
    rec_id: str = typer.Argument(..., help="Recommendation ID"),
    format: str = typer.Option("markdown", help="Report format: markdown | json"),
):
    from ..api import _recommendation_store, _candidate_store
    if rec_id not in _recommendation_store:
        typer.echo(f"Recommendation {rec_id} not found. Run 'foundry recommend' first.")
        raise typer.Exit(1)
    rec = _recommendation_store[rec_id]
    report = generate_markdown_report(rec, _candidate_store)
    typer.echo(report)

@app.command()
def manifest(
    rec_id: str = typer.Argument(..., help="Recommendation ID"),
    output: Optional[str] = typer.Option(None, help="Output file for manifest"),
):
    from ..api import _recommendation_store, _candidate_store
    if rec_id not in _recommendation_store:
        typer.echo(f"Recommendation {rec_id} not found.")
        raise typer.Exit(1)
    rec = _recommendation_store[rec_id]
    if rec.recommended_candidate_id == "NO_FEASIBLE_DEPLOYMENT":
        typer.echo("No feasible deployment to generate manifest for.")
        raise typer.Exit(1)
    candidate = _candidate_store.get(rec.recommended_candidate_id)
    if not candidate:
        typer.echo("Candidate not found.")
        raise typer.Exit(1)
    manifest = generate_manifest(candidate, rec)
    yaml_output = manifest_to_yaml(manifest)
    if output:
        with open(output, "w") as f:
            f.write(yaml_output)
        typer.echo(f"Manifest written to {output}")
    else:
        typer.echo(yaml_output)

@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Bind address"),
    port: int = typer.Option(8080, help="Listen port"),
    reload: bool = typer.Option(False, help="Auto-reload on changes"),
):
    import uvicorn
    typer.echo(f"Starting Tesseract Foundry API on {host}:{port}")
    uvicorn.run("foundry.api:app", host=host, port=port, reload=reload)

if __name__ == "__main__":
    app()
