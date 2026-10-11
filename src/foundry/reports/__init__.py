from ..models import ServiceRecommendation, DeploymentCandidate
from typing import Optional
from datetime import datetime

def _evidence_label(et):
    labels = {
        "synthetic": "SYNTHETIC (not validated against real hardware)",
        "estimated": "ESTIMATED (model-based prediction, low confidence)",
        "measured": "MEASURED (actual benchmark data)",
        "validated": "VALIDATED (measured + passed qualification checks)",
    }
    return labels.get(et.value if hasattr(et, "value") else str(et), "UNKNOWN")

NL = chr(10)

def generate_qualification_report(rec, candidates=None, economics=None, evidence_map=None):
    lines = []
    lines.append("=" * 72)
    lines.append("TESSERACT FOUNDRY - COMMERCIAL SERVICE QUALIFICATION REPORT")
    lines.append("=" * 72)
    lines.append("")
    lines.append("Report ID:      " + str(rec.recommendation_id))
    lines.append("Status:         " + str(rec.approval_status))
    lines.append("Generated:      " + datetime.utcnow().isoformat())
    lines.append("")
    if rec.recommended_candidate_id in ("NO_FEASIBLE_DEPLOYMENT", "NO_VERIFIED_DEPLOYMENT"):
        lines.append("--- NO FEASIBLE SERVICE ---")
        lines.append("No candidate configuration satisfied all qualification constraints.")
        lines.append("")
        if rec.rejected_candidates:
            lines.append("Rejected Candidates:")
            for r in rec.rejected_candidates:
                lines.append("  - " + r.candidate_id + ": " + r.reason)
        lines.append("")
        lines.append("Recommendation: Do not launch this model under the specified conditions.")
        lines.append("")
        lines.append("=" * 72)
        return NL.join(lines)
    cand = candidates.get(rec.recommended_candidate_id) if candidates else None
    econ = rec.economic_assumptions or (economics.get(rec.recommended_candidate_id) if economics else None)
    ev = rec.expected_performance or (evidence_map.get(rec.recommended_candidate_id) if evidence_map else None)
    if cand:
        lines.append("SELECTED DEPLOYMENT CONFIGURATION")
        lines.append("-" * 40)
        lines.append("  GPU:             " + str(cand.hardware_id))
        lines.append("  Engine:          " + cand.serving_engine.value + " v" + cand.engine_version)
        lines.append("  Precision:       " + cand.precision.value)
        lines.append("  Tensor Parallel: " + str(cand.tensor_parallelism))
        lines.append("  GPU Count:       " + str(cand.gpu_count))
        lines.append("  Batch Size:      " + str(cand.max_batch_size))
        lines.append("  Context Length:  " + str(cand.max_context_length))
        lines.append("  Expected VRAM:   " + f"{cand.expected_vram_gb:.1f} GB")
        lines.append("")
    if ev:
        lines.append("EXPECTED PERFORMANCE")
        lines.append("-" * 40)
        lines.append("  Evidence Type:   " + _evidence_label(ev.evidence_type))
        lines.append("  Benchmark Tool:  " + ev.benchmark_tool + " v" + ev.benchmark_version)
        lines.append("  Sample Count:    " + str(ev.sample_count))
        lines.append("")
        if ev.ttft_ms:
            lines.append("  TTFT (ms):       P50=" + str(ev.ttft_ms.p50) + "  P95=" + str(ev.ttft_ms.p95) + "  P99=" + str(ev.ttft_ms.p99))
        if ev.itl_ms:
            lines.append("  ITL (ms):        P50=" + str(ev.itl_ms.p50) + "  P95=" + str(ev.itl_ms.p95))
        if ev.request_latency_ms:
            lines.append("  Req Latency (ms): P50=" + str(ev.request_latency_ms.p50) + "  P95=" + str(ev.request_latency_ms.p95))
        if ev.throughput and ev.throughput.completed_requests_per_second:
            lines.append("  Throughput:      " + str(ev.throughput.completed_requests_per_second) + " req/s")
        lines.append("")
    if econ:
        lines.append("ECONOMICS")
        lines.append("-" * 40)
        lines.append("  GPU Monthly Cost:     $" + f"{float(econ.allocated_gpu_cost_monthly):>10,.2f}")
        lines.append("  Total Monthly Cost:   $" + f"{float(econ.total_cost_monthly):>10,.2f}")
        lines.append("  Est. Monthly Revenue: $" + f"{float(econ.estimated_revenue_monthly):>10,.2f}")
        lines.append("  Contribution Margin:  $" + f"{float(econ.contribution_margin):>10,.2f}  ({econ.margin_pct:.1f}%)")
        lines.append("  Cost per 1M Input:    $" + f"{float(econ.cost_per_million_input_tokens):.4f}")
        lines.append("  Cost per 1M Output:   $" + f"{float(econ.cost_per_million_output_tokens):.4f}")
        if econ.break_even_requests_per_month:
            lines.append("  Break-Even Volume:    " + f"{econ.break_even_requests_per_month:,.0f} req/month")
        lines.append("")
    if economics and rec.feasible_candidates:
        lines.append("ALTERNATIVE CONFIGURATIONS")
        lines.append("-" * 40)
        for cid in rec.feasible_candidates:
            if cid == rec.recommended_candidate_id:
                continue
            ae = economics.get(cid)
            if ae:
                lines.append("  " + cid + ": $" + f"{float(ae.total_cost_monthly):,.2f}/mo | Margin: {ae.margin_pct:.1f}%")
    if rec.rejected_candidates:
        lines.append("REJECTED CONFIGURATIONS")
        lines.append("-" * 40)
        for r in rec.rejected_candidates:
            lines.append("  " + r.candidate_id + ": " + r.reason)
    lines.append("")
    lines.append("-" * 72)
    lines.append("COMMERCIAL VIABILITY ASSESSMENT")
    lines.append("-" * 72)
    if econ:
        if econ.margin_pct >= 20:
            lines.append("  STATUS: VIABLE - Positive contribution margin with adequate buffer.")
        elif econ.margin_pct >= 0:
            lines.append("  STATUS: MARGINAL - Positive margin but below 20% target.")
        else:
            lines.append("  STATUS: NOT VIABLE - Negative contribution margin.")
    lines.append("")
    lines.append("RECOMMENDATION")
    lines.append("-" * 40)
    if ev and hasattr(ev.evidence_type, "value") and ev.evidence_type.value == "synthetic":
        lines.append("  WARNING: All performance data is synthetic. Real GPU benchmarks required.")
    lines.append("  Deployment requires explicit approval action (autoDeploy=false).")
    lines.append("")
    lines.append("-" * 72)
    lines.append("Report generated by Tesseract Foundry")
    lines.append("=" * 72)
    return NL.join(lines)

def generate_markdown_report(rec, candidates=None):
    return generate_qualification_report(rec, candidates)

def generate_json_report(rec):
    return rec.model_dump()