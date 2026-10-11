from decimal import Decimal, ROUND_HALF_UP
from typing import Optional
from ..models import (
    EconomicEvaluation, CostAssumptions, DemandAssumptions,
    SensitivityAnalysis, DeploymentCandidate, WorkloadProfile,
    CostBasis,
)
from ..hardware import lookup_gpu

def round_decimal(d: Decimal, places: int = 6) -> Decimal:
    return d.quantize(Decimal(10) ** -places, rounding=ROUND_HALF_UP)

def evaluate(
    candidate: DeploymentCandidate,
    workload: WorkloadProfile,
    input_price_per_million: Optional[Decimal] = None,
    output_price_per_million: Optional[Decimal] = None,
    min_margin_pct: float = 20.0,
    cost_basis: CostBasis = CostBasis.avoidable_cash,
) -> EconomicEvaluation:
    scenario_id = f"scenario-{candidate.candidate_id}"
    hw_spec = lookup_gpu(candidate.hardware_id.replace("-1", "").replace("-4x", ""))
    base_hourly = float(candidate.gpu_count) * 3.50
    if hw_spec:
        vram = hw_spec.get("vram", 80)
        base_hourly = float(candidate.gpu_count) * (3.50 if vram >= 80 else 2.50 if vram >= 40 else 0.80)
    gpu_hourly = Decimal(str(base_hourly))
    overhead_pct = Decimal("5")
    energy_per_kwh = Decimal("0.10")
    power_watts = 350 * candidate.gpu_count

    cost = CostAssumptions(
        cost_basis=cost_basis,
        gpu_hourly_cost=gpu_hourly,
        attributable_overhead_pct=overhead_pct,
        energy_cost_per_kwh=energy_per_kwh,
        estimated_power_draw_watts=power_watts,
    )

    hours_per_month = Decimal("730")
    gpu_cost_monthly = gpu_hourly * hours_per_month
    energy_kwh_monthly = Decimal(power_watts * 730) / Decimal("1000")
    energy_cost_monthly = energy_kwh_monthly * energy_per_kwh
    overhead_cost = gpu_cost_monthly * overhead_pct / Decimal("100")
    total_cost = gpu_cost_monthly + energy_cost_monthly + overhead_cost

    avg_input = Decimal(str(workload.input_tokens.mean))
    avg_output = Decimal(str(workload.output_tokens.mean))
    monthly_requests = workload.request_rate_rps * 3600 * 730
    monthly_requests_d = Decimal(str(int(monthly_requests)))
    total_input_tokens = monthly_requests_d * avg_input
    total_output_tokens = monthly_requests_d * avg_output

    inp_price = input_price_per_million or Decimal("0.15")
    out_price = output_price_per_million or Decimal("0.60")
    revenue = (total_input_tokens / Decimal("1_000_000")) * inp_price + (total_output_tokens / Decimal("1_000_000")) * out_price

    margin = revenue - total_cost
    margin_pct_val = float((margin / revenue * 100) if revenue > 0 else 0.0)

    cost_per_mil_in = (total_cost * Decimal("1_000_000") / total_input_tokens) if total_input_tokens > 0 else Decimal("0")
    cost_per_mil_out = (total_cost * Decimal("1_000_000") / total_output_tokens) if total_output_tokens > 0 else Decimal("0")
    cost_per_req = total_cost / monthly_requests_d if monthly_requests_d > 0 else Decimal("0")
    rev_per_req = revenue / monthly_requests_d if monthly_requests_d > 0 else Decimal("0")

    be_requests = None
    if rev_per_req > 0 and cost_per_req > 0:
        be = total_cost / rev_per_req
        be_requests = float(be)

    utilization = min(100.0, (workload.request_rate_rps / (candidate.max_batch_size * 10)) * 100)

    sensitivity = SensitivityAnalysis()
    sens_results: dict[str, list[Decimal]] = {"demand": [], "gpu_cost": []}
    for mult in sensitivity.demand_range:
        adj_revenue = revenue * Decimal(str(mult))
        sens_results["demand"].append(adj_revenue - total_cost)
    for mult in sensitivity.gpu_cost_range:
        adj_cost = gpu_cost_monthly * Decimal(str(mult)) + energy_cost_monthly + overhead_cost
        sens_results["gpu_cost"].append(revenue - adj_cost)

    return EconomicEvaluation(
        candidate_id=candidate.candidate_id,
        scenario_id=scenario_id,
        cost_assumptions=cost,
        demand_assumptions=DemandAssumptions(
            monthly_request_volume=int(monthly_requests),
            avg_input_tokens_per_request=float(avg_input),
            avg_output_tokens_per_request=float(avg_output),
        ),
        allocated_gpu_cost_monthly=round_decimal(gpu_cost_monthly, 2),
        attributable_variable_cost_monthly=round_decimal(energy_cost_monthly + overhead_cost, 2),
        total_cost_monthly=round_decimal(total_cost, 2),
        estimated_revenue_monthly=round_decimal(revenue, 2),
        contribution_margin=round_decimal(margin, 2),
        margin_pct=round(margin_pct_val, 2),
        break_even_requests_per_month=be_requests,
        predicted_utilization_pct=round(utilization, 1),
        cost_per_million_input_tokens=round_decimal(cost_per_mil_in, 4),
        cost_per_million_output_tokens=round_decimal(cost_per_mil_out, 4),
        cost_per_request=round_decimal(cost_per_req, 6),
        revenue_per_request=round_decimal(rev_per_req, 6),
        sensitivity=sensitivity,
        sensitivity_results=sens_results,
        status="estimated",
    )
