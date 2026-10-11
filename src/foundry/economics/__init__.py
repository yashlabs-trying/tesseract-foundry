from decimal import Decimal, ROUND_HALF_UP
from typing import Optional
from ..models import (
    EconomicEvaluation, CostAssumptions, DemandAssumptions,
    SensitivityAnalysis, DeploymentCandidate, WorkloadProfile,
    HardwareProfile, CostBasis, CapacityType,
)
from ..hardware import lookup_gpu


def round_decimal(d: Decimal, places: int = 6) -> Decimal:
    return d.quantize(Decimal(10) ** -places, rounding=ROUND_HALF_UP)


def evaluate(
    candidate: DeploymentCandidate,
    workload: WorkloadProfile,
    hardware: HardwareProfile,
    input_price_per_million: Optional[Decimal] = None,
    output_price_per_million: Optional[Decimal] = None,
    cost_basis: CostBasis = CostBasis.avoidable_cash,
    gpu_hourly_override: Optional[Decimal] = None,
) -> EconomicEvaluation:
    scenario_id = f"scenario-{candidate.candidate_id}"

    hourly_cost = gpu_hourly_override if gpu_hourly_override is not None else hardware.cost_per_gpu_hour_usd
    if hourly_cost == Decimal("0"):
        hw_spec = lookup_gpu(hardware.gpu_model)
        if hw_spec:
            vram = hw_spec.get("vram", 80)
            base = 3.50 if vram >= 80 else 2.50 if vram >= 40 else 0.80
        else:
            base = 3.50
        hourly_cost = Decimal(str(base))

    total_hourly = hourly_cost * candidate.gpu_count

    peak_hours = workload.peak_hours_per_day
    idle_hours = workload.idle_hours_per_day
    off_hours = 24 - peak_hours - idle_hours
    peak_utilization = 1.0
    idle_utilization = 0.1
    off_utilization = 0.5

    effective_hours = peak_hours * peak_utilization + idle_hours * idle_utilization + max(0, off_hours) * off_utilization
    daily_gpu_cost = total_hourly * Decimal(str(effective_hours))
    monthly_gpu_cost = daily_gpu_cost * Decimal("30")

    overhead_pct = Decimal("5")
    overhead_cost = monthly_gpu_cost * overhead_pct / Decimal("100")

    power_watts = 350 * candidate.gpu_count
    daily_energy_kwh = Decimal(str(power_watts * effective_hours)) / Decimal("1000")
    monthly_energy_kwh = daily_energy_kwh * Decimal("30")
    energy_cost = monthly_energy_kwh * Decimal("0.10")

    total_cost = monthly_gpu_cost + overhead_cost + energy_cost

    avg_input = Decimal(str(workload.input_tokens.mean))
    avg_output = Decimal(str(workload.output_tokens.mean))
    monthly_requests = Decimal(str(int(workload.request_rate_rps * 3600 * 24 * 30)))
    total_input_tokens = monthly_requests * avg_input
    total_output_tokens = monthly_requests * avg_output

    inp_price = input_price_per_million or Decimal("0.15")
    out_price = output_price_per_million or Decimal("0.60")
    revenue = (total_input_tokens / Decimal("1_000_000")) * inp_price + (total_output_tokens / Decimal("1_000_000")) * out_price

    margin = revenue - total_cost
    margin_pct_val = float((margin / revenue * 100) if revenue > 0 else 0.0)

    cost_per_mil_in = (total_cost * Decimal("1_000_000") / total_input_tokens) if total_input_tokens > 0 else Decimal("0")
    cost_per_mil_out = (total_cost * Decimal("1_000_000") / total_output_tokens) if total_output_tokens > 0 else Decimal("0")
    cost_per_req = total_cost / monthly_requests if monthly_requests > 0 else Decimal("0")
    rev_per_req = revenue / monthly_requests if monthly_requests > 0 else Decimal("0")

    be_requests = None
    if rev_per_req > 0:
        be_requests = float(total_cost / rev_per_req)

    utilization = min(100.0, float(monthly_requests / Decimal(str(candidate.max_batch_size * 100 * 3600 * 24 * 30))) * 100)

    cost = CostAssumptions(
        cost_basis=cost_basis,
        gpu_hourly_cost=total_hourly,
        attributable_overhead_pct=overhead_pct,
        energy_cost_per_kwh=Decimal("0.10"),
        estimated_power_draw_watts=power_watts,
    )

    demand = DemandAssumptions(
        monthly_request_volume=int(monthly_requests),
        avg_input_tokens_per_request=float(avg_input),
        avg_output_tokens_per_request=float(avg_output),
    )

    sensitivity = SensitivityAnalysis()
    sens_results = {"demand": [], "gpu_cost": []}
    for mult in sensitivity.demand_range:
        adj_revenue = revenue * Decimal(str(mult))
        sens_results["demand"].append(adj_revenue - total_cost)
    for mult in sensitivity.gpu_cost_range:
        adj_cost = monthly_gpu_cost * Decimal(str(mult)) + overhead_cost + energy_cost
        sens_results["gpu_cost"].append(revenue - adj_cost)

    return EconomicEvaluation(
        candidate_id=candidate.candidate_id,
        scenario_id=scenario_id,
        cost_assumptions=cost,
        demand_assumptions=demand,
        allocated_gpu_cost_monthly=round_decimal(monthly_gpu_cost, 2),
        attributable_variable_cost_monthly=round_decimal(overhead_cost + energy_cost, 2),
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


def evaluate_for_providers(
    candidate: DeploymentCandidate,
    workload: WorkloadProfile,
    provider_hardware: list[HardwareProfile],
    input_price_per_million: Optional[Decimal] = None,
    output_price_per_million: Optional[Decimal] = None,
) -> dict[str, EconomicEvaluation]:
    results = {}
    for hw in provider_hardware:
        results[hw.hardware_id] = evaluate(
            candidate, workload, hw,
            input_price_per_million, output_price_per_million,
        )
    return results
