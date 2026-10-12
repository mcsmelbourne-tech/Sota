"""Size resolver (paper App. E.2, Table 10): equal scenario-risk sizing under portfolio caps."""

from __future__ import annotations

import math
from dataclasses import dataclass

from sota.config import RiskLimits
from sota.models import MULTIPLIER, Package
from sota.risk import PortfolioRisk, package_scenario_risk


@dataclass(frozen=True, slots=True)
class SizeResult:
    units: int
    risk_per_unit: float  # dollars, scenario risk
    capital_per_unit: float  # dollars, max loss incl. collateral buffer
    binding: str  # which constraint set the size, or why it was rejected

    @property
    def accepted(self) -> bool:
        return self.units > 0


def capital_per_share(pkg: Package, buffer: float) -> float | None:
    if pkg.max_loss is None:
        return None
    return pkg.max_loss * (1.0 + buffer)


def resolve_size(
    pkg: Package, spot: float, atm_iv: float, limits: RiskLimits, risk: PortfolioRisk
) -> SizeResult:
    nav = risk.nav
    u = pkg.underlying
    risk_unit = package_scenario_risk(pkg, spot, atm_iv, limits.iv_shock) * MULTIPLIER
    if risk_unit <= 0.0:
        return SizeResult(0, 0.0, 0.0, "rejected: zero scenario risk")
    cap_share = capital_per_share(pkg, limits.collateral_buffer)
    if cap_share is None:
        return SizeResult(0, risk_unit, math.inf, "rejected: unbounded max loss")
    cap_unit = cap_share * MULTIPLIER

    if risk.open_positions >= limits.max_positions:
        return SizeResult(0, risk_unit, cap_unit, "rejected: portfolio position cap")
    if risk.positions_by_underlying.get(u, 0) >= limits.max_positions_per_underlying:
        return SizeResult(0, risk_unit, cap_unit, "rejected: per-underlying position cap")

    bounds: dict[str, float] = {
        "target risk": limits.target_risk_frac * nav / risk_unit,
        "capital cap": (limits.nav_frac_cap * nav / cap_unit) if cap_unit > 0 else math.inf,
        "underlying risk cap": (
            limits.risk_cap_per_underlying * nav - risk.scenario_risk_by_underlying.get(u, 0.0)
        )
        / risk_unit,
        "total risk cap": (limits.risk_cap_total * nav - risk.scenario_risk_total) / risk_unit,
    }
    if not pkg.hedged:
        dd_unit = abs(pkg.delta) * spot * MULTIPLIER
        if dd_unit > 0.0:
            bounds["net delta cap"] = (
                limits.net_delta_cap * nav - abs(risk.dollar_delta)
            ) / dd_unit

    binding, raw = min(bounds.items(), key=lambda kv: kv[1])
    units = max(int(math.floor(raw)), 0)
    label = binding if units > 0 else f"rejected: {binding}"
    return SizeResult(units, risk_unit, cap_unit, label)
