"""Threshold rule (paper Table 8, App. G.2): one fixed threshold on one feature per family."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from sota.features import MarketFeatures
from sota.models import Side, Tenor
from sota.selectors.base import BaseSelector, PortfolioView, Proposal


@dataclass(frozen=True, slots=True)
class ThresholdRule:
    family: str
    feature: str  # iv_rv | bf_iv | sk_iv | abs_ret
    op: str  # "<=" or ">="
    level: float

    def hit(self, value: float) -> bool:
        if math.isnan(value):
            return False
        return value <= self.level if self.op == "<=" else value >= self.level


TABLE8_RULES: tuple[ThresholdRule, ...] = (
    ThresholdRule("ls", "iv_rv", "<=", 0.90),
    ThresholdRule("ib", "iv_rv", ">=", 1.20),
    ThresholdRule("ic", "iv_rv", ">=", 1.25),
    ThresholdRule("lg", "bf_iv", "<=", 0.0),
    ThresholdRule("bf", "bf_iv", "<=", 0.0),
    ThresholdRule("dg", "sk_iv", ">=", 0.15),
    ThresholdRule("cv", "sk_iv", "<=", 0.0),
    ThresholdRule("ol", "abs_ret", ">=", 0.03),
    ThresholdRule("dv", "abs_ret", ">=", 0.02),
)


def rule_inputs(f: MarketFeatures) -> dict[str, float]:
    iv = f.iv if not math.isnan(f.iv) and f.iv > 0 else float("nan")
    return {
        "iv_rv": f.iv_rv_ratio,
        "bf_iv": f.bf / iv,
        "sk_iv": f.sk / iv,
        "abs_ret": abs(f.ret),
    }


class ThresholdSelector(BaseSelector):
    name = "threshold"

    def __init__(
        self,
        tenor: Tenor = Tenor.T8_30,
        rules: Sequence[ThresholdRule] = TABLE8_RULES,
        participation_gate: bool = True,
    ) -> None:
        self.tenor = tenor
        self.rules = tuple(rules)
        self.participation_gate = participation_gate  # paper: trade only when ts < 0

    @staticmethod
    def _side(rule: ThresholdRule, f: MarketFeatures) -> Side:
        if rule.family in ("ol", "dv"):
            return Side.LONG if f.ret > 0 else Side.SHORT
        if rule.family == "cv":
            return Side.SHORT  # sk <= 0: calls rich relative to puts -> bear call spread
        return Side.LONG  # dg with sk >= 0.15: puts rich -> sell put / buy call

    def propose(
        self, features: dict[str, MarketFeatures], portfolio: PortfolioView
    ) -> list[Proposal]:
        out: list[Proposal] = []
        for u, f in features.items():
            if self.participation_gate and not (f.ts < 0):
                continue
            inputs = rule_inputs(f)
            open_families = {p.package.family for p in portfolio.open_positions(u)}
            for rule in self.rules:
                value = inputs[rule.feature]
                if rule.family in open_families or not rule.hit(value):
                    continue
                out.append(
                    Proposal(
                        u,
                        rule.family,
                        self._side(rule, f),
                        self.tenor,
                        {},
                        f"{rule.feature}={value:.3f} {rule.op} {rule.level}",
                    )
                )
        return out
