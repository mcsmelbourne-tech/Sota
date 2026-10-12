"""Wires settings into concrete data sources, selectors, brokers and runners."""

from __future__ import annotations

from sota.config import Mode, Settings
from sota.data.base import ChainSource
from sota.execution.base import Broker
from sota.execution.manual import ManualTicketBroker
from sota.journal import Journal
from sota.notify.base import Notifier
from sota.notify.channels import build_notifier
from sota.runner import Runner
from sota.selectors.base import Selector


def build_source(settings: Settings, synthetic: bool = False) -> ChainSource:
    if synthetic:
        from datetime import date

        from sota.data.synthetic import SyntheticChainSource

        return SyntheticChainSource(settings.universe, date(2024, 1, 1), 400)
    if settings.mode is Mode.BACKTEST:
        from sota.data.theta import ThetaDataSource

        return ThetaDataSource(settings.theta, settings.data_dir, rate=settings.risk_free_rate)
    from sota.data.alpaca import AlpacaChainSource

    return AlpacaChainSource(settings.alpaca, rate=settings.risk_free_rate)


def build_selector(settings: Settings, source: ChainSource, name: str | None = None) -> Selector:
    name = name or settings.selector
    if name == "threshold":
        from sota.selectors.threshold import ThresholdSelector

        return ThresholdSelector(participation_gate=settings.threshold_gate)
    if name == "llm":
        from sota.selectors.llm import LLMSelector

        return LLMSelector(settings.llm, settings.limits)
    if name == "oracle":
        from sota.selectors.oracle import HindsightOracle

        return HindsightOracle(source, limits=settings.limits)
    raise ValueError(f"unknown selector {name!r}")


def build_broker(
    settings: Settings, journal: Journal, notifier: Notifier
) -> Broker | ManualTicketBroker:
    if settings.mode is Mode.LIVE:
        series = journal.nav_series("live")
        nav = series[-1][1] if series else settings.initial_nav
        return ManualTicketBroker(journal, notifier, nav, settings.effective_costs)
    if settings.mode is Mode.PAPER:
        from sota.execution.alpaca_paper import AlpacaPaperBroker

        return AlpacaPaperBroker(settings.alpaca)
    raise ValueError("backtest mode has no broker; use sota.backtest.TradingEnv")


def build_runner(
    settings: Settings,
    journal: Journal | None = None,
    notifier: Notifier | None = None,
    synthetic: bool = False,
) -> Runner:
    journal = journal or Journal(settings.journal_path)
    notifier = notifier or build_notifier(settings.notify)
    source = build_source(settings, synthetic=synthetic)
    selector = build_selector(settings, source)
    broker = build_broker(settings, journal, notifier)
    return Runner(settings, source, selector, broker, journal, notifier)
