from __future__ import annotations

import os
from datetime import date

import pytest

from sota.config import RiskLimits, Settings
from sota.data.synthetic import SyntheticChainSource
from sota.models import Chain


@pytest.fixture(autouse=True)
def _no_sota_env(monkeypatch):
    """Tests must not pick up SOTA_* variables exported in the developer's shell."""
    for key in list(os.environ):
        if key.startswith("SOTA_"):
            monkeypatch.delenv(key, raising=False)


@pytest.fixture(scope="session")
def source() -> SyntheticChainSource:
    return SyntheticChainSource(["AAA", "BBB"], date(2024, 1, 1), days=140, seed=7)


@pytest.fixture(scope="session")
def asof(source: SyntheticChainSource) -> date:
    return source.start


@pytest.fixture()
def chain(source: SyntheticChainSource, asof: date) -> Chain:
    chain = source.get_chain("AAA", asof)
    assert chain is not None
    chain.ensure_greeks()
    return chain


@pytest.fixture()
def limits() -> RiskLimits:
    return RiskLimits()


@pytest.fixture()
def settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        universe=["AAA", "BBB"],
        data_dir=tmp_path / "data",
        journal_path=tmp_path / "journal.sqlite",
    )
