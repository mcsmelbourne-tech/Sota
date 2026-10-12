"""Application settings. Values come from defaults, `.env`, and `SOTA_*` environment variables.

Nested fields use `__`, e.g. `SOTA_ALPACA__API_KEY=...`, `SOTA_LIMITS__MAX_POSITIONS=8`.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_UNIVERSE = ["SPY", "AAPL", "AMZN", "GOOGL", "META", "MSFT", "MU", "NVDA", "PLTR", "TSLA"]


class Mode(StrEnum):
    BACKTEST = "backtest"
    PAPER = "paper"  # auto-executes against the Alpaca paper account
    LIVE = "live"  # advisory only: issues tickets, never places orders


class RiskLimits(BaseModel):
    """Action-space, sizing and hedging parameters (paper Table 10, App. D)."""

    max_positions: int = 12
    max_positions_per_underlying: int = 3
    max_order_lines: int = 8
    candidates_per_underlying: int = 6
    delta_step: float = 0.05
    target_risk_frac: float = 0.005  # phi: scenario risk per package, fraction of NAV
    iv_shock: float = 0.10  # kappa: relative implied-vol shock
    risk_sigma: float = 2.0  # spot shock in daily sigmas for scenario risk (paper: 1, Taylor)
    nav_frac_cap: float = 0.10  # f_NAV: capital per package
    risk_cap_per_underlying: float = 0.05  # rho_u
    risk_cap_total: float = 0.20  # rho_tot
    net_delta_cap: float = 0.50  # delta_max: net dollar delta, fraction of NAV
    max_notional_frac: float = (
        0.50  # units x 100 x spot <= this x NAV: a package's |delta| <= its notional
    )
    collateral_buffer: float = 0.05  # b
    hedge_band_frac: float = 0.005  # Delta_band: no-hedge band half-width, fraction of NAV
    force_close_dte: int = 1
    stop_loss_frac: float = 0.60  # of entry premium
    ruin_floor_frac: float = 0.20  # of initial NAV
    max_rel_half_spread: float = (
        0.25  # legs with (ask-bid)/2 above this fraction of the mark are untradable
    )
    smooth_surface: bool = True  # mark/price legs off a fitted IV smile instead of raw EOD mids


class Costs(BaseModel):
    """Trading and execution assumptions (paper Table 5)."""

    option_fee: float = 0.65  # per contract per leg
    assignment_fee: float = 5.0  # per event
    share_commission: float = 0.005  # per share
    borrow_rate: float = 0.005  # annual, on short shares
    half_spread_mult: float = 0.0  # 0 = midpoint fills; 1 = cross the full quoted half-spread


COST_PRESETS: dict[str, Costs] = {
    "paper": Costs(),
    # Robinhood: zero commission; regulatory/exchange fees are a few cents per contract.
    "robinhood": Costs(option_fee=0.04, assignment_fee=0.0, share_commission=0.0, borrow_rate=0.0),
    "ibkr": Costs(option_fee=0.65, assignment_fee=0.0, share_commission=0.005, borrow_rate=0.005),
}


class AlpacaSettings(BaseModel):
    api_key: SecretStr | None = None
    secret_key: SecretStr | None = None
    feed: Literal["indicative", "opra"] = "indicative"


class ThetaSettings(BaseModel):
    """`library` = thetadata package (API key, no Terminal); `rest` = Theta Terminal v3 running locally."""

    transport: Literal["library", "rest"] = "library"
    api_key: SecretStr | None = None  # library; else THETADATA_API_KEY / .env / creds.txt discovery
    base_url: str = "http://127.0.0.1:25503/v3"  # rest
    timeout_s: float = 120.0
    requests_per_minute: int = 20  # Free tier cap; paid tiers limit concurrency instead, so use 0


class LLMSettings(BaseModel):
    """Any OpenAI-compatible chat-completions endpoint (OpenAI, OpenRouter, Ollama, vLLM)."""

    base_url: str = "https://api.openai.com/v1"
    api_key: SecretStr | None = None
    model: str = "gpt-4.1"
    temperature: float = 0.2
    timeout_s: float = 120.0


class NotifySettings(BaseModel):
    telegram_bot_token: SecretStr | None = None
    telegram_chat_id: str | None = None
    smtp_host: str | None = None
    smtp_port: int = 587
    smtp_user: str | None = None
    smtp_password: SecretStr | None = None
    email_from: str | None = None
    email_to: str | None = None
    macos: bool = True


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SOTA_", env_nested_delimiter="__", env_file=".env", extra="ignore"
    )

    mode: Mode = Mode.PAPER
    universe: list[str] = Field(default_factory=lambda: list(DEFAULT_UNIVERSE))
    initial_nav: float = 1_000_000.0
    risk_free_rate: float = 0.0
    selector: Literal["threshold", "llm", "oracle"] = "threshold"
    threshold_gate: bool = True  # paper Table 8: trade only when term structure ts < 0
    threshold_tenor: Literal["0-7", "8-30", "31-90", "91-180"] = "8-30"
    cost_preset: Literal["paper", "robinhood", "ibkr"] = "paper"
    data_dir: Path = Path("data")
    journal_path: Path = Path("data/journal.sqlite")
    web_host: str = "127.0.0.1"
    web_port: int = 8765

    limits: RiskLimits = Field(default_factory=RiskLimits)
    costs: Costs | None = None  # explicit override of the preset
    alpaca: AlpacaSettings = Field(default_factory=AlpacaSettings)
    theta: ThetaSettings = Field(default_factory=ThetaSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    notify: NotifySettings = Field(default_factory=NotifySettings)

    @property
    def effective_costs(self) -> Costs:
        return self.costs if self.costs is not None else COST_PRESETS[self.cost_preset]


def load_settings(**overrides: object) -> Settings:
    return Settings(**overrides)  # type: ignore[arg-type]
