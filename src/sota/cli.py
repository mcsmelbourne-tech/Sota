"""Command-line interface: `sota --help`."""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from sota.backtest.env import TradingEnv
from sota.backtest.metrics import compute_metrics
from sota.config import Mode, Settings
from sota.factory import build_runner, build_selector, build_source
from sota.journal import Journal
from sota.notify.base import LogNotifier
from sota.pipeline import compute_state, load_chains

app = typer.Typer(help="SOTA options strategy cockpit.", no_args_is_help=True)
live_app = typer.Typer(
    help="Live/advisory mode: tickets for manual entry in your broker app.", no_args_is_help=True
)
paper_app = typer.Typer(
    help="Paper mode: auto-trade the Alpaca paper account.", no_args_is_help=True
)
app.add_typer(live_app, name="live")
app.add_typer(paper_app, name="paper")
console = Console()

DateOpt = Annotated[str | None, typer.Option("--date", help="YYYY-MM-DD (default today)")]


def _settings(mode: str | None = None) -> Settings:
    settings = Settings()
    if mode is not None:
        settings = settings.model_copy(update={"mode": Mode(mode)})
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    return settings


def _date(s: str | None) -> date:
    return date.fromisoformat(s) if s else date.today()


def _trading_days(source, underlying: str, start: date, end: date) -> list[date]:
    if hasattr(source, "trading_days"):
        return list(source.trading_days(underlying, start, end))
    days, d = [], start
    while d <= end:
        if d.weekday() < 5:
            days.append(d)
        d = d.fromordinal(d.toordinal() + 1)
    return days


@app.command()
def info() -> None:
    """Show effective settings (secrets masked)."""
    s = _settings()
    table = Table(title="SOTA settings")
    table.add_column("key")
    table.add_column("value")
    table.add_row("mode", s.mode.value)
    table.add_row("selector", s.selector)
    table.add_row("universe", ", ".join(s.universe))
    table.add_row("initial_nav", f"{s.initial_nav:,.0f}")
    table.add_row("cost_preset", s.cost_preset)
    table.add_row("costs", s.effective_costs.model_dump_json())
    table.add_row("limits", s.limits.model_dump_json())
    table.add_row("alpaca keys", "set" if s.alpaca.api_key else "missing")
    theta_auth = "set" if s.theta.api_key else "via THETADATA_API_KEY/.env/creds.txt"
    table.add_row(
        "theta",
        f"{s.theta.transport} / {s.theta.base_url if s.theta.transport == 'rest' else f'api key {theta_auth}'}",
    )
    table.add_row(
        "llm", f"{s.llm.base_url} / {s.llm.model} / key {'set' if s.llm.api_key else 'missing'}"
    )
    table.add_row("journal", str(s.journal_path))
    console.print(table)


@app.command()
def scan(asof: DateOpt = None, synthetic: bool = False) -> None:
    """Compute the 10 market-state features for the universe (past dates use Theta Data)."""
    d = _date(asof)
    s = _settings("backtest" if d < date.today() else None)
    source = build_source(s, synthetic=synthetic)
    chains = load_chains(source, s.universe, d)
    features = compute_state(source, chains, d)
    table = Table(title=f"Market state {d}")
    for col in ("underlying", "spot", "ret", "rv", "iv", "iv/rv", "wedge", "ts", "sk", "bf", "doi"):
        table.add_column(col, justify="right")
    for u, f in features.items():
        table.add_row(
            u,
            f"{f.spot:.2f}",
            f"{f.ret:+.3%}",
            f"{f.rv:.3f}",
            f"{f.iv:.3f}",
            f"{f.iv_rv_ratio:.2f}",
            f"{f.wedge:+.3f}",
            f"{f.ts:+.3f}",
            f"{f.sk:+.3f}",
            f"{f.bf:+.3f}",
            f"{f.doi:+.3f}",
        )
    console.print(table)


@app.command()
def fetch(start: str, end: str) -> None:
    """Download Theta Data chains for a date range into the local cache (resumable; Ctrl-C safe)."""
    from rich.progress import Progress

    s = _settings("backtest")
    source = build_source(s)
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    fetched = skipped = empty = 0
    try:
        with Progress(console=console) as progress:
            for u in s.universe:
                # realized vol needs ~21 trading days of closes before the first decision date
                source.ensure_closes(u, first - timedelta(days=45), last)  # type: ignore[attr-defined]
                days = source.trading_days(u, first, last)  # type: ignore[attr-defined]
                task = progress.add_task(f"{u:<6}", total=len(days))
                for d in days:
                    if source.store.has_chain(u, d):  # type: ignore[attr-defined]
                        skipped += 1
                    elif source.get_chain(u, d) is None:
                        empty += 1
                    else:
                        fetched += 1
                    progress.advance(task)
    except KeyboardInterrupt:
        console.print("[yellow]interrupted; re-run the same command to resume[/yellow]")
    console.print(f"fetched {fetched}, already cached {skipped}, no data {empty}")


@app.command()
def backtest(
    start: str,
    end: str,
    selector: Annotated[
        list[str], typer.Option("--selector", "-s", help="threshold | llm | oracle (repeatable)")
    ] = ["threshold"],  # noqa: B006
    synthetic: bool = typer.Option(False, help="Use the synthetic market instead of Theta Data"),
    nav: float | None = None,
) -> None:
    """Run the point-in-time environment over a date range and report the paper's eight metrics."""
    s = _settings("backtest")
    source = build_source(s, synthetic=synthetic)
    dates = _trading_days(source, s.universe[0], date.fromisoformat(start), date.fromisoformat(end))
    if not dates:
        raise typer.BadParameter(
            "no trading days in range (is Theta Terminal running / data cached?)"
        )
    table = Table(
        title=f"Backtest {dates[0]} -> {dates[-1]} ({len(dates)} days, {len(s.universe)} underlyings)"
    )
    table.add_column("selector")
    for col in ("TR", "ASR", "ACR", "ASoR", "AVOL", "MDD", "WR", "PLR", "trades"):
        table.add_column(col, justify="right")
    for name in selector:
        env = TradingEnv(source, s.universe, s.limits, s.effective_costs, nav or s.initial_nav)
        result = env.run(dates, build_selector(s, source, name))
        m = compute_metrics(result.nav_values, result.trade_pnls(), rf=s.risk_free_rate).as_row()
        table.add_row(
            name,
            m["TR"],
            m["ASR"],
            m["ACR"],
            m["ASoR"],
            m["AVOL"],
            m["MDD"],
            m["WR"],
            m["PLR"],
            m["trades"],
        )
    console.print(table)


@app.command()
def positions(mode: str | None = None) -> None:
    """List open positions recorded in the journal."""
    s = _settings()
    journal = Journal(s.journal_path)
    m = mode or s.mode.value
    table = Table(title=f"Open positions ({m})")
    for col in ("id", "opened", "package", "units", "entry", "stop/sh", "force close"):
        table.add_column(col)
    for p in journal.list_positions(m, "open"):
        table.add_row(
            p.id,
            str(p.entry_date),
            p.package.describe(),
            str(p.units),
            f"{p.entry_premium:+.2f}",
            f"{p.stop_loss_per_share:.2f}",
            str(p.force_close_date),
        )
    console.print(table)


# --------------------------------------------------------------------------- paper
@paper_app.command("run")
def paper_run(asof: DateOpt = None, reconcile: bool = True) -> None:
    """One decision cycle against the Alpaca paper account (exits, proposals, orders, hedges)."""
    runner = build_runner(_settings("paper"))
    if reconcile:
        for note in runner.reconcile():
            console.print(f"[dim]reconcile:[/dim] {note}")
    console.print(runner.run_once(_date(asof)).summary())


# --------------------------------------------------------------------------- live
@live_app.command("propose")
def live_propose(
    asof: DateOpt = None,
    nav: float | None = None,
    synthetic: bool = typer.Option(False, help="Dry run against the synthetic market"),
) -> None:
    """Scan, select, and issue order tickets. Never places orders."""
    runner = build_runner(_settings("live"), synthetic=synthetic)
    if nav is not None:
        runner.broker.set_nav(nav)  # type: ignore[union-attr]
    report = runner.run_once(_date(asof))
    console.print(report.summary())
    from sota.execution.manual import format_ticket

    for t in report.tickets:
        console.rule(f"ticket {t.id}")
        console.print(format_ticket(t))


@live_app.command("tickets")
def live_tickets(status: str = "pending") -> None:
    """List tickets (pending by default)."""
    s = _settings("live")
    table = Table(title=f"Tickets ({status})")
    for col in ("id", "created", "intent", "package", "units", "limit", "status"):
        table.add_column(col)
    for t in Journal(s.journal_path).list_tickets(status):
        table.add_row(
            t.id,
            t.created.strftime("%Y-%m-%d %H:%M"),
            t.intent,
            t.package.describe(),
            str(t.units),
            f"{t.limit_price:+.2f}",
            t.status,
        )
    console.print(table)


@live_app.command("confirm")
def live_confirm(
    ticket_id: str,
    price: Annotated[
        float, typer.Option("--price", help="Net fill per share: debit positive, credit negative")
    ],
    units: int | None = None,
    asof: DateOpt = None,
) -> None:
    """Record the fill you obtained manually; opens or closes the journal position."""
    s = _settings("live")
    from sota.execution.manual import ManualTicketBroker

    journal = Journal(s.journal_path)
    broker = ManualTicketBroker(journal, LogNotifier(), s.initial_nav, s.effective_costs)
    pos = broker.confirm(ticket_id, price, units, _date(asof))
    console.print(f"ticket {ticket_id} filled -> position {pos.id} ({pos.status})")


@live_app.command("skip")
def live_skip(ticket_id: str) -> None:
    """Mark a ticket as not taken."""
    s = _settings("live")
    from sota.execution.manual import ManualTicketBroker

    ManualTicketBroker(
        Journal(s.journal_path), LogNotifier(), s.initial_nav, s.effective_costs
    ).skip(ticket_id)
    console.print(f"ticket {ticket_id} skipped")


@live_app.command("set-nav")
def live_set_nav(nav: float, asof: DateOpt = None) -> None:
    """Record your live account NAV (used for sizing and risk caps)."""
    s = _settings("live")
    Journal(s.journal_path).record_nav(_date(asof), "live", nav)
    console.print(f"live NAV set to {nav:,.0f}")


@app.command()
def web(host: str | None = None, port: int | None = None) -> None:
    """Start the web UI (tickets, positions, NAV)."""
    import uvicorn

    s = _settings()
    uvicorn.run("sota.web.app:app", host=host or s.web_host, port=port or s.web_port, reload=False)


if __name__ == "__main__":
    app()
