"""Command-line interface: `sota --help`."""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from sota.backtest.env import BacktestResult, TradingEnv
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
notify_app = typer.Typer(
    help="Notification channels: test delivery, find ids.", no_args_is_help=True
)
app.add_typer(live_app, name="live")
app.add_typer(paper_app, name="paper")
app.add_typer(notify_app, name="notify")
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
    table.add_row("initial_nav (backtest)", f"{s.initial_nav:,.0f}")
    table.add_row(
        "sizing_nav (paper)",
        f"{s.sizing_nav:,.0f}" if s.sizing_nav else "not set -> real account equity",
    )
    lim = s.limits
    table.add_row(
        "risk profile",
        f"risk {lim.target_risk_frac:.1%}/package, capital cap {lim.nav_frac_cap:.0%}, "
        f"hedge {'on' if lim.hedge else 'off'}, families "
        f"{','.join(lim.families) if lim.families else 'all'}",
    )
    table.add_row(
        "threshold rule", f"gate {'on' if s.threshold_gate else 'off'}, tenor {s.threshold_tenor}"
    )
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
    chains = load_chains(source, s.universe, d, s.limits)
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
def sizing(
    nav: float = typer.Option(..., help="Account size to size for"),
    asof: DateOpt = None,
    synthetic: bool = False,
    tradable_only: bool = typer.Option(False, help="Hide structures that resolve to 0 contracts"),
) -> None:
    """What the resolvers would trade at this account size: units per family/tenor and why not."""
    import math

    from sota.models import Side, Tenor
    from sota.pipeline import DEFAULT_ATM_IV, atm_iv_map
    from sota.resolvers.contract import resolve_package
    from sota.resolvers.size import resolve_size
    from sota.risk import PortfolioRisk
    from sota.strategies.families import FAMILIES

    d = _date(asof)
    s = _settings("backtest" if d < date.today() or synthetic else None)
    source = build_source(s, synthetic=synthetic)
    if asof is None and not synthetic:  # default to the most recent cached/available day
        for back in range(10):
            if load_chains(source, s.universe[:1], d - timedelta(days=back), s.limits):
                d = d - timedelta(days=back)
                break
    chains = load_chains(source, s.universe, d, s.limits)
    atm_iv = atm_iv_map(compute_state(source, chains, d))
    lim = s.limits
    hedging = "on" if lim.hedge else "off"
    table = Table(
        title=f"Sizing at NAV ${nav:,.0f} on {d} (risk {lim.target_risk_frac:.1%}/package, "
        f"capital cap {lim.nav_frac_cap:.0%}, hedging {hedging})"
    )
    for col in (
        "underlying",
        "family",
        "tenor",
        "premium",
        "max loss",
        "risk/unit",
        "units",
        "binding",
    ):
        table.add_column(col, justify="right" if col not in ("binding", "family") else "left")
    tradable = total = 0
    for u, chain in chains.items():
        for code, spec in FAMILIES.items():
            if lim.families is not None and code not in lim.families:
                continue
            side = Side.LONG
            for tenor in (Tenor.T8_30, Tenor.T31_90):
                pkg = resolve_package(
                    chain,
                    code,
                    side,
                    tenor,
                    min_dte=lim.force_close_dte + 1,
                    max_rel_half_spread=lim.max_rel_half_spread,
                )
                total += 1
                if pkg is None:
                    if not tradable_only:
                        table.add_row(
                            u,
                            f"{code} {spec.name}",
                            tenor.value,
                            "-",
                            "-",
                            "-",
                            "0",
                            "no liquid contracts",
                        )
                    continue
                res = resolve_size(
                    pkg, chain.spot, atm_iv.get(u, DEFAULT_ATM_IV), lim, PortfolioRisk(nav=nav), d
                )
                tradable += res.accepted
                if tradable_only and not res.accepted:
                    continue
                ml = "unbounded" if pkg.max_loss is None else f"${pkg.max_loss * 100:,.0f}"
                style = "" if res.accepted else "dim"
                table.add_row(
                    u,
                    f"{code} {spec.name}",
                    tenor.value,
                    f"${pkg.net_premium * 100:+,.0f}",
                    ml,
                    f"${res.risk_per_unit:,.0f}" if math.isfinite(res.risk_per_unit) else "-",
                    str(res.units),
                    res.binding,
                    style=style,
                )
    console.print(table)
    console.print(f"{tradable}/{total} family-tenor combinations fit at least 1 contract")


@app.command()
def fetch(
    start: str,
    end: str,
    max_outage_min: int = typer.Option(
        60, help="Give up after this long without a successful request"
    ),
) -> None:
    """Download Theta Data chains for a date range into the local cache (resumable; Ctrl-C safe).

    Survives network outages and laptop sleep: a failed request is retried every minute, with a
    fresh connection, until `--max-outage-min` passes without success.
    """
    import time
    from datetime import datetime

    from rich.progress import Progress

    from sota.data.theta import error_detail

    s = _settings("backtest")
    source = build_source(s)
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    fetched = skipped = empty = 0
    heartbeat = 0 if console.is_terminal else 25  # progress bars don't render into a log file

    def resilient(call, label: str):
        waited = 0.0
        while True:
            try:
                return call()
            except KeyboardInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001 - any failure here is retried, not fatal
                if waited >= max_outage_min * 60:
                    raise
                console.print(
                    f"{datetime.now():%H:%M:%S} {label}: {error_detail(exc)} -> retry in 60s"
                )
                time.sleep(60)
                waited += 60
                source.reset_transport()  # type: ignore[attr-defined]

    try:
        with Progress(console=console) as progress:
            for u in s.universe:
                # realized vol needs ~21 trading days of closes before the first decision date
                resilient(
                    lambda u=u: source.ensure_closes(u, first - timedelta(days=45), last),  # type: ignore[attr-defined]
                    f"{u} closes",
                )
                days = source.trading_days(u, first, last)  # type: ignore[attr-defined]
                task = progress.add_task(f"{u:<6}", total=len(days))
                done_before = fetched + skipped + empty
                for i, d in enumerate(days, 1):
                    if source.store.has_chain(u, d):  # type: ignore[attr-defined]
                        skipped += 1
                    elif resilient(lambda u=u, d=d: source.get_chain(u, d), f"{u} {d}") is None:
                        empty += 1
                    else:
                        fetched += 1
                    progress.advance(task)
                    if heartbeat and i % heartbeat == 0:
                        console.print(f"{datetime.now():%H:%M:%S} {u} {i}/{len(days)}")
                console.print(
                    f"{u}: {fetched + skipped + empty - done_before}/{len(days)} days done "
                    f"(total fetched {fetched}, cached {skipped}, no data {empty})"
                )
    except KeyboardInterrupt:
        console.print("[yellow]interrupted; re-run the same command to resume[/yellow]")
    except Exception as exc:  # noqa: BLE001
        console.print(
            f"[red]giving up after {max_outage_min} min of failures: {error_detail(exc)}[/red]"
        )
        console.print("[yellow]the cache is intact; re-run the same command to resume[/yellow]")
        raise typer.Exit(code=1) from exc
    console.print(f"fetched {fetched}, already cached {skipped}, no data {empty}")


@app.command()
def backtest(
    start: str,
    end: str,
    selector: Annotated[
        list[str],
        typer.Option(
            "--selector",
            "-s",
            help="threshold | llm | oracle | replay:<proposals.json> (repeatable)",
        ),
    ] = ["threshold"],  # noqa: B006
    synthetic: bool = typer.Option(False, help="Use the synthetic market instead of Theta Data"),
    nav: float | None = None,
    out: Annotated[
        Path | None,
        typer.Option(help="Directory for <selector>_positions.csv and <selector>_nav.csv"),
    ] = None,
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
        sel = build_selector(s, source, name)
        result = env.run(dates, sel)
        m = compute_metrics(result.nav_values, result.trade_pnls(), rf=s.risk_free_rate).as_row()
        label = name
        if result.aborted:
            covered = f"{len(result.nav)}/{len(dates)} days"
            label = f"{name} [ABORTED {covered}]"
            console.print(
                f"[red]{name}: selector unavailable after {covered}; metrics below are partial "
                f"and must not be compared with complete runs.[/red]\n[red]  {result.aborted}[/red]"
            )
        table.add_row(
            label,
            m["TR"],
            m["ASR"],
            m["ACR"],
            m["ASoR"],
            m["AVOL"],
            m["MDD"],
            m["WR"],
            m["PLR"],
            m["trades"],
            style="dim" if result.aborted else "",
        )
        if usage := getattr(sel, "usage", None):
            console.print(f"[dim]{name}: {usage.summary()}[/dim]")
        if out is not None:
            _write_backtest_csv(out, name, result)
    console.print(table)


def _write_backtest_csv(out: Path, name: str, result: BacktestResult) -> None:
    import csv
    import json

    from sota.selectors.replay import proposals_from_positions

    name = name.split(":", 1)[0]  # "replay:<path>" -> "replay"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}_proposals.json").write_text(
        json.dumps(proposals_from_positions(result.positions), indent=1)
    )
    with (out / f"{name}_positions.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            "id underlying family side tenor entry_date exit_date status units entry exit "
            "fees pnl rationale".split()
        )
        for p in result.positions:
            w.writerow(
                [
                    p.id,
                    p.package.underlying,
                    p.package.family,
                    p.package.side.value,
                    p.package.tenor.value,
                    p.entry_date,
                    p.exit_date or "",
                    p.status,
                    p.units,
                    f"{p.entry_premium:.4f}",
                    "" if p.exit_premium is None else f"{p.exit_premium:.4f}",
                    f"{p.fees_paid:.2f}",
                    "" if p.realized_pnl is None else f"{p.realized_pnl:.2f}",
                    p.tags.get("rationale", ""),
                ]
            )
    with (out / f"{name}_nav.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "nav"])
        w.writerows((d, f"{v:.2f}") for d, v in result.nav)
    console.print(f"[dim]{name}: wrote {out / f'{name}_positions.csv'}[/dim]")


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


@app.command()
def scorecard(mode: str | None = None) -> None:
    """Forward-test scorecard against the pre-registered criteria (see README)."""
    from sota.scorecard import RULES, build_scorecard

    s = _settings()
    m = mode or s.mode.value
    journal = Journal(s.journal_path)
    card = build_scorecard(journal.nav_series(m), journal.list_positions(m, None))
    colours = {"pass": "green", "warn": "yellow", "fail": "red", "pending": "dim"}
    if card.start is None:
        console.print(f"[dim]{card.verdict}[/dim]")
        return
    table = Table(
        title=f"Forward test ({m}) since {card.start}: day {card.days_elapsed}, "
        f"NAV ${card.nav:,.0f} from ${card.start_nav:,.0f}"
    )
    for col in ("criterion", "value", "target", "status", "detail"):
        table.add_column(col)
    for c in card.criteria:
        table.add_row(
            c.label,
            c.value,
            c.target,
            f"[{colours[c.status]}]{c.status}[/{colours[c.status]}]",
            c.detail,
        )
    console.print(table)
    console.print(
        f"checkpoints {card.checkpoint_6m} / {card.checkpoint_12m}  ->  [bold]{card.verdict}[/bold]"
    )
    console.print("\n".join(f"[dim]{r}[/dim]" for r in RULES))


# --------------------------------------------------------------------------- notify
@notify_app.command("test")
def notify_test() -> None:
    """Send a test message through every configured channel and report each one's outcome."""
    from sota.notify.channels import build_notifier

    s = _settings()
    multi = build_notifier(s.notify, include_log=False)
    body = f"mode={s.mode.value} selector={s.selector} universe={','.join(s.universe)}"
    ok = True
    for ch in multi.channels:  # type: ignore[attr-defined]
        try:
            ch.send("SOTA test notification", body)
            console.print(f"[green]{ch.name}: delivered[/green]")
        except Exception as exc:  # noqa: BLE001 - report every channel, then exit non-zero
            ok = False
            console.print(f"[red]{ch.name}: {exc}[/red]")
    if not ok:
        raise typer.Exit(code=1)


@notify_app.command("chat-id")
def notify_chat_id() -> None:
    """Print the Telegram chat ids that have messaged the bot (send it any message first)."""
    from sota.notify.channels import telegram_chat_ids

    s = _settings()
    if not s.notify.telegram_bot_token:
        raise typer.BadParameter("set SOTA_NOTIFY__TELEGRAM_BOT_TOKEN first (from @BotFather)")
    chats = telegram_chat_ids(s.notify.telegram_bot_token.get_secret_value())
    if not chats:
        console.print(
            "[yellow]no messages yet: open the bot in Telegram, press Start, send 'hi', rerun[/yellow]"
        )
        return
    for chat_id, name in chats:
        console.print(f"{chat_id}  {name}")
    console.print(
        "[dim]put the id in SOTA_NOTIFY__TELEGRAM_CHAT_ID, then: uv run sota notify test[/dim]"
    )


# --------------------------------------------------------------------------- paper
@paper_app.command("run")
def paper_run(
    asof: DateOpt = None,
    reconcile: bool = True,
    force: bool = typer.Option(False, help="Run even if the market is closed (orders would queue)"),
) -> None:
    """One decision cycle against the Alpaca paper account (exits, proposals, orders, hedges)."""
    runner = build_runner(_settings("paper"))
    is_open = getattr(runner.broker, "market_is_open", lambda: True)
    if not force and not is_open():
        # a DAY order placed after an early close fills at tomorrow's open, not today's marks
        console.print("[yellow]market is closed (holiday or early close); skipping[/yellow]")
        return
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
