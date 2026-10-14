# sota

Options strategy cockpit built on *SOTA: Stock Options Trading Agents Guided by Option-Implied
Return Distributions* (Xie & Liu, 2026). The paper's structure is implemented deterministically —
10 option-implied features, 9 strategy families, contract/size/hedge resolvers, risk caps, exits,
and the eight evaluation metrics — with the strategy *selector* as a swappable module.

Three modes:

| mode | data | execution |
|---|---|---|
| `backtest` | Theta Data (`thetadata` library with an API key, or the Theta Terminal REST server) or `--synthetic` | point-in-time environment, midpoint fills + fees |
| `paper` | Alpaca market data | auto-executes on the Alpaca **paper** account (`paper=True` hard-coded) |
| `live` | Alpaca market data (read-only) | **never places orders** — issues tickets you enter manually (e.g. Robinhood), then confirm |

## Setup

```bash
uv sync --extra dev
cp .env.example .env        # fill in keys; SOTA_MODE=paper|live|backtest
uv run pytest
```

## Try it without any API keys

```bash
uv run sota backtest 2024-03-01 2024-09-30 --synthetic -s threshold -s oracle
SOTA_THRESHOLD_GATE=false uv run sota live propose --synthetic --date 2024-06-03 --nav 50000
uv run sota live tickets
uv run sota live confirm <ticket-id> --price 3.62
uv run sota web                     # http://127.0.0.1:8765
```

## Daily use

```bash
uv run sota scan                    # the 10 features for the universe
uv run sota paper run               # paper: exits, proposals, orders, hedges, journal
uv run sota live propose            # live: tickets via Telegram / email / macOS / web UI
uv run sota live confirm ID --price P [--units N]   # record your fill
uv run sota positions --mode live
```

Selectors: `threshold` (paper Table 8 rules; `SOTA_THRESHOLD_GATE=false` disables the `ts<0`
participation gate), `llm` (anonymized state -> JSON candidates via any OpenAI-compatible API),
`oracle` (hindsight reference, backtest only), `replay:<proposals.json>` (re-opens a recorded run's
trades under the current limits; files come from `sota backtest --out DIR`). Cost presets: `paper`
(Table 5), `robinhood`, `ibkr`.

## Forward test — pre-registered 2026-10-14

Three historical years gave no consistent edge with the paper's full menu (iron condors lost in two
of three years and drove the one big winner); replays without condors were modest and positive in
two of three years (-9.8% / +18.5% / +10.7%, worst drawdown 12.4%). That configuration — `llm`
selector, families `ol dv cv dg ib bf`, $50k, Robinhood costs, unhedged — is now paper trading on
Alpaca, one decision per day at 15:50 ET. The following criteria were fixed **before** the first
run and are evaluated by `sota scorecard` and the cockpit (`src/sota/scorecard.py`):

| # | criterion | threshold |
|---|---|---|
| C1 | total return after costs | > 0 at 6 months; > ~4% (cash) at 12 months |
| C2 | maximum drawdown, daily closes | < 15% at every point |
| C3 | trade rate (annualized after 60 days) | 20–50 per year |
| C4 | worst calendar month | better than −8% |

Verdict rule: two or more failures at the 6-month checkpoint end the experiment. All four must pass
at 12 months before any real-money (advisory-ticket) use, and then only at a fraction of size.
Changing the configuration restarts the clock.

## Paper schedule

`launchd/com.sota.paper.plist` (installed in `~/Library/LaunchAgents`) runs `scripts/paper_run.sh`
at 15:45 and 16:45 local; the script keeps whichever firing falls in the 15:50 America/New_York
window, so US daylight-saving changes never move the decision time. One run per trading day; logs in
`data/paper/run_<date>.log`. The Mac must be awake at that time (`pmset repeat wakeorpoweron`).
Pause with `launchctl bootout gui/$(id -u)/com.sota.paper`.

Theta Data transport: `SOTA_THETA__TRANSPORT=library` (default) talks to Theta's servers directly
through the `thetadata` package using `SOTA_THETA__API_KEY` (or `THETADATA_API_KEY`); `rest` uses a
locally running Theta Terminal v3 at `SOTA_THETA__BASE_URL`. Both yield identical chains, and every
fetched day is cached under `data/theta/` so repeated backtests run offline.

Layout: `src/sota/{features,strategies,resolvers,risk,pipeline}.py` are the paper's mechanics;
`backtest/` the environment and metrics; `selectors/` the policies; `data/` Theta/Alpaca/synthetic
sources; `execution/` the paper broker and manual-ticket adapter; `runner.py` the daily cycle;
`journal.py` the SQLite record of every proposal, ticket, fill and NAV.

