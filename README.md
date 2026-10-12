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
`oracle` (hindsight reference, backtest only). Cost presets: `paper` (Table 5), `robinhood`, `ibkr`.

Theta Data transport: `SOTA_THETA__TRANSPORT=library` (default) talks to Theta's servers directly
through the `thetadata` package using `SOTA_THETA__API_KEY` (or `THETADATA_API_KEY`); `rest` uses a
locally running Theta Terminal v3 at `SOTA_THETA__BASE_URL`. Both yield identical chains, and every
fetched day is cached under `data/theta/` so repeated backtests run offline.

Layout: `src/sota/{features,strategies,resolvers,risk,pipeline}.py` are the paper's mechanics;
`backtest/` the environment and metrics; `selectors/` the policies; `data/` Theta/Alpaca/synthetic
sources; `execution/` the paper broker and manual-ticket adapter; `runner.py` the daily cycle;
`journal.py` the SQLite record of every proposal, ticket, fill and NAV.

