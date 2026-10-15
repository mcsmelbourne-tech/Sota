# sota

A tool that puts the idea from the paper *SOTA: Stock Options Trading Agents Guided by
Option-Implied Return Distributions* into practice, and a test of that idea on three years of real
option prices.

The short version: we could not make it work for a retail account. The framework runs as the paper
describes it, and over the paper's own six-month period our numbers look similar to theirs. But over
three years, with a $50,000 account, no hedging and retail costs, the strategy lost money in one
year out of three, and lost a lot. Possible reasons, which we cannot separate with the data we have:

- We used a general LLM (Grok) as the selector. The paper trains its own model. We did not run their
  model, so maybe their model does better than ours.
- The paper tests six months. We tested three years. The six months in the paper were a calm period
  for this kind of strategy, and the bad year in our test is a trending year with big moves.
- The paper uses $1,000,000 and hedges with shares. A retail account cannot do that. Without the
  hedge, the strategy depends on the direction of the stock, and that is where our losses came from.
- The paper uses ten stocks. We used three (SPY, NVDA, TSLA), because the other seven have wide
  bid-ask spreads that eat the profit at retail size.

The details are in "Results" below.

## The paper

**SOTA: Stock Options Trading Agents Guided by Option-Implied Return Distributions**
Yizhen Xie (Carnegie Mellon University) and Mengyang Liu (Amazon), 2026.
Accepted at the NeurIPS 2026 Agenthon Workshop. License CC BY 4.0.

- Abstract and PDF: https://arxiv.org/abs/2610.10407
- HTML version: https://arxiv.org/html/2610.10407v1

### The paper in simple words

**The problem.** One stock can have thousands of option contracts: many strikes, many expiry
dates, calls and puts. A trading program that has to pick contracts one by one has too many
choices. Most older programs avoid this by always trading one fixed structure, for example always a
straddle. Then they cannot change strategy when the market changes.

**The idea.** Do not let the model pick contracts. Let it pick a *strategy* from a short list, and
let simple fixed rules (the paper calls them "resolvers") turn that choice into real contracts, a
position size, and a hedge. The model makes the big decision. The rules do the rest.

**What the model sees.** For each stock, 10 numbers that come from the option prices, not from
the news. In plain words: how the stock moved today, how much it has been moving (realized
volatility), how much the market expects it to move (implied volatility) and how that changed
today, the gap between the two, whether long-dated options are more or less expensive than
short-dated ones, whether puts are more expensive than calls (skew), how curved the smile is, which
way option buyers are pushing (flow), and how open interest changed. Plus the open positions and the
account state. The model never sees the stock name, the price level, or the date; the authors hide
these so the model cannot remember what happened on that day.

**What the model picks.** One line per decision: the stock, one of nine strategy families, long or
short, a time bucket (0–7, 8–30, 31–90 or 91–180 days to expiry), and a point on a delta grid that
says how far from the current price. The nine families, in four groups:

| group | family | what it bets on |
|---|---|---|
| direction | outright call or put; debit vertical; credit vertical; defined-risk reversal | the stock goes up (or down) |
| volatility | long straddle; long strangle | the stock moves a lot, either way |
| volatility | iron butterfly; iron condor | the stock stays in a range |
| curvature | butterfly | the stock lands near one price |

It can open, close, roll, or do nothing. It first proposes up to six candidates per stock; the tool
prices only those; then it makes the final choice. This way the model never has to read the full
option chain.

**The fixed rules.** The contract resolver picks the expiry closest to the time bucket and the
strike closest to the asked delta. The size resolver gives every position the same amount of risk
(in the paper, 0.5% of the account for a one-day shock in price, volatility and time). The hedge
resolver buys or sells shares against the volatility positions, so they do not depend on the
direction of the stock. Hard limits: at most 12 open positions, 3 per stock, a stop loss at 60% of
the premium, close everything one day before expiry, and stop trading if the account falls to 20%
of where it started.

**How they trained it.** They started with a 27B open model (Qwen3.8). First, they had a stronger
model trade in the environment, kept its best runs, and taught the smaller model to copy them.
Then they let the smaller model trade on its own and rewarded it for portfolio returns after costs
(reinforcement learning). Training data is Sep 2024 to Feb 2025.

**What they report.** On 10 names (SPY plus nine big US stocks), $1,000,000 account, with hedging,
from March to August 2025: **+18.3% total return, Sharpe 1.60, maximum drawdown 9.0%.** All the
rule-based and machine-learning baselines lost money in the same period. A "hindsight oracle" that
sees the future shows how much was possible. One more finding: giving the model news during the
second training step made the result worse (+18.3% became −2.7%).

**What this repo does with it.** We built the whole framework again (features, families,
resolvers, limits, exits, metrics) from the description in the paper. We put a general LLM (Grok)
in the place of their trained model as the selector. Then we tested it over three years instead of
six months, with a retail account size. We also connected it to a paper-trading account and a web
page, so it can run every day by itself. The result of the test is below, after the setup notes.

## What is in this repo

The structure of the paper is implemented as fixed code: the 10 option-implied features, the 9
strategy families, the contract/size/hedge resolvers, the risk limits, the exits, and the eight
metrics. The strategy *selector* is a separate piece that you can change.

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

**Status (2026-10-14, evening): paused after one day.** The new backtests we ran the same evening
(see "Results" below) showed that this setup lost 51% in the 2023-24 year. There is no reason to
keep testing it. We keep the rules here as a record of how the test was planned.

The plan was: `llm` selector, families `ol dv cv dg ib bf` (no iron condors), $50k account,
Robinhood costs, no hedging, one decision per day at 15:50 ET on the Alpaca paper account. The rules
below were written down **before** the first run. `sota scorecard` and the web page check them
(`src/sota/scorecard.py`).

| # | rule | limit |
|---|---|---|
| C1 | total return after costs | more than 0 at 6 months; more than ~4% (cash rate) at 12 months |
| C2 | maximum drawdown, daily closes | less than 15% at every point |
| C3 | number of trades (per year, counted after 60 days) | 20 to 50 |
| C4 | worst calendar month | better than −8% |

If two or more rules fail at 6 months, the test ends. All four must pass at 12 months before any
real money is used, and then only with a small part of the account. If the setup changes, the clock
starts again from zero.

## Results — what we found (2026-10-14)

This part is for a person who wants to know one thing: does the idea from the paper make money for
a normal retail account? Short answer: **no, not in our tests.** Below is how we got to that answer.

### How we tested

- Real option prices from Theta Data (end of day) for SPY, NVDA and TSLA, from June 2023 to
  October 2026. The data is not adjusted for stock splits (NVDA did a 10:1 split in June 2024), so
  the code handles that.
- Three test years. Year 0 = Sep 2023 to Aug 2024. Year 1 = Sep 2024 to Aug 2025 (this one contains
  the paper's own test period, Mar to Aug 2025). Holdout = Oct 2025 to Oct 2026. The holdout year
  is after the paper was written, so no model could have seen it before.
- A retail setup: $50,000 account, Robinhood fees, no hedging with shares, and every fill pays half
  of the bid-ask spread. The paper uses $1,000,000, hedging and lower costs. We tested that setup
  too (see below).
- Three selectors. `threshold` = the rules from the paper (Table 8). `llm` = Grok 4.7, given the
  same state and the same list of strategies as in the paper; it never sees prices, tickers or
  dates. `oracle` = it can look at the future, so it shows the best result possible. It is not a
  strategy. It is the ceiling.
- All numbers are total return after costs for the year. MDD = maximum drawdown = the biggest fall
  from a high point to a low point.

### The numbers

| selector | Year 0 (Sep 23 – Aug 24) | Year 1 (Sep 24 – Aug 25) | Holdout (Oct 25 – Oct 26) |
|---|---|---|---|
| threshold rules | −51.7% (MDD 52%) | +24.4% (MDD 18%) | −14.9% (MDD 29%) |
| LLM, all families | **−33.6%** (MDD 36%) | +9.8% (MDD 26%) | +31.6% (MDD 11%) |
| LLM, no iron condors | **−51.2%** (MDD 54%) | +28.5% (MDD 14%) | +11.9% (MDD 20%) |
| oracle (ceiling) | +186% | +153% | +88% (first half only) |

All rows use the same $50k retail setup and the same list of strategies (the nine families from the
paper, minus long straddles and strangles, because those do not make sense without hedging). The
oracle number for the holdout covers only Oct 2025 to Apr 2026, because the oracle needs to see up
to 180 days into the future and our data ends in Oct 2026.

Every real selector has at least one bad year. The two LLM rows are the important ones. Both lose
between one third and one half of the account in Year 0. You cannot put money in a strategy that
can do that.

### Why it loses

We exported every trade from every run (`sota backtest --out`) and looked at where the money went.

1. **The LLM sells volatility, no matter which strategies you allow.** With all families allowed,
   about 40 of its ~65 trades per year are iron condors (sell a call spread and a put spread far
   from the price). Those made −$11.9k in Year 0, −$5.4k in Year 1 and +$8.9k in the holdout. When
   we took iron condors away, it moved to iron butterflies (same idea, but sell at the current
   price, buy wings for protection) for 31% to 58% of its trades. Those made −$21.2k, +$8.0k and
   +$6.1k. Same bet, different name, worse result.
2. **Selling volatility works in quiet years and breaks in trending years.** In Year 0, NVDA went
   up 3x and TSLA moved 10% in one week around earnings. When this happens, the position jumps past
   the stop loss between one close and the next, so the loss is much bigger than planned. In Year 0,
   32 trades hit the stop loss. Every one was a loss. Total: −$23k.
3. **The stop loss is not the problem.** We replayed the same trades with a wider stop and with no
   stop. Both were worse in Year 0 and Year 1. The 60% stop does its job.
4. **The good years were not skill.** The trades that made money in Year 1 and the holdout were
   long calls and long call spreads on SPY, NVDA and TSLA, during a strong bull market. Every
   bearish trade the LLM made lost money (0 wins out of 6 long puts). The market went up. The model
   was not right; it was carried.

### Does a bigger account help?

No. We replayed the same trades with $50k, $100k, $250k and $1M. Year 0 is between −51% and −55% at
every size. The percent does not change. A bigger account only loses more dollars.

Hedging is a different story. With $1M, hedging with shares and the paper's low costs (the exact
setup of the paper), Year 0 becomes −11%. But Year 1 becomes +10% and the holdout becomes −2.5%.
Over three years that is about −4% in total, with a Sharpe ratio near zero. Hedging removes the
disaster, but it also removes the profit. What is left is worse than a savings account.

### Is the paper wrong?

Not exactly. Over the paper's own six-month period (Mar to Aug 2025), our environment gives similar
numbers. The paper reports +18.3% for its trained agent. Our oracle shows that +20% was possible in
that period with the paper's setup, and our LLM trades replayed in the paper's setup give +10%. So
the paper's number is believable. The problem is the period. Six months is too short, and those six
months were a good time for selling volatility. Run the same idea over three years and there is no
edge. We did not run the authors' own trained model, so we cannot say *their* model fails. We can
say that the framework with a strong general LLM does not survive a trending year, and no selector
we built did either.

### What we take from this

- The paper is a useful **framework**: the features, the strategy families, the resolvers, the
  risk limits, the exits, the metrics. All of this is implemented here, tested, and works. The
  oracle numbers prove the environment is correct: there is money to make every year, but the
  selector cannot find it.
- The paper is **not a strategy** you can trade with a retail account. Do not put money in it.
- The first day of paper trading (2026-10-14) also found two problems that only show up with real
  orders. Orders sent at the mid price did not fill in the last ten minutes of the session (fixed:
  orders now go out at the price the backtest assumes). Close orders were trusted without checking
  (fixed: if a close does not fill, the position goes back to open). Both fixes are in this repo.
- Total cost to learn all this: about $120 in LLM credits, and no money lost in the market.

### Ideas we did not test, and why we stopped

- **Only buy options, never sell premium.** This part of the book made money in two of three years
  (about +4% per year). But all of it came from bullish trades in a bull market. Just holding SPY
  did much better in the same period. A proper test costs about $40 in LLM credits. We think the
  answer is already clear.
- **A filter for trending markets** (stop trading when the market trends strongly). This goes
  after the real problem. But we would design it after seeing the data, so it would need its own
  clean test on new data. It is cheap to try with the replay tool (`-s replay:...`) if someone
  wants to.

### Reproduce

```bash
uv run sota fetch 2023-06-01 2026-10-09                      # Theta Data free tier, ~2 hours
uv run sota backtest 2023-09-01 2024-08-30 -s threshold -s llm -s oracle --out data/bt/y0
uv run sota backtest 2024-09-03 2025-08-29 -s threshold -s llm -s oracle --out data/bt/y1
uv run sota backtest 2025-10-13 2026-10-09 -s threshold -s llm --out data/bt/holdout
scripts/replay_grid.sh                                        # family / stop variants, from cache
uv run sota backtest 2023-09-01 2024-08-30 -s replay:data/bt/y0/llm_proposals.json --nav 1000000
```

The LLM runs cost about $12 to $15 per year with `grok-4.7` at low reasoning effort. Every reply is
cached under `data/llm_cache/`, so running the same test again is free. The trade-by-trade CSV
files behind the tables above are in `data/bt/`.

## Paper schedule

`launchd/com.sota.paper.plist` runs `scripts/paper_run.sh` at 15:45 and 16:45 local time; the
script keeps whichever firing falls in the 15:50 America/New_York window, so US daylight-saving
changes never move the decision time. One run per trading day; logs in `data/paper/run_<date>.log`.
The Mac must be awake at that time (`pmset repeat wakeorpoweron`).

Install (the plist has `__REPO__` where the repo path goes):

```bash
sed "s#__REPO__#$PWD#g" launchd/com.sota.paper.plist > ~/Library/LaunchAgents/com.sota.paper.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.sota.paper.plist
```

Pause with `launchctl bootout gui/$(id -u)/com.sota.paper`.

Theta Data transport: `SOTA_THETA__TRANSPORT=library` (default) talks to Theta's servers directly
through the `thetadata` package using `SOTA_THETA__API_KEY` (or `THETADATA_API_KEY`); `rest` uses a
locally running Theta Terminal v3 at `SOTA_THETA__BASE_URL`. Both yield identical chains, and every
fetched day is cached under `data/theta/` so repeated backtests run offline.

Layout: `src/sota/{features,strategies,resolvers,risk,pipeline}.py` are the paper's mechanics;
`backtest/` the environment and metrics; `selectors/` the policies; `data/` Theta/Alpaca/synthetic
sources; `execution/` the paper broker and manual-ticket adapter; `runner.py` the daily cycle;
`journal.py` the SQLite record of every proposal, ticket, fill and NAV.

