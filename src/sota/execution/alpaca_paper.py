"""Alpaca paper-trading broker. `paper=True` is hard-coded: this adapter cannot touch a live account."""

from __future__ import annotations

from sota.config import AlpacaSettings
from sota.data.alpaca import _creds
from sota.execution.base import AccountInfo, Intent
from sota.models import Package

PAPER_HOST = "paper-api.alpaca.markets"


class AlpacaPaperBroker:
    mode = "paper"

    def __init__(self, settings: AlpacaSettings) -> None:
        from alpaca.trading.client import TradingClient

        key, secret = _creds(settings)
        self.client = TradingClient(key, secret, paper=True)
        base_url = str(getattr(self.client, "_base_url", PAPER_HOST))
        if PAPER_HOST not in base_url:
            raise RuntimeError(f"refusing to run: trading client is not pointed at {PAPER_HOST}")

    def account(self) -> AccountInfo:
        a = self.client.get_account()
        return AccountInfo(float(a.equity), float(a.cash))

    def submit_package(self, pkg: Package, units: int, limit_price: float, intent: Intent) -> str:
        from alpaca.trading.enums import OrderClass, OrderSide, PositionIntent, TimeInForce
        from alpaca.trading.requests import LimitOrderRequest, OptionLegRequest

        sign = 1 if intent == "open" else -1
        legs: list[OptionLegRequest] = []
        for leg in pkg.legs:
            ratio = leg.ratio * sign
            buy = ratio > 0
            if intent == "open":
                pi = PositionIntent.BUY_TO_OPEN if buy else PositionIntent.SELL_TO_OPEN
            else:
                pi = PositionIntent.BUY_TO_CLOSE if buy else PositionIntent.SELL_TO_CLOSE
            legs.append(
                OptionLegRequest(
                    symbol=leg.contract.occ_symbol,
                    ratio_qty=abs(ratio),
                    side=OrderSide.BUY if buy else OrderSide.SELL,
                    position_intent=pi,
                )
            )
        # Alpaca mleg convention: positive limit = net debit, negative = net credit.
        net = limit_price if intent == "open" else -limit_price
        px = round(net, 2)
        if len(legs) == 1:
            single = legs[0]
            req = LimitOrderRequest(
                symbol=single.symbol,
                qty=units * single.ratio_qty,
                side=single.side,
                time_in_force=TimeInForce.DAY,
                limit_price=max(abs(px), 0.01),
                position_intent=single.position_intent,
            )
        else:
            req = LimitOrderRequest(
                qty=units,
                order_class=OrderClass.MLEG,
                time_in_force=TimeInForce.DAY,
                limit_price=px,
                legs=legs,
            )
        order = self.client.submit_order(req)
        return str(order.id)

    def submit_shares(self, underlying: str, qty: int) -> str:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        if qty == 0:
            return ""
        req = MarketOrderRequest(
            symbol=underlying,
            qty=abs(qty),
            side=OrderSide.BUY if qty > 0 else OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )
        return str(self.client.submit_order(req).id)

    def order_status(self, order_id: str) -> tuple[str, float | None]:
        o = self.client.get_order_by_id(order_id)
        status = o.status.value if hasattr(o.status, "value") else str(o.status)
        price = float(o.filled_avg_price) if o.filled_avg_price else None
        return status, price

    def positions(self) -> list[dict[str, str]]:
        return [
            {
                "symbol": p.symbol,
                "qty": str(p.qty),
                "market_value": str(p.market_value),
                "asset_class": str(p.asset_class),
            }
            for p in self.client.get_all_positions()
        ]
