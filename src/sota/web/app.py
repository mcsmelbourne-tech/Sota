"""Minimal web UI: tickets, positions, NAV, and ticket confirmation."""

from __future__ import annotations

from datetime import date

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from sota.config import Settings
from sota.execution.manual import ManualTicketBroker, format_ticket
from sota.journal import Journal, position_to_dict, ticket_to_dict
from sota.notify.base import LogNotifier

app = FastAPI(title="SOTA cockpit")


def _journal() -> Journal:
    return Journal(Settings().journal_path)


class Confirm(BaseModel):
    price: float
    units: int | None = None


@app.get("/api/tickets")
def tickets(status: str | None = "pending") -> list[dict]:
    return [
        {**ticket_to_dict(t), "text": format_ticket(t)} for t in _journal().list_tickets(status)
    ]


@app.get("/api/positions")
def positions(mode: str | None = None) -> list[dict]:
    s = Settings()
    return [position_to_dict(p) for p in _journal().list_positions(mode or s.mode.value, "open")]


@app.get("/api/nav")
def nav(mode: str | None = None) -> list[dict]:
    s = Settings()
    return [
        {"date": d.isoformat(), "nav": v} for d, v in _journal().nav_series(mode or s.mode.value)
    ]


@app.post("/api/tickets/{ticket_id}/confirm")
def confirm(ticket_id: str, body: Confirm) -> dict:
    s = Settings()
    broker = ManualTicketBroker(_journal(), LogNotifier(), s.initial_nav, s.effective_costs)
    try:
        pos = broker.confirm(ticket_id, body.price, body.units, date.today())
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return position_to_dict(pos)


@app.post("/api/tickets/{ticket_id}/skip")
def skip(ticket_id: str) -> dict:
    s = Settings()
    broker = ManualTicketBroker(_journal(), LogNotifier(), s.initial_nav, s.effective_costs)
    try:
        return ticket_to_dict(broker.skip(ticket_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


INDEX_HTML = """<!doctype html><html><head><meta charset="utf-8"><title>SOTA cockpit</title>
<style>body{font:14px system-ui;margin:2rem;max-width:1100px}pre{background:#f4f4f4;padding:.6rem;white-space:pre-wrap}
section{margin-bottom:2rem}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:.3rem .6rem;text-align:left}
button{margin-right:.4rem}input{width:6rem}</style></head><body>
<h1>SOTA cockpit</h1>
<section><h2>Pending tickets</h2><div id="tickets"></div></section>
<section><h2>Open positions</h2><table id="positions"></table></section>
<section><h2>NAV</h2><table id="nav"></table></section>
<script>
async function load(){
  const t=await (await fetch('/api/tickets')).json();
  document.getElementById('tickets').innerHTML=t.length?t.map(x=>`<pre>${x.text}</pre>
   <div>fill price <input id="p-${x.id}" type="number" step="0.01"> units <input id="u-${x.id}" type="number" value="${x.units}">
   <button onclick="confirmT('${x.id}')">confirm</button><button onclick="skipT('${x.id}')">skip</button></div>`).join(''):'<p>none</p>';
  const p=await (await fetch('/api/positions')).json();
  document.getElementById('positions').innerHTML='<tr><th>id</th><th>opened</th><th>underlying</th><th>family</th><th>units</th><th>entry</th><th>force close</th></tr>'+
   p.map(x=>`<tr><td>${x.id}</td><td>${x.entry_date}</td><td>${x.package.underlying}</td><td>${x.package.family}/${x.package.side}</td><td>${x.units}</td><td>${x.entry_premium.toFixed(2)}</td><td>${x.force_close_date}</td></tr>`).join('');
  const n=await (await fetch('/api/nav')).json();
  document.getElementById('nav').innerHTML='<tr><th>date</th><th>nav</th></tr>'+n.slice(-20).map(x=>`<tr><td>${x.date}</td><td>${x.nav.toLocaleString()}</td></tr>`).join('');
}
async function confirmT(id){const price=parseFloat(document.getElementById('p-'+id).value);const units=parseInt(document.getElementById('u-'+id).value);
  const r=await fetch(`/api/tickets/${id}/confirm`,{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({price,units})});
  if(!r.ok){alert((await r.json()).detail)} load();}
async function skipT(id){await fetch(`/api/tickets/${id}/skip`,{method:'POST'});load();}
load();setInterval(load,30000);
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_HTML
