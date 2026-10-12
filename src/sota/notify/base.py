"""Notification interface and fan-out."""

from __future__ import annotations

import logging
from typing import Protocol

log = logging.getLogger(__name__)


class Notifier(Protocol):
    name: str

    def send(self, title: str, body: str) -> None: ...


class LogNotifier:
    name = "log"

    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []

    def send(self, title: str, body: str) -> None:
        self.sent.append((title, body))
        log.info("%s\n%s", title, body)


class MultiNotifier:
    """Delivers to every channel; one failing channel never blocks the others."""

    name = "multi"

    def __init__(self, channels: list[Notifier]) -> None:
        self.channels = channels

    def send(self, title: str, body: str) -> None:
        for ch in self.channels:
            try:
                ch.send(title, body)
            except Exception as exc:  # noqa: BLE001
                log.warning("notifier %s failed: %s", ch.name, exc)
