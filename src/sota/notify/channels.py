"""Notification channels: Telegram, email (SMTP), macOS desktop."""

from __future__ import annotations

import html
import logging
import smtplib
import subprocess
from email.message import EmailMessage

import httpx

from sota.config import NotifySettings
from sota.notify.base import LogNotifier, MultiNotifier, Notifier

# httpx logs every request URL at INFO, and Telegram's URL carries the bot token.
logging.getLogger("httpx").addFilter(lambda rec: "api.telegram.org" not in rec.getMessage())


class TelegramError(RuntimeError):
    pass


class TelegramNotifier:
    name = "telegram"

    def __init__(self, token: str, chat_id: str, client: httpx.Client | None = None) -> None:
        self.url = f"https://api.telegram.org/bot{token}/sendMessage"
        self.chat_id = chat_id
        self.client = client or httpx.Client(timeout=15)

    def send(self, title: str, body: str) -> None:
        # HTML mode: unlike Markdown, underscores and asterisks in tickers/ids never break a message.
        text = f"<b>{html.escape(title)}</b>\n<pre>{html.escape(body[:3800])}</pre>"
        try:
            r = self.client.post(
                self.url, json={"chat_id": self.chat_id, "text": text, "parse_mode": "HTML"}
            )
        except httpx.HTTPError as exc:  # the exception text would include the token-bearing URL
            raise TelegramError(f"{type(exc).__name__} while sending") from None
        if r.status_code >= 400:
            detail = (
                r.json().get("description", "")
                if "json" in r.headers.get("content-type", "")
                else ""
            )
            raise TelegramError(f"HTTP {r.status_code} {detail}".strip())


def telegram_chat_ids(token: str, client: httpx.Client | None = None) -> list[tuple[str, str]]:
    """(chat_id, display name) for every chat that has messaged the bot recently."""
    client = client or httpx.Client(timeout=15)
    r = client.get(f"https://api.telegram.org/bot{token}/getUpdates")
    r.raise_for_status()
    seen: dict[str, str] = {}
    for upd in r.json().get("result", []):
        msg = upd.get("message") or upd.get("channel_post") or {}
        chat = msg.get("chat") or {}
        if "id" in chat and str(chat["id"]) not in seen:
            name = chat.get("title") or " ".join(
                x for x in (chat.get("first_name"), chat.get("last_name")) if x
            )
            seen[str(chat["id"])] = name or chat.get("username", "")
    return sorted(seen.items())


class EmailNotifier:
    name = "email"

    def __init__(
        self, host: str, port: int, user: str | None, password: str | None, sender: str, to: str
    ) -> None:
        self.host, self.port, self.user, self.password, self.sender, self.to = (
            host,
            port,
            user,
            password,
            sender,
            to,
        )

    def send(self, title: str, body: str) -> None:
        msg = EmailMessage()
        msg["Subject"] = title
        msg["From"] = self.sender
        msg["To"] = self.to
        msg.set_content(body)
        with smtplib.SMTP(self.host, self.port, timeout=20) as s:
            s.starttls()
            if self.user and self.password:
                s.login(self.user, self.password)
            s.send_message(msg)


class MacOSNotifier:
    name = "macos"

    def send(self, title: str, body: str) -> None:
        first_line = body.splitlines()[0] if body else ""
        script = 'display notification "{}" with title "{}"'.format(
            first_line.replace("\\", "\\\\").replace('"', '\\"'),
            title.replace("\\", "\\\\").replace('"', '\\"'),
        )
        subprocess.run(["osascript", "-e", script], check=False, capture_output=True, timeout=10)


def build_notifier(settings: NotifySettings, include_log: bool = True) -> Notifier:
    channels: list[Notifier] = []
    if settings.telegram_bot_token and settings.telegram_chat_id:
        channels.append(
            TelegramNotifier(
                settings.telegram_bot_token.get_secret_value(), settings.telegram_chat_id
            )
        )
    if settings.smtp_host and settings.email_from and settings.email_to:
        channels.append(
            EmailNotifier(
                settings.smtp_host,
                settings.smtp_port,
                settings.smtp_user,
                settings.smtp_password.get_secret_value() if settings.smtp_password else None,
                settings.email_from,
                settings.email_to,
            )
        )
    if settings.macos:
        channels.append(MacOSNotifier())
    if include_log or not channels:
        channels.append(LogNotifier())
    return MultiNotifier(channels)
