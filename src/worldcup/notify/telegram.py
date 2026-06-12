"""Telegram push for the daily radar digest.

Reads TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID from the environment or
configs/secrets.env (same convention as the other API keys). Sends the text as
HTML <pre> blocks so the monospace tables/odds stay aligned on mobile, chunked
to Telegram's 4096-char limit. **No-ops cleanly when the token/chat aren't set**,
so it's safe to wire into daily_refresh.sh behind `|| true`.

Setup (one-time, the user does this — I never handle the credential creation):
  1. In Telegram, message @BotFather → /newbot → copy the bot token.
  2. Send your new bot any message (e.g. "hi") so it has an update to read.
  3. Put TELEGRAM_BOT_TOKEN=... in configs/secrets.env, then:
       python -m worldcup.notify.telegram --get-chat-id     # discovers your chat id
     add TELEGRAM_CHAT_ID=... to configs/secrets.env, then:
       python -m worldcup.notify.telegram --test            # confirms it works

Usage:
  echo "hi" | python -m worldcup.notify.telegram            # stdin
  python -m worldcup.notify.telegram --file path/to/msg.txt # file
  python -m worldcup.notify.telegram --text "ping" --silent # inline, no sound
"""
from __future__ import annotations

import argparse
import html
import os
import sys
import time
from pathlib import Path

import httpx

SECRETS_PATH = Path(__file__).resolve().parents[3] / "configs" / "secrets.env"
API = "https://api.telegram.org"
# Telegram hard limit is 4096 chars/message; chunk well under it because HTML
# escaping (& → &amp;) and the <pre>…</pre> wrapper add characters.
CHUNK = 3500


def _secret(key: str) -> str | None:
    """env var wins, else parse configs/secrets.env (KEY=value, # comment)."""
    if os.environ.get(key):
        return os.environ[key].strip()
    try:
        for line in SECRETS_PATH.read_text().splitlines():
            line = line.strip()
            if line.startswith(key + "="):
                return line.split("=", 1)[1].split("#")[0].strip()
    except FileNotFoundError:
        pass
    return None


def _chunks(text: str, n: int = CHUNK) -> list[str]:
    """Split on line boundaries (never mid-line); hard-split any monster line."""
    out: list[str] = []
    buf = ""
    for line in text.splitlines(keepends=True):
        while len(line) > n:
            if buf:
                out.append(buf)
                buf = ""
            out.append(line[:n])
            line = line[n:]
        if len(buf) + len(line) > n:
            out.append(buf)
            buf = line
        else:
            buf += line
    if buf:
        out.append(buf)
    return out or [""]


def _chunks_html(text: str, n: int = CHUNK) -> list[str]:
    """Split an HTML digest on blank-line (card/section) boundaries so a chunk never
    cuts through a <blockquote> or other tag. Each block is self-contained HTML; if a
    single block somehow exceeds n it's emitted whole (Telegram will reject only that
    one — acceptable vs. splitting mid-tag and corrupting every following message)."""
    blocks = text.split("\n\n")
    out: list[str] = []
    buf = ""
    for blk in blocks:
        cand = blk if not buf else buf + "\n\n" + blk
        if len(cand) > n and buf:
            out.append(buf)
            buf = blk
        else:
            buf = cand
    if buf:
        out.append(buf)
    return out or [""]


def send_message(text: str, *, token: str | None = None, chat_id: str | None = None,
                 silent: bool = False, as_code: bool = True,
                 as_html: bool = False) -> tuple[bool, str]:
    """Send `text` to the configured chat. Returns (ok, human_message).

    Three modes:
      as_html=True  → text is already Telegram HTML (cards/blockquotes); send as-is,
                      chunked on card boundaries. (the rich digest)
      as_code=True  → wrap each chunk in <pre> monospace (the legacy aligned tables).
      else          → plain text.
    ok=False with a 'not set' message means it was skipped (unconfigured), which
    callers in the daily pipeline treat as a no-op rather than an error."""
    token = token or _secret("TELEGRAM_BOT_TOKEN")
    chat_id = chat_id or _secret("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return False, "TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set — skipped (no-op)."

    parts = _chunks_html(text) if as_html else _chunks(text)
    for i, part in enumerate(parts, 1):
        if as_html:
            body = part
        elif as_code:
            body = f"<pre>{html.escape(part)}</pre>"
        else:
            body = part
        payload = {
            "chat_id": chat_id,
            "text": body,
            "disable_web_page_preview": True,
            "disable_notification": silent,
        }
        if as_code or as_html:
            payload["parse_mode"] = "HTML"
        # Retry transient network / 5xx blips (we have hit SSL EOF in the wild);
        # bail immediately on 4xx (bad token/chat — retrying won't help).
        last_err = None
        for attempt in range(3):
            try:
                r = httpx.post(f"{API}/bot{token}/sendMessage", json=payload, timeout=30)
            except httpx.HTTPError as e:
                last_err = f"network error: {type(e).__name__}: {e}"
            else:
                if r.status_code == 200:
                    last_err = None
                    break
                if r.status_code == 429:
                    # Rate-limited, not fatal: Telegram says how long to back off.
                    try:
                        wait = int(r.json().get("parameters", {}).get("retry_after", 3))
                    except Exception:
                        wait = 3
                    last_err = f"HTTP 429: rate limited (retry_after={wait}s)"
                    if attempt < 2:
                        time.sleep(min(wait, 30) + 1)
                    continue
                if 400 <= r.status_code < 500:
                    return False, (f"Telegram API error on part {i}/{len(parts)}: "
                                   f"HTTP {r.status_code}: {r.text[:200]}")
                last_err = f"HTTP {r.status_code}: {r.text[:120]}"
            if attempt < 2:
                time.sleep(1.5)
        if last_err:
            return False, f"failed on part {i}/{len(parts)} after 3 tries: {last_err}"
        if i < len(parts):
            time.sleep(1)  # official limit ≈1 msg/s per chat — pace multi-part digests
    return True, f"sent {len(parts)} message(s) to chat {chat_id}."


def discover_chat_id(token: str | None = None) -> tuple[bool, str]:
    """Call getUpdates and list chats that have messaged the bot recently."""
    token = token or _secret("TELEGRAM_BOT_TOKEN")
    if not token:
        return False, "TELEGRAM_BOT_TOKEN not set in configs/secrets.env."
    try:
        r = httpx.get(f"{API}/bot{token}/getUpdates", timeout=30).json()
    except httpx.HTTPError as e:
        return False, f"network error: {type(e).__name__}: {e}"
    if not r.get("ok"):
        return False, f"Telegram API error: {str(r)[:200]}"
    seen: dict = {}
    for u in r.get("result", []):
        msg = u.get("message") or u.get("channel_post") or u.get("edited_message") or {}
        chat = msg.get("chat", {})
        cid = chat.get("id")
        if cid is not None:
            label = chat.get("title") or chat.get("username") or chat.get("first_name") or "?"
            seen[cid] = f"{label}  ({chat.get('type', '?')})"
    if not seen:
        return False, ("No chats found. Send your bot a message first (open the bot in "
                       "Telegram and type anything), then re-run --get-chat-id.")
    lines = ["Chats that have messaged your bot:"]
    for cid, label in seen.items():
        lines.append(f"  TELEGRAM_CHAT_ID={cid}    # {label}")
    lines.append("\nAdd the right one to configs/secrets.env, then run --test.")
    return True, "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Push text to Telegram (daily radar digest).")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--file", help="read message body from this file")
    src.add_argument("--text", help="inline message body")
    ap.add_argument("--silent", action="store_true", help="deliver without notification sound")
    ap.add_argument("--plain", action="store_true", help="send as plain text (no <pre> monospace)")
    ap.add_argument("--html", action="store_true",
                    help="body is already Telegram HTML (cards/blockquotes); send as-is")
    ap.add_argument("--get-chat-id", action="store_true",
                    help="discover your chat id from recent messages to the bot")
    ap.add_argument("--test", action="store_true", help="send a one-line test ping")
    args = ap.parse_args()

    if args.get_chat_id:
        ok, msg = discover_chat_id()
        print(msg)
        sys.exit(0 if ok else 1)

    if args.test:
        text = "✅ WC2026 radar — Telegram wired up. Your daily digest will arrive here."
    elif args.file:
        text = Path(args.file).read_text()
    elif args.text:
        text = args.text
    else:
        text = sys.stdin.read()

    ok, msg = send_message(text, silent=args.silent, as_html=args.html,
                           as_code=not (args.plain or args.html))
    print(msg)
    # Skipped-because-unconfigured is a clean no-op (exit 0); real failures exit 1.
    sys.exit(0 if (ok or "not set" in msg) else 1)


if __name__ == "__main__":
    main()
