"""
Naval strike ingest — Ukrainian → Russian Black Sea Fleet & coastal targets.

Mirror of telegram_ingest.py but for the Ukraine→Russia naval-drone campaign.
Scrapes a Telegram channel's public HTML preview (default: @GeneralStaffZSU),
matches messages that reference a naval strike using Ukrainian keyword
patterns, extracts the target, and upserts to data/naval_events.csv.

The events log is deliberately verbose (raw text preserved) so a human can
audit any false positive. Compared to the air pipeline, naval strikes are
orders of magnitude rarer (tens per year vs 100s per night) — every row is
meaningful.

CSV schema (naval_events.csv):
    posted_at         ISO timestamp of the source post (UTC)
    target            Matched target name (from naval_targets.csv) or "Unknown"
    drones_launched   Extracted count if the post says "N drones"; else NaN
    action            "hit" / "damaged" / "destroyed" / "sunk" / "attempted"
    ship_hit          Ship name if identifiable (e.g. "Ivanovets"); else empty
    source_channel    e.g. "GeneralStaffZSU"
    message_id        Telegram message ID (idempotency key)
    text              Truncated raw text (first 400 chars, newlines stripped)

Design constraints:
  * Idempotent on message_id (re-scraping the same window updates in place)
  * Never fail the parent sync loop — return counts, swallow soft errors
  * Language: source text is Ukrainian; patterns are Cyrillic
"""
from __future__ import annotations
import re
import html as _html
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests


# -----------------------------------------------------------------------------
# Source configuration — swap CHANNEL_HANDLE to point at any public channel.
# -----------------------------------------------------------------------------
CHANNEL_HANDLE = "GeneralStaffZSU"
CHANNEL_URL    = f"https://t.me/s/{CHANNEL_HANDLE}"
UA_HEADER      = {"User-Agent": "Mozilla/5.0 (compatible; naval-monitor/1.0)"}


# -----------------------------------------------------------------------------
# Language patterns
# -----------------------------------------------------------------------------
# Any of these markers in a message qualify it as a NAVAL-strike report.
# Two tiers:
#   Tier 1 (strong): explicit USV/sea-drone platform reference
#   Tier 2 (medium): naval target-type words (ship, boat, tanker, fleet, port)
# We combine both here — the strike-verb filter downstream still applies, so
# a message needs BOTH a naval marker AND a strike verb to become an event.
NAVAL_MARKERS = re.compile(
    # Tier 1 — explicit sea-drone platforms
    r"(морськ\w*\s+дрон|надводн\w*\s+безекіпажн\w*|"
    r"безекіпажн\w*\s+катер\w*|БЕК|USV|Magura|Sea\s*Baby|Marichka|"
    # Tier 2 — naval target words (ship, boat, tanker, fleet, port, refinery)
    r"корабл\w*|катер\w*|танкер\w*|буксир\w*|сторожов\w*|"
    r"фрегат\w*|десантн\w*|підводн\w*|порт\w*|НПЗ|нафт\w+\s+термінал)",
    re.IGNORECASE | re.UNICODE,
)

# Ship-strike action verbs — must be present for a positive strike event.
STRIKE_VERBS = re.compile(
    r"(ураж\w+|знищ\w+|пошкодж\w+|потопл\w+|атак\w+|удар\w+|"
    r"постраждал\w+|збит\w+)",
    re.IGNORECASE | re.UNICODE,
)

# Target names — Ukrainian spellings (with English fallback for OSINT reposts).
# Keys must match target column in naval_targets.csv.
TARGET_PATTERNS = [
    (re.compile(r"Севастопол\w*|Sevastopol",       re.IGNORECASE | re.UNICODE), "Sevastopol"),
    (re.compile(r"Новоросійськ\w*|Novorossiysk",   re.IGNORECASE | re.UNICODE), "Novorossiysk"),
    (re.compile(r"Феодосі[яйю]\w*|Feodosia",       re.IGNORECASE | re.UNICODE), "Feodosia"),
    (re.compile(r"Керч\w*|Kerch",                  re.IGNORECASE | re.UNICODE), "Kerch Strait"),
    (re.compile(r"Ялт[аи]\w*|Yalta",               re.IGNORECASE | re.UNICODE), "Yalta"),
    (re.compile(r"Балаклав\w*|Balaklava",          re.IGNORECASE | re.UNICODE), "Balaklava"),
    (re.compile(r"Анап[аи]\w*|Anapa",              re.IGNORECASE | re.UNICODE), "Anapa"),
    (re.compile(r"Туапс\w*|Tuapse",                re.IGNORECASE | re.UNICODE), "Tuapse"),
    (re.compile(r"Сочі\w*|Sochi",                  re.IGNORECASE | re.UNICODE), "Sochi"),
]

# Extract drone counts when phrased as "N морських дронів" or "N Magura".
DRONE_COUNT = re.compile(
    r"(\d+)\s*(?:морськ|Magura|Sea\s*Baby|БЕК|USV)",
    re.IGNORECASE | re.UNICODE,
)

# Extract ship names when phrased as "патрульний катер 'Ivanovets'" etc.
# Highly incomplete — Ukrainian reports often just say "патрульний катер".
SHIP_NAME = re.compile(
    r"[«\"']([A-ZА-ЯЇЄІ][A-Za-zА-Яа-яЇїЄєІі\s-]{2,25})[»\"']",
    re.UNICODE,
)


# -----------------------------------------------------------------------------
# Fetch + parse (same technique as telegram_ingest.py: public /s/ preview HTML)
# -----------------------------------------------------------------------------
def fetch_channel_html(url: str = CHANNEL_URL, timeout: int = 30) -> str:
    r = requests.get(url, headers=UA_HEADER, timeout=timeout)
    r.raise_for_status()
    return r.text


def fetch_history(max_pages: int = 12, timeout: int = 30) -> list[dict]:
    """Walk backward through the channel via the ?before= pagination trick."""
    seen_ids: set[int] = set()
    all_msgs: list[dict] = []
    url = CHANNEL_URL
    for _ in range(max_pages):
        resp = requests.get(url, headers=UA_HEADER, timeout=timeout)
        if resp.status_code != 200:
            break
        batch = _parse_batch(resp.text)
        new = [m for m in batch if m["id"] not in seen_ids]
        if not new:
            break
        for m in new:
            seen_ids.add(m["id"])
        all_msgs.extend(new)
        oldest_id = min(m["id"] for m in new)
        url = f"{CHANNEL_URL}?before={oldest_id}"
        time.sleep(0.5)  # polite delay — this is a scraped public preview
    all_msgs.sort(key=lambda m: m["datetime"] or datetime.min.replace(tzinfo=timezone.utc))
    return all_msgs


def _parse_batch(html: str) -> list[dict]:
    """Split channel preview HTML into per-message dicts.
    Same regex approach as telegram_ingest._parse_batch."""
    blocks = re.split(
        rf'(?=<div class="tgme_widget_message [^"]*"\s+data-post="{CHANNEL_HANDLE}/)',
        html,
    )
    out = []
    for b in blocks:
        mid_m = re.search(rf'data-post="{CHANNEL_HANDLE}/(\d+)"', b)
        if not mid_m:
            continue
        ts_m = re.search(r'<time[^>]*datetime="([^"]+)"', b)
        text_m = re.search(
            r'<div class="tgme_widget_message_text[^"]*"[^>]*>(.*?)</div>',
            b, re.S,
        )
        text = ""
        if text_m:
            raw = re.sub(r'<br\s*/?>', '\n', text_m.group(1))
            raw = re.sub(r'<[^>]+>', ' ', raw)
            text = _html.unescape(raw).strip()
        try:
            dt = datetime.fromisoformat(
                ts_m.group(1).replace("Z", "+00:00")
            ) if ts_m else None
        except (ValueError, AttributeError):
            dt = None
        out.append({"id": int(mid_m.group(1)), "datetime": dt, "text": text})
    return out


# -----------------------------------------------------------------------------
# Filter + extract
# -----------------------------------------------------------------------------
def _extract_target(text: str) -> str:
    for pat, name in TARGET_PATTERNS:
        if pat.search(text):
            return name
    return "Unknown"


def _extract_count(text: str) -> int | None:
    m = DRONE_COUNT.search(text)
    return int(m.group(1)) if m else None


def _classify_action(text: str) -> str:
    """Very coarse — cascade of most-severe → least-severe verbs."""
    if re.search(r"потопл\w+|потон\w+|sunk", text, re.IGNORECASE | re.UNICODE):
        return "sunk"
    if re.search(r"знищ\w+|destroyed", text, re.IGNORECASE | re.UNICODE):
        return "destroyed"
    if re.search(r"пошкодж\w+|damaged", text, re.IGNORECASE | re.UNICODE):
        return "damaged"
    if re.search(r"ураж\w+|hit", text, re.IGNORECASE | re.UNICODE):
        return "hit"
    return "attempted"


def _extract_ship(text: str) -> str:
    m = SHIP_NAME.search(text)
    return m.group(1) if m else ""


def messages_to_naval_events(messages: list[dict]) -> pd.DataFrame:
    """Filter naval-strike messages and extract event rows."""
    rows = []
    for m in messages:
        text = (m.get("text") or "").strip()
        if not text:
            continue
        if not NAVAL_MARKERS.search(text):
            continue
        if not STRIKE_VERBS.search(text):
            # Marker present but no strike verb — usually context/history mention
            continue
        rows.append({
            "posted_at":       m["datetime"].isoformat() if m["datetime"] else None,
            "target":          _extract_target(text),
            "drones_launched": _extract_count(text),
            "action":          _classify_action(text),
            "ship_hit":        _extract_ship(text),
            "source_channel":  CHANNEL_HANDLE,
            "message_id":      int(m["id"]),
            "text":            text[:400].replace("\n", " "),
        })
    return pd.DataFrame(rows, columns=[
        "posted_at", "target", "drones_launched", "action",
        "ship_hit", "source_channel", "message_id", "text",
    ])


def upsert_naval_events(new_rows: pd.DataFrame, csv_path) -> tuple[int, int]:
    """Idempotent merge by (source_channel, message_id). Returns (added, updated)."""
    csv_path = Path(csv_path)
    cols = ["posted_at", "target", "drones_launched", "action",
            "ship_hit", "source_channel", "message_id", "text"]
    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        for c in cols:
            if c not in existing.columns:
                existing[c] = None
        existing = existing[cols]
    else:
        existing = pd.DataFrame(columns=cols)

    if new_rows.empty:
        return 0, 0

    def _key(df):
        return df["source_channel"].astype(str) + "|" + df["message_id"].astype(str)

    ek = _key(existing) if not existing.empty else pd.Series([], dtype=str)
    nk = _key(new_rows)
    added   = int((~nk.isin(ek.tolist())).sum())
    updated = int(nk.isin(ek.tolist()).sum())

    kept = existing[~_key(existing).isin(nk.tolist())] if not existing.empty else existing
    combined = pd.concat([kept, new_rows], ignore_index=True)
    combined = combined.sort_values("posted_at", ascending=False)
    combined.to_csv(csv_path, index=False)
    return added, updated


# -----------------------------------------------------------------------------
# End-to-end sync (mirror of telegram_ingest.sync)
# -----------------------------------------------------------------------------
def sync(naval_events_csv, pages: int = 12) -> dict:
    """Fetch → filter → upsert. Returns run summary."""
    messages = fetch_history(max_pages=pages)
    rows = messages_to_naval_events(messages)
    added, updated = upsert_naval_events(rows, naval_events_csv)
    return {
        "channel":         CHANNEL_HANDLE,
        "messages_seen":   len(messages),
        "naval_matches":   len(rows),
        "events_added":    added,
        "events_updated":  updated,
        "date_range": (
            messages[0]["datetime"].date().isoformat() if messages else None,
            messages[-1]["datetime"].date().isoformat() if messages else None,
        ),
        "at": datetime.now().isoformat(timespec="seconds"),
    }


if __name__ == "__main__":
    import sys
    from pathlib import Path
    csv = Path(__file__).parent / "data" / "naval_events.csv"
    result = sync(csv, pages=6)
    print(result)
