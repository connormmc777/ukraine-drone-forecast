"""
Ukrainian Air Force (@kpszsu) Telegram ingestion.

Pulls the public web preview of the official UA Air Force Telegram channel,
extracts drone-sighting messages, maps Ukrainian oblast names (incl. genitive
case forms like 'Полтавщини') to our CSV oblast names, and aggregates per
(date, oblast). Each row that lands in observations.csv represents the count
of drone-group sightings mentioning that oblast on that date — a defensible
proxy for launch/transit activity, with source='UA Air Force TG'.

No auth required — we fetch t.me/s/kpszsu (public preview, ~20 messages).
For historical backfill you'd need the MTProto API; recent ingest works
without any credentials.
"""
from __future__ import annotations
import re
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta, date as date_cls


def night_date_for(dt: datetime) -> date_cls:
    """Map a sighting timestamp to the 'night-of' date used by official UA AF
    summaries. Sightings after 15:00 UTC belong to the NEXT day's night
    summary; sightings before 15:00 UTC are scored against the SAME day's
    night summary (which covers ~18:00 UTC of D-1 through ~06:00 UTC of D)."""
    dt_utc = dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    if dt_utc.hour >= 15:
        return (dt_utc + timedelta(days=1)).date()
    return dt_utc.date()

CHANNEL_URL = "https://t.me/s/kpszsu"
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"

# Ukrainian month names in the genitive case as they appear in summaries:
# "У ніч на 12 травня" / "12 травня (з …)"
UA_MONTHS_GENITIVE = {
    'січня': 1, 'лютого': 2, 'березня': 3, 'квітня': 4,
    'травня': 5, 'червня': 6, 'липня': 7, 'серпня': 8,
    'вересня': 9, 'жовтня': 10, 'листопада': 11, 'грудня': 12,
}

# Each entry: (regex over the Ukrainian message text, our CSV oblast name).
# Order matters — longer/more specific patterns first so 'Київщина' matches
# 'Kyiv Oblast' before 'Києва' matches 'Kyiv City'. All patterns are
# case-insensitive.
OBLAST_PATTERNS = [
    # Kyiv: oblast vs city
    (r'Київщин[аиоюуі]|Київської області', 'Kyiv Oblast'),
    (r'Києв[аіоу]|у Київ\b|на Київ\b|Києві|Київ\s*[-–—]|Київ\.|Київ,', 'Kyiv City'),
    # Standard '-щина / -щини' oblast names
    (r'Харківщин[аиоюуі]|Харківськ|Харків|Харков', 'Kharkiv'),
    (r'Одещин[аиоюуі]|Одеськ|Одес[аиую]', 'Odesa'),
    (r'Львівщин[аиоюуі]|Львівськ|Львов?[аіу]', 'Lviv'),
    (r'Дніпропетровщин[аиоюуі]|Дніпропетровськ|Дніпр[оау]\b', 'Dnipropetrovsk'),
    (r'Донеччин[аиоюуі]|Донецьк[аоуі]?', 'Donetsk'),
    (r'Запорізьк|Запоріжж[ія]', 'Zaporizhzhia'),
    (r'Миколаївщин[аиоюуі]|Миколаївськ|Миколаїв|Миколаєв', 'Mykolaiv'),
    (r'Херсонщин[аиоюуі]|Херсонськ|Херсон', 'Kherson'),
    (r'Полтавщин[аиоюуі]|Полтавськ|Полтав[аиую]', 'Poltava'),
    (r'Сумщин[аиоюуі]|Сумськ|Сум[иау]\b', 'Sumy'),
    (r'Чернігівщин[аиоюуі]|Чернігівськ|Чернігів|Чернігов', 'Chernihiv'),
    (r'Вінниччин[аиоюуі]|Вінниц[яію]', 'Vinnytsia'),
    (r'Черкащин[аиоюуі]|Черкас[иау]', 'Cherkasy'),
    (r'Кіровоградщин[аиоюуі]|Кіровоград', 'Kirovohrad'),
    (r'Житомирщин[аиоюуі]|Житомир', 'Zhytomyr'),
    (r'Рівненщин[аиоюуі]|Рівнен|Рівне\b|на Рівне', 'Rivne'),
    (r'Волин[іьюя]', 'Volyn'),
    (r'Тернопільщин[аиоюуі]|Тернопіл', 'Ternopil'),
    (r'Хмельниччин[аиоюуі]|Хмельниць', 'Khmelnytskyi'),
    (r'Прикарпатт|Івано-Франківщин[аиоюуі]|Івано-Франківськ', 'Ivano-Frankivsk'),
    (r'Чернівецьк|Чернівц[іяю]|Буковин', 'Chernivtsi'),
    (r'Закарпатт|Закарпатськ', 'Zakarpattia'),
]

# Common city/town mentions that imply an oblast. Reports often say
# 'a drone toward Pavlohrad' rather than 'toward Dnipropetrovsk Oblast'.
CITY_TO_OBLAST = [
    (r'Павлоград[ауі]?', 'Dnipropetrovsk'),
    (r'Кривий Ріг|Кривому Розі|Кривого Рогу', 'Dnipropetrovsk'),
    (r'Кам.янське|Кам.янського', 'Dnipropetrovsk'),
    (r'Нікопол', 'Dnipropetrovsk'),
    (r'Охтирк[аи]', 'Sumy'),
    (r'Старокостянтинів|Старокостянтинова|Старокостянтинові', 'Khmelnytskyi'),
    (r'Чорноморськ[еаоу]?|Чорноморське', 'Odesa'),
    (r'Південне', 'Odesa'),
    (r'Миргород[ауі]?', 'Poltava'),
    (r'Кременчук[ауі]?', 'Poltava'),
    (r'Лубни', 'Poltava'),
    (r'Чигирин[ауі]?', 'Cherkasy'),
    (r'Умань|Умані', 'Cherkasy'),
    (r'Сміла|Сміли', 'Cherkasy'),
    (r'Яготин[ауі]?', 'Kyiv Oblast'),
    (r'Бровари', 'Kyiv Oblast'),
    (r'Біл[аоі] Церкв[аиі]', 'Kyiv Oblast'),
    (r'Ірпінь|Бучі|Буча', 'Kyiv Oblast'),
    (r'Фастів|Бориспіл', 'Kyiv Oblast'),
    (r'Бахмут|Авдіївка|Покровськ|Краматорськ|Слов.янськ|Костянтинівка', 'Donetsk'),
    (r'Маріуполь', 'Donetsk'),
    (r'Мелітополь|Бердянськ|Енергодар', 'Zaporizhzhia'),
    (r'Куп.янськ|Ізюм|Чугуїв', 'Kharkiv'),
    (r'Конотоп|Шостка|Глухів', 'Sumy'),
    (r'Ніжин|Прилуки', 'Chernihiv'),
    (r'Очаків|Вознесенськ|Коблеве|Коблево', 'Mykolaiv'),
    (r'Скадовськ|Каховк[аи]', 'Kherson'),
    (r'Ужгород|Мукачево', 'Zakarpattia'),
    (r'Дрогобич|Стрий|Червоноград', 'Lviv'),
    (r'Шепетівка|Кам.янець-Подільський', 'Khmelnytskyi'),
    (r'Бердичів|Коростень|Новоград-Волинський', 'Zhytomyr'),
    (r'Луцьк|Ковель', 'Volyn'),
    (r'Дубно|Сарни', 'Rivne'),
    (r'Чортків|Кременець', 'Ternopil'),
    (r'Калуш|Коломия', 'Ivano-Frankivsk'),
    (r'Олександрія|Світловодськ', 'Kirovohrad'),
    (r'Жмеринка|Хмільник', 'Vinnytsia'),
]

COMPILED_PATTERNS = [(re.compile(pat, re.IGNORECASE | re.UNICODE), name)
                     for pat, name in (OBLAST_PATTERNS + CITY_TO_OBLAST)]

# Drone-related markers — every sighted/launched UAV message uses 🛵 or
# explicitly mentions БпЛА. We exclude pure missile (🚀 / Ракета) reports.
DRONE_MARKERS = re.compile(r'(🛵|БпЛА|шахед|герань|Shahed|Geran)',
                            re.IGNORECASE | re.UNICODE)
MISSILE_ONLY_MARKERS = re.compile(r'^(🚀|Ракет)', re.UNICODE)


def fetch_channel_html(url: str = CHANNEL_URL, timeout: int = 30) -> str:
    resp = requests.get(url, headers={'User-Agent': USER_AGENT}, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def fetch_history(max_pages: int = 25, timeout: int = 30) -> list[dict]:
    """Walk backward through the public preview to collect older messages.
    Each call returns ~20 messages; max_pages=25 reaches roughly 500 messages
    (~2 weeks of channel activity during a busy war period).
    Returns a list of {'id', 'datetime', 'text'} sorted oldest-first."""
    headers = {'User-Agent': USER_AGENT}
    all_msgs: dict[int, dict] = {}
    cursor: int | None = None
    for _ in range(max_pages):
        url = (f"{CHANNEL_URL}?before={cursor}" if cursor else CHANNEL_URL)
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        batch = _parse_batch(resp.text)
        if not batch:
            break
        for m in batch:
            all_msgs[m['id']] = m
        new_oldest = min(m['id'] for m in batch)
        if cursor == new_oldest:
            break
        cursor = new_oldest
    return sorted(all_msgs.values(), key=lambda m: m['datetime'])


def _parse_batch(html: str) -> list[dict]:
    """Split a tg preview HTML into per-message dicts. Robust to either the
    front page (with trailing UI) or paginated responses."""
    blocks = re.split(
        r'(?=<div class="tgme_widget_message [^"]*"\s+data-post="kpszsu/)',
        html,
    )
    out = []
    for b in blocks:
        mid_m = re.search(r'data-post="kpszsu/(\d+)"', b)
        if not mid_m:
            continue
        ts_m = re.search(r'<time[^>]*datetime="([^"]+)"', b)
        text_m = re.search(
            r'<div class="tgme_widget_message_text[^"]*"[^>]*>(.*?)</div>',
            b, re.S,
        )
        text = ''
        if text_m:
            import html as H
            raw = re.sub(r'<br\s*/?>', '\n', text_m.group(1))
            raw = re.sub(r'<[^>]+>', ' ', raw)
            text = H.unescape(raw).strip()
        try:
            dt = datetime.fromisoformat(ts_m.group(1).replace('Z', '+00:00')) if ts_m else None
        except (ValueError, AttributeError):
            dt = None
        out.append({'id': int(mid_m.group(1)), 'datetime': dt, 'text': text})
    return out


def parse_messages(html: str) -> list[dict]:
    """Return a list of {'datetime', 'text'} for each message in the preview."""
    return [
        {'datetime': m['datetime'], 'text': m['text'], 'id': m['id']}
        for m in _parse_batch(html) if m['datetime'] is not None
    ]


# ---------- Morning/evening summary parsing ----------

# A summary headline matches all of these variants:
#   "ЗБИТО/ПОДАВЛЕНО 192 ВОРОЖІ БПЛА"
#   "ЗБИТО/ПОДАВЛЕНО БАЛІСТИЧНУ РАКЕТУ ТА 149 ВОРОЖИХ БПЛА"
#   "ЗБИТО/ПОДАВЛЕНО 175 ВОРОЖИХ БПЛА ТА 5 РАКЕТ"
#   "ЗБИТО/ПОДАВЛЕНО 4 РАКЕТИ ТА 503 ВОРОЖИХ БПЛА"   ← combined attack
#   "ЗБИТО/ПОДАВЛЕНО 41 РАКЕТУ ТА 652 ВОРОЖІ БПЛА"
# We anchor on the headline prefix and capture the number that
# immediately precedes "ВОРОЖ..." regardless of what came before.
_SUMMARY_HEADLINE = re.compile(
    # Match any number followed by ВОРОЖ (enemy drones) or ЦІЛ (targets,
    # used on combined-attack nights with missiles+drones+bombs) within
    # ~120 chars of ПОДАВЛЕНО.
    r'ЗБИТО.{0,5}ПОДАВЛЕНО.{0,120}?(\d+)\s+(?:ВОРОЖ|ЦІЛ)',
    re.IGNORECASE | re.UNICODE,
)
# Missile count appears either before ("4 РАКЕТИ ТА 503...") or after
# ("175 ВОРОЖИХ БПЛА ТА 5 РАКЕТ"). Try both directions.
_MISSILES_IN_HEADLINE_BEFORE = re.compile(
    r'ПОДАВЛЕНО\s+(\d+)\s+РАКЕТ', re.IGNORECASE | re.UNICODE,
)
_MISSILES_IN_HEADLINE_AFTER = re.compile(
    r'ВОРОЖ\w+\s+БПЛА\s+ТА\s+(\d+)\s+РАКЕТ', re.IGNORECASE | re.UNICODE,
)
# Night summary variants:
#   "У ніч на 12 травня (з 18:00 11 травня)"
#   "Протягом ночі (з 00:00 10 травня)"
_NIGHT_DATE = re.compile(
    r'(?:У\s+ніч\s+на|Протягом\s+ночі\s*\(\s*з\s+[\d:.]+)\s+(\d{1,2})\s+(\w+)',
    re.IGNORECASE | re.UNICODE,
)
# Day summary: "1 травня (з 08:00 по 15:30) противник атакував"
_DAY_DATE = re.compile(
    r'(?:^|\s)(\d{1,2})\s+(\w+)\s*\(\s*з\s+[\d:.]+\s+по\s+[\d:.]+\s*\)',
    re.IGNORECASE | re.UNICODE | re.MULTILINE,
)
_LAUNCHED = re.compile(
    r'противник\s+атакував\s+(?:(\d+)(?:-?ма)?)|атакував\s+(\d+)',
    re.IGNORECASE | re.UNICODE,
)
# Fallback: handles all the morphological variants seen in 2026:
#   "216 ударними БпЛА"     (plural instrumental — typical)
#   "141 ударним БпЛА"      (singular instrumental — May 15)
#   "524 ударних БпЛА"      (genitive plural — May 18)
#   "67-ма ударними БпЛА"   (with hyphenated suffix)
_LAUNCHED_LOOSE = re.compile(
    r'(\d+)\s*(?:-?ма\s+)?удар\w+\s+БпЛА', re.IGNORECASE | re.UNICODE,
)
_SHAHEDS = re.compile(
    r'(?:близько|понад)?\s*(\d+)\s+із\s+них\s*[–—\-–—]\s*[«"]?шахед',
    re.IGNORECASE | re.UNICODE,
)
_HITS = re.compile(
    r'Зафіксовано\s+влучання\s+(?:балістичної\s+ракети\s+та\s+)?(\d+)\s+ударних\s+БпЛА'
    r'(?:\s+на\s+(\d+)\s+локаціях)?',
    re.IGNORECASE | re.UNICODE,
)


def _resolve_summary_date(text: str, posted_at: datetime) -> tuple[date_cls | None, str]:
    """Returns (date, period) where period in {'night','day','unknown'}."""
    m = _NIGHT_DATE.search(text)
    if m:
        day, month_name = int(m.group(1)), m.group(2).lower()
        if month_name in UA_MONTHS_GENITIVE:
            return date_cls(posted_at.year, UA_MONTHS_GENITIVE[month_name], day), 'night'
    m = _DAY_DATE.search(text)
    if m:
        day, month_name = int(m.group(1)), m.group(2).lower()
        if month_name in UA_MONTHS_GENITIVE:
            return date_cls(posted_at.year, UA_MONTHS_GENITIVE[month_name], day), 'day'
    return None, 'unknown'


def parse_summary(msg: dict) -> dict | None:
    """If msg is an official UA AF summary, return its parsed numbers; else None."""
    text = msg['text']
    head = _SUMMARY_HEADLINE.search(text[:200])
    if not head:
        return None
    intercepted = int(head.group(1))
    # Missiles can be listed before or after the drones in the headline
    missiles_intercepted = 0
    m_before = _MISSILES_IN_HEADLINE_BEFORE.search(text[:300])
    m_after = _MISSILES_IN_HEADLINE_AFTER.search(text[:300])
    if m_before and int(m_before.group(1)) != intercepted:
        missiles_intercepted = int(m_before.group(1))
    elif m_after:
        missiles_intercepted = int(m_after.group(1))

    summary_date, period = _resolve_summary_date(text, msg['datetime'])

    launched = None
    m = _LAUNCHED_LOOSE.search(text)
    if m:
        launched = int(m.group(1))
    if launched is None:
        m = _LAUNCHED.search(text)
        if m:
            launched = int(m.group(1) or m.group(2))

    shahed = None
    m = _SHAHEDS.search(text)
    if m:
        shahed = int(m.group(1))

    hits, hit_locations = 0, 0
    m = _HITS.search(text)
    if m:
        hits = int(m.group(1))
        hit_locations = int(m.group(2)) if m.group(2) else 0

    return {
        'date': summary_date.isoformat() if summary_date else None,
        'period': period,
        'launched': launched,
        'intercepted': intercepted,
        'shaheds_estimated': shahed,
        'missiles_intercepted': missiles_intercepted,
        'hits': hits,
        'hit_locations': hit_locations,
        'posted_at': msg['datetime'].isoformat(),
        'message_id': msg.get('id'),
        'source': 'UA Air Force TG summary',
    }


def is_drone_message(text: str) -> bool:
    if not text:
        return False
    # If the message is solely a missile/rocket report, skip it
    if MISSILE_ONLY_MARKERS.match(text.strip()):
        # …unless it also has a drone marker (mixed report)
        if not DRONE_MARKERS.search(text):
            return False
    return bool(DRONE_MARKERS.search(text))


def extract_oblasts(text: str) -> list[str]:
    """Return the unique CSV oblast names mentioned in the text."""
    hits = []
    for pat, name in COMPILED_PATTERNS:
        if pat.search(text):
            if name not in hits:
                hits.append(name)
    return hits


def messages_to_observation_rows(messages: list[dict]) -> tuple[pd.DataFrame, list[dict]]:
    """Aggregate real-time sightings to one row per (date, oblast). Skips
    long-form summary messages (those are parsed separately).
    Returns (rows, unmatched_messages)."""
    sightings = []
    unmatched = []
    for m in messages:
        if not is_drone_message(m['text']):
            continue
        # Summary messages contain БпЛА too — exclude them from per-message
        # sighting aggregation.
        if _SUMMARY_HEADLINE.search(m['text'][:200]):
            continue
        oblasts = extract_oblasts(m['text'])
        if not oblasts:
            unmatched.append({
                'datetime': m['datetime'].isoformat(),
                'text': m['text'][:200],
            })
            continue
        d = night_date_for(m['datetime']).isoformat()
        for ob in oblasts:
            sightings.append({'observation_date': d, 'oblast': ob})

    if not sightings:
        return pd.DataFrame(columns=['observation_date', 'oblast',
                                     'observed_drones', 'source']), unmatched

    df = pd.DataFrame(sightings)
    agg = df.groupby(['observation_date', 'oblast']).size().reset_index(name='observed_drones')
    agg['source'] = 'UA Air Force TG'
    return agg[['observation_date', 'oblast', 'observed_drones', 'source']], unmatched


def messages_to_sightings_log(messages: list[dict]) -> pd.DataFrame:
    """One row per drone message with the ORIGINAL timestamp preserved
    (unlike messages_to_observation_rows which aggregates to per-night counts).

    Feeds the live "recent drone-track alerts" panel — where 'live' means
    minute-fresh per-message, not per-night aggregate."""
    rows = []
    for m in messages:
        if not is_drone_message(m['text']):
            continue
        if _SUMMARY_HEADLINE.search(m['text'][:200]):
            continue
        obs = extract_oblasts(m['text'])
        row = {
            'message_id': int(m['id']),
            'posted_at': m['datetime'].isoformat() if m['datetime'] else None,
            'oblast': obs[0] if obs else 'UNKNOWN',
            'all_oblasts': ','.join(obs) if obs else '',
            'text': (m['text'] or '')[:400].replace('\n', ' ').strip(),
        }
        rows.append(row)
    return pd.DataFrame(rows, columns=[
        'message_id', 'posted_at', 'oblast', 'all_oblasts', 'text',
    ])


def upsert_sightings(new_rows: pd.DataFrame, csv_path) -> tuple[int, int]:
    """Merge per-message sightings idempotently on message_id.
    Returns (added, updated)."""
    from pathlib import Path
    csv_path = Path(csv_path)
    cols = ['message_id', 'posted_at', 'oblast', 'all_oblasts', 'text']
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

    existing_ids = set(existing['message_id'].astype(int).tolist()) if not existing.empty else set()
    new_ids = set(new_rows['message_id'].astype(int).tolist())
    added = len(new_ids - existing_ids)
    updated = len(new_ids & existing_ids)

    # Replace overlapping rows with the fresh copy so text corrections propagate.
    kept = existing[~existing['message_id'].astype(int).isin(new_ids)]
    combined = pd.concat([kept, new_rows], ignore_index=True)
    combined = combined.sort_values('posted_at', ascending=False)
    combined.to_csv(csv_path, index=False)
    return added, updated


def merge_into_observations(new_rows: pd.DataFrame, csv_path) -> tuple[int, int]:
    """Idempotent merge: (date, oblast, source) is the unique key. Re-fetching
    the same window updates existing rows in place."""
    existing = pd.read_csv(csv_path)
    existing['observation_date'] = existing['observation_date'].astype(str)
    if new_rows.empty:
        return 0, 0
    key_cols = ['observation_date', 'oblast', 'source']
    existing_key = existing[key_cols].astype(str).agg('|'.join, axis=1)
    new_key = new_rows[key_cols].astype(str).agg('|'.join, axis=1)
    updated = int(existing_key.isin(new_key.tolist()).sum())
    kept = existing[~existing_key.isin(new_key.tolist())]
    combined = pd.concat([kept, new_rows], ignore_index=True)
    combined = combined.sort_values(['observation_date', 'oblast'])
    combined.to_csv(csv_path, index=False)
    return len(new_rows) - updated, updated


def upsert_daily_totals(summaries: list[dict], csv_path) -> tuple[int, int]:
    """Append parsed summaries to daily_totals.csv. Key: (date, period).
    Returns (rows_added, rows_updated)."""
    from pathlib import Path
    csv_path = Path(csv_path)
    cols = ['date', 'period', 'launched', 'intercepted', 'shaheds_estimated',
            'missiles_intercepted', 'hits', 'hit_locations',
            'posted_at', 'message_id', 'source']
    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        for c in cols:
            if c not in existing.columns:
                existing[c] = None
        existing = existing[cols]
    else:
        existing = pd.DataFrame(columns=cols)

    new_df = pd.DataFrame([s for s in summaries if s.get('date')])
    if new_df.empty:
        return 0, 0
    for c in cols:
        if c not in new_df.columns:
            new_df[c] = None
    new_df = new_df[cols]

    key_cols = ['date', 'period']
    existing_keys = set(zip(existing['date'].astype(str),
                            existing['period'].astype(str)))
    new_keys = set(zip(new_df['date'].astype(str),
                       new_df['period'].astype(str)))
    updated = len(existing_keys & new_keys)
    kept = existing[~existing.apply(lambda r: (str(r['date']), str(r['period'])) in new_keys, axis=1)]
    combined = pd.concat([kept, new_df], ignore_index=True)
    combined = combined.sort_values(['date', 'period'])
    combined.to_csv(csv_path, index=False)
    return len(new_df) - updated, updated


def scale_observations_to_totals(observations_df, daily_totals_df):
    """For each (date) with at least one summary and TG sighting rows in
    observations.csv, replace observed_drones with literal-count estimates:
        scaled[oblast] = round(sighting_share[oblast] × total_launched_for_date)

    Sightings without a matching summary are passed through unchanged.
    Returns a copy of observations_df with sources marked '...(scaled)' where
    scaling has been applied."""
    obs = observations_df.copy()
    if daily_totals_df is None or daily_totals_df.empty:
        return obs

    obs['observation_date'] = obs['observation_date'].astype(str)
    daily = daily_totals_df.copy()
    daily['date'] = daily['date'].astype(str)
    # Sum night + day summaries per date for full-day total
    daily_full = daily.groupby('date', as_index=False)['launched'].sum()

    is_tg = obs['source'].fillna('').str.startswith('UA Air Force TG')
    for d, total in zip(daily_full['date'], daily_full['launched']):
        if not total or pd.isna(total):
            continue
        mask = is_tg & (obs['observation_date'] == d)
        if not mask.any():
            continue
        sightings = obs.loc[mask, 'observed_drones'].sum()
        if sightings <= 0:
            continue
        obs.loc[mask, 'observed_drones'] = (
            obs.loc[mask, 'observed_drones'] * (float(total) / sightings)
        ).round().astype(int)
        obs.loc[mask, 'source'] = 'UA Air Force TG (scaled)'
    return obs


def sync(observations_csv, daily_totals_csv=None, pages: int = 12) -> dict:
    """End-to-end: fetch ~`pages` pages of history, parse, aggregate,
    upsert sightings and daily totals."""
    messages = fetch_history(max_pages=pages)
    rows, unmatched = messages_to_observation_rows(messages)
    added, updated = merge_into_observations(rows, observations_csv)

    # Per-message sightings log — one row per Telegram post with the ORIGINAL
    # timestamp preserved. Powers the live drone-track alerts panel.
    sightings_added = sightings_updated = 0
    try:
        from pathlib import Path as _Path
        _sightings_csv = _Path(observations_csv).parent / 'drone_sightings.csv'
        sightings_df = messages_to_sightings_log(messages)
        sightings_added, sightings_updated = upsert_sightings(
            sightings_df, _sightings_csv
        )
    except Exception:
        pass  # never fail the main sync if sightings-log has an issue

    summaries = []
    for m in messages:
        s = parse_summary(m)
        if s and s.get('date'):
            summaries.append(s)
    summary_added = summary_updated = 0
    if daily_totals_csv and summaries:
        summary_added, summary_updated = upsert_daily_totals(summaries, daily_totals_csv)

    # Update the launch-site forensics log (silent-launcher detection)
    ls_added = ls_updated = 0
    try:
        import launch_site_tracker as _lst
        from pathlib import Path as _Path
        _ls_csv = _Path(daily_totals_csv).parent / 'launch_site_log.csv' if daily_totals_csv else None
        if _ls_csv is not None:
            ls_df = _lst.parse_messages_for_sites(messages)
            if not ls_df.empty:
                ls_added, ls_updated = _lst.upsert_launch_site_log(ls_df, _ls_csv)
    except Exception:
        pass  # never fail the main sync if launch-site tracking has an issue

    # Update the weekly actuals ledger (current week's cumulative row)
    weekly_update = {}
    try:
        import weekly_tracker as _wt
        from pathlib import Path as _Path
        if daily_totals_csv:
            _db = _Path(daily_totals_csv).parent.parent / 'data' / 'forecast_history.db'
            if not _db.exists():
                _db = _Path(daily_totals_csv).parent / 'forecast_history.db'
            if _db.exists():
                _wt.init_tables(_db)
                daily_df = pd.read_csv(daily_totals_csv) if _Path(daily_totals_csv).exists() else pd.DataFrame()
                obs_df = pd.read_csv(observations_csv) if _Path(observations_csv).exists() else pd.DataFrame()
                if not daily_df.empty:
                    weekly_update = _wt.update_today_actual(daily_df, obs_df, _db)
                # Auto-close any past weeks that haven't been frozen
                closed = _wt.auto_close_pending_weeks(_db, _Path(daily_totals_csv).parent)
                if closed:
                    weekly_update['auto_closed_weeks'] = [c['week_start'] for c in closed]
    except Exception as _e:
        weekly_update = {'error': f'{type(_e).__name__}: {_e}'}

    return {
        'messages_seen': len(messages),
        'drone_messages': int(sum(1 for m in messages if is_drone_message(m['text']))),
        'rows_added': added,
        'rows_updated': updated,
        'summaries_parsed': len(summaries),
        'summary_rows_added': summary_added,
        'summary_rows_updated': summary_updated,
        'launch_site_rows_added': ls_added,
        'launch_site_rows_updated': ls_updated,
        'sightings_rows_added': sightings_added,
        'sightings_rows_updated': sightings_updated,
        'weekly_ledger_update': weekly_update,
        'unmatched_messages': unmatched[:8],  # cap UI clutter
        'date_range': (
            messages[0]['datetime'].date().isoformat() if messages else None,
            messages[-1]['datetime'].date().isoformat() if messages else None,
        ),
    }
