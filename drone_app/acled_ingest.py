"""
ACLED ingestion for the Ukraine drone forecast app.

Pulls Air/drone strike events from ACLED, normalises ACLED's admin1 values
to the oblast names used in oblast_features.csv, and aggregates events per
(date, oblast). Each row that lands in observations.csv represents the
number of recorded strike *incidents* that day in that oblast, with
source='ACLED'. This is a proxy for launch activity — not a 1:1 drone
count — but it grows when activity rises and stays at zero during a
genuine pause.

Auth: ACLED uses OAuth2 password grant. Get an account at acleddata.com
(free), then pass email + password to fetch_events(...). The token is
short-lived; we fetch a new one on every sync, which is fine for daily
use.
"""
from __future__ import annotations
import re
import requests
import pandas as pd
from datetime import date, timedelta
from typing import Iterable

TOKEN_URL = "https://acleddata.com/oauth/token"
READ_URL = "https://acleddata.com/api/acled/read"

# Mapping from any ACLED admin1 value we've seen (or might see) -> the
# oblast name used in oblast_features.csv. Keys are *normalised* (lowercased,
# trimmed of " oblast"/" oblasti"/" oblasti"/" raion" suffixes). Anything
# unmatched is returned by ingest as 'unmatched_admin1' so we can extend
# this map without losing data silently.
OBLAST_CANONICAL = {
    'kharkiv': 'Kharkiv', 'kharkivska': 'Kharkiv',
    'kyiv': 'Kyiv City', 'kyiv city': 'Kyiv City',
    'kyivska': 'Kyiv Oblast', 'kyiv oblast': 'Kyiv Oblast',
    'odesa': 'Odesa', 'odeska': 'Odesa', 'odessa': 'Odesa',
    'lviv': 'Lviv', 'lvivska': 'Lviv',
    'dnipropetrovsk': 'Dnipropetrovsk', 'dnipropetrovska': 'Dnipropetrovsk',
    'donetsk': 'Donetsk', 'donetska': 'Donetsk',
    'zaporizhzhia': 'Zaporizhzhia', 'zaporizhzhya': 'Zaporizhzhia',
    'zaporizhia': 'Zaporizhzhia', 'zaporizka': 'Zaporizhzhia',
    'mykolaiv': 'Mykolaiv', 'mykolaivska': 'Mykolaiv',
    'kherson': 'Kherson', 'khersonska': 'Kherson',
    'poltava': 'Poltava', 'poltavska': 'Poltava',
    'sumy': 'Sumy', 'sumska': 'Sumy',
    'chernihiv': 'Chernihiv', 'chernihivska': 'Chernihiv',
    'vinnytsia': 'Vinnytsia', 'vinnytska': 'Vinnytsia',
    'cherkasy': 'Cherkasy', 'cherkaska': 'Cherkasy',
    'kirovohrad': 'Kirovohrad', 'kirovohradska': 'Kirovohrad',
    'zhytomyr': 'Zhytomyr', 'zhytomyrska': 'Zhytomyr',
    'rivne': 'Rivne', 'rivnenska': 'Rivne',
    'volyn': 'Volyn', 'volynska': 'Volyn',
    'ternopil': 'Ternopil', 'ternopilska': 'Ternopil',
    'khmelnytskyi': 'Khmelnytskyi', 'khmelnytska': 'Khmelnytskyi',
    'ivano-frankivsk': 'Ivano-Frankivsk', 'ivano-frankivska': 'Ivano-Frankivsk',
    'chernivtsi': 'Chernivtsi', 'chernivetska': 'Chernivtsi',
    'zakarpattia': 'Zakarpattia', 'zakarpatska': 'Zakarpattia',
    'luhansk': None, 'luhanska': None,  # not in our forecast set
    'crimea': None, 'autonomous republic of crimea': None,
    'sevastopol': None,
}

_SUFFIX_RE = re.compile(r'\s+(oblast|oblasti|raion|region)\s*$', re.IGNORECASE)


def normalize_admin1(name: str) -> str:
    if not isinstance(name, str):
        return ''
    n = name.strip().lower()
    n = _SUFFIX_RE.sub('', n)
    return n


def map_admin1_to_oblast(name: str) -> str | None:
    """Returns the canonical oblast name, or None if the admin1 should be
    dropped (Luhansk, Crimea, Sevastopol — not modelled), or '' if we don't
    recognise it at all."""
    norm = normalize_admin1(name)
    if norm in OBLAST_CANONICAL:
        return OBLAST_CANONICAL[norm]
    return ''  # unknown — caller logs & extends the map


class ACLEDError(RuntimeError):
    pass


def get_token(email: str, password: str, timeout: int = 20) -> str:
    resp = requests.post(
        TOKEN_URL,
        headers={'Content-Type': 'application/x-www-form-urlencoded'},
        data={
            'username': email,
            'password': password,
            'grant_type': 'password',
            'client_id': 'acled',
            'scope': 'authenticated',
        },
        timeout=timeout,
    )
    if resp.status_code != 200:
        raise ACLEDError(
            f"Auth failed ({resp.status_code}). Check email/password at "
            f"acleddata.com. Body: {resp.text[:200]}"
        )
    tok = resp.json().get('access_token')
    if not tok:
        raise ACLEDError(f"No access_token in response: {resp.text[:200]}")
    return tok


def fetch_events(
    token: str,
    start: date,
    end: date,
    country: str = 'Ukraine',
    sub_event_types: Iterable[str] = ('Air/drone strike',),
    timeout: int = 60,
    page_limit: int = 5000,
) -> pd.DataFrame:
    """Fetch ACLED events between start and end (inclusive). Returns a
    DataFrame with at least event_date, admin1, sub_event_type, notes."""
    fields = 'event_id_cnty|event_date|admin1|admin2|location|' \
             'event_type|sub_event_type|notes|fatalities'
    params = {
        '_format': 'json',
        'country': country,
        'event_date': f"{start.isoformat()}|{end.isoformat()}",
        'event_date_where': 'BETWEEN',
        'fields': fields,
        'limit': page_limit,
    }
    if sub_event_types:
        params['sub_event_type'] = ':OR:sub_event_type='.join(sub_event_types)

    resp = requests.get(
        READ_URL,
        params=params,
        headers={'Authorization': f"Bearer {token}",
                 'Content-Type': 'application/json'},
        timeout=timeout,
    )
    if resp.status_code != 200:
        raise ACLEDError(
            f"Read failed ({resp.status_code}). Body: {resp.text[:300]}"
        )
    body = resp.json()
    # ACLED's response shape: {'status': 200, 'success': True, 'data': [...]}
    if isinstance(body, dict) and 'data' in body:
        return pd.DataFrame(body['data'])
    if isinstance(body, list):
        return pd.DataFrame(body)
    raise ACLEDError(f"Unexpected response shape: {str(body)[:200]}")


def events_to_observation_rows(events: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Aggregate events into one row per (date, oblast). Returns (rows, unmatched_admin1s)."""
    if events.empty:
        return pd.DataFrame(columns=['observation_date', 'oblast',
                                     'observed_drones', 'source']), []

    df = events.copy()
    df['observation_date'] = pd.to_datetime(df['event_date']).dt.date.astype(str)
    df['mapped'] = df['admin1'].apply(map_admin1_to_oblast)

    unmatched = sorted(set(df.loc[df['mapped'] == '', 'admin1'].dropna().unique()))
    df = df[df['mapped'].notna() & (df['mapped'] != '')]

    agg = (
        df.groupby(['observation_date', 'mapped'])
          .size().reset_index(name='observed_drones')
    )
    agg = agg.rename(columns={'mapped': 'oblast'})
    agg['source'] = 'ACLED'
    return agg[['observation_date', 'oblast', 'observed_drones', 'source']], unmatched


def merge_into_observations(
    new_rows: pd.DataFrame,
    existing_csv_path,
) -> tuple[int, int]:
    """Append rows that aren't already present, keyed by (date, oblast, source).
    For (date, oblast, source='ACLED') we replace any prior row so re-fetching
    is idempotent. Returns (rows_added, rows_updated)."""
    existing = pd.read_csv(existing_csv_path)
    existing['observation_date'] = existing['observation_date'].astype(str)

    if new_rows.empty:
        return 0, 0

    # Replace any existing ACLED rows for the same (date, oblast)
    key_cols = ['observation_date', 'oblast', 'source']
    existing_key = existing[key_cols].astype(str).agg('|'.join, axis=1)
    new_key = new_rows[key_cols].astype(str).agg('|'.join, axis=1)
    updated = int(existing_key.isin(new_key.tolist()).sum())

    kept = existing[~existing_key.isin(new_key.tolist())]
    combined = pd.concat([kept, new_rows], ignore_index=True)
    combined = combined.sort_values(['observation_date', 'oblast'])
    combined.to_csv(existing_csv_path, index=False)

    added = len(new_rows) - updated
    return added, updated


def sync(
    email: str,
    password: str,
    start: date,
    end: date,
    observations_csv,
) -> dict:
    """End-to-end: auth, fetch, aggregate, merge. Returns a status dict."""
    token = get_token(email, password)
    events = fetch_events(token, start, end)
    rows, unmatched = events_to_observation_rows(events)
    added, updated = merge_into_observations(rows, observations_csv)
    return {
        'events_fetched': len(events),
        'rows_added': added,
        'rows_updated': updated,
        'unmatched_admin1': unmatched,
        'date_range': (start.isoformat(), end.isoformat()),
    }
