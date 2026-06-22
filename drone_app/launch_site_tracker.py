"""
Launch-site forensics: extract Russian drone launch-site mentions from
UA AF summary messages, persist them across syncs, and flag silent
launchers (sites that suddenly drop from active to zero).

The persistence file (`data/launch_site_log.csv`) accumulates one row per
(week_start, site, mentions) tuple so we keep history even after the
Telegram channel preview rotates old messages out.
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path
from datetime import timedelta

import pandas as pd


# Known Russian launch sites, with lat/lon (for the flight-time matrix)
LAUNCH_SITES = {
    'Bryansk':            (53.25, 34.37),  # NW Russia
    'Kursk':              (51.74, 36.18),  # central Russia
    'Shatalovo':          (54.13, 32.32),  # NW Russia (Smolensk Oblast)
    'Orel':               (52.97, 36.07),  # central Russia
    'Voronezh':           (51.66, 39.20),  # central-S Russia
    'Lipetsk':            (52.61, 39.59),  # central Russia (Su-34 base)
    'Millerovo':          (48.92, 40.40),  # SE Russia (Don)
    'Rostov':             (47.23, 39.71),  # S Russia
    'Primorsko-Akhtarsk': (46.04, 38.17),  # S Russia (Sea of Azov)
    'Gvardiyske (Crimea)':(45.10, 33.97),  # occupied Crimea
    'Chauda (Crimea)':    (45.05, 35.83),  # occupied Crimea
    'Donetsk (occupied)': (48.00, 37.80),  # occupied Donetsk
}

# Ukrainian/Russian text patterns for each site as it appears in UA AF
# summaries. Order doesn't matter — we count all matches per message.
SITE_PATTERNS = {
    'Bryansk':            r'(Брянськ|Брянщин|Брянс)',
    'Kursk':              r'(Курськ|Курщин|Курс)',
    'Shatalovo':          r'Шаталов',
    'Orel':               r'(Орел|Орл\b|Орлов)',
    'Voronezh':           r'(Воронеж|Воронез)',
    'Lipetsk':            r'Липецьк',
    'Millerovo':          r'(Міллєр|Міллер|Миллер)',
    'Rostov':             r'Ростов',
    'Primorsko-Akhtarsk': r'(Приморсько-Ахтарськ|Приморсько-Ахтар)',
    'Gvardiyske (Crimea)':r'Гвардійськ',
    'Chauda (Crimea)':    r'Чауд',
    'Donetsk (occupied)': r'ТОТ\s+Донецьк',
}

# Shahed-136 cruise speed (km/h) — public estimate
SHAHED_CRUISE_KMH = 185


@dataclass
class LaunchSiteSummary:
    """Summary view of recent launch-site activity."""
    weekly_mentions: pd.DataFrame  # rows = week_start, cols = site
    silent_alerts: list             # sites that dropped active→0
    new_activations: list           # sites that appeared this week
    surge_alerts: list              # sites that 3×+ in latest week vs prior


def extract_site_mentions(text: str) -> dict[str, int]:
    """Count how many times each launch site is mentioned in a message."""
    out = {}
    for site, pat in SITE_PATTERNS.items():
        matches = re.findall(pat, text, re.IGNORECASE | re.UNICODE)
        if matches:
            out[site] = len(matches)
    return out


def parse_messages_for_sites(messages: list[dict]) -> pd.DataFrame:
    """Given a list of {'datetime', 'text'} messages (from telegram_ingest),
    return a DataFrame with rows = week_start, columns = launch sites,
    values = mention counts. Only counts ЗБИТО summary messages."""
    rows = []
    for m in messages:
        if 'ЗБИТО' not in m['text'][:200]:
            continue
        # Always coerce to pandas Timestamp so .normalize/.tz_localize work
        dt = pd.Timestamp(m['datetime'])
        ws = (dt - pd.Timedelta(days=dt.weekday())).normalize()
        if ws.tzinfo is not None:
            ws = ws.tz_localize(None)
        sites = extract_site_mentions(m['text'])
        for site, count in sites.items():
            rows.append({'week_start': ws, 'site': site, 'mentions': count})
    if not rows:
        return pd.DataFrame(columns=['week_start', 'site', 'mentions'])
    df = pd.DataFrame(rows)
    df = df.groupby(['week_start', 'site'], as_index=False)['mentions'].sum()
    return df


def upsert_launch_site_log(new_df: pd.DataFrame, csv_path) -> tuple[int, int]:
    """Merge fresh mention counts into the persistent log. Keys: (week_start, site).
    New entries appended; existing entries take MAX(new, old) since later syncs
    might capture more mentions for the same week as the channel adds messages."""
    csv_path = Path(csv_path)
    if csv_path.exists():
        existing = pd.read_csv(csv_path, parse_dates=['week_start'])
    else:
        existing = pd.DataFrame(columns=['week_start', 'site', 'mentions'])

    if new_df.empty:
        return 0, 0

    new_df = new_df.copy()
    new_df['week_start'] = pd.to_datetime(new_df['week_start'])

    # Outer-merge to find existing vs new
    merged = existing.merge(new_df, on=['week_start', 'site'], how='outer',
                             suffixes=('_old', '_new'))
    merged['mentions'] = merged[['mentions_old', 'mentions_new']].max(axis=1)
    merged = merged[['week_start', 'site', 'mentions']]
    merged['mentions'] = merged['mentions'].astype(int)
    merged = merged.sort_values(['week_start', 'site'])

    added = (~new_df.set_index(['week_start','site']).index.isin(
        existing.set_index(['week_start','site']).index
    )).sum() if not existing.empty else len(new_df)
    updated = len(new_df) - added

    merged.to_csv(csv_path, index=False)
    return int(added), int(updated)


def detect_changes(log_df: pd.DataFrame,
                   silent_threshold: int = 2,
                   surge_multiplier: float = 3.0) -> LaunchSiteSummary:
    """Compare the two most-recent weeks for silent launchers and surges."""
    if log_df.empty:
        return LaunchSiteSummary(pd.DataFrame(), [], [], [])

    log_df['week_start'] = pd.to_datetime(log_df['week_start'])
    weeks = sorted(log_df['week_start'].unique())
    pivot = log_df.pivot_table(index='week_start', columns='site',
                                values='mentions', aggfunc='sum',
                                fill_value=0).sort_index()

    silent = []
    new_activations = []
    surges = []
    if len(weeks) >= 2:
        latest = pivot.iloc[-1]
        prior = pivot.iloc[-2]
        for site in pivot.columns:
            l = int(latest.get(site, 0))
            p = int(prior.get(site, 0))
            if p >= silent_threshold and l == 0:
                silent.append({'site': site, 'prior_mentions': p,
                                'latest_mentions': l,
                                'weeks_silent': 1})
            if p == 0 and l > 0:
                new_activations.append({'site': site, 'latest_mentions': l})
            if p > 0 and l >= p * surge_multiplier and l >= 3:
                surges.append({'site': site, 'prior_mentions': p,
                                'latest_mentions': l, 'multiplier': l/max(p,1)})

    return LaunchSiteSummary(
        weekly_mentions=pivot,
        silent_alerts=silent,
        new_activations=new_activations,
        surge_alerts=surges,
    )


def flight_time_matrix(oblast_df: pd.DataFrame) -> pd.DataFrame:
    """Build distance + flight-time table for each (oblast, launch_site) pair."""
    rows = []
    for _, ob in oblast_df.iterrows():
        olat, olon = float(ob['lat']), float(ob['lon'])
        for site, (slat, slon) in LAUNCH_SITES.items():
            dlat_km = (slat - olat) * 111
            dlon_km = (slon - olon) * 71  # at ~49°N
            dist_km = (dlat_km**2 + dlon_km**2) ** 0.5
            rows.append({
                'oblast': ob['oblast'],
                'launch_site': site,
                'distance_km': round(dist_km, 1),
                'flight_time_hr': round(dist_km / SHAHED_CRUISE_KMH, 2),
            })
    return pd.DataFrame(rows)


def closest_sites(oblast_df: pd.DataFrame) -> pd.DataFrame:
    """For each oblast, return the closest launch site + flight time."""
    ftm = flight_time_matrix(oblast_df)
    idx = ftm.groupby('oblast')['distance_km'].idxmin()
    closest = ftm.loc[idx].reset_index(drop=True)
    return closest.rename(columns={
        'launch_site': 'closest_site',
        'distance_km': 'distance_km',
        'flight_time_hr': 'flight_time_hr',
    })


def backsolve_launch_time(observation_utc, oblast: str,
                          oblasts_df: pd.DataFrame, site: str = None) -> dict:
    """Given a sighting time + oblast (and optional explicit launch site),
    estimate the probable launch time and origin site."""
    ob_row = oblasts_df[oblasts_df['oblast'] == oblast]
    if ob_row.empty:
        return {'error': f'oblast {oblast} not in features table'}
    olat, olon = float(ob_row['lat'].iloc[0]), float(ob_row['lon'].iloc[0])

    if site is None:
        # Pick closest site
        best_site, best_d = None, float('inf')
        for s, (slat, slon) in LAUNCH_SITES.items():
            d = (((slat - olat) * 111)**2 + ((slon - olon) * 71)**2) ** 0.5
            if d < best_d:
                best_site, best_d = s, d
        site = best_site

    slat, slon = LAUNCH_SITES[site]
    dist_km = (((slat - olat) * 111)**2 + ((slon - olon) * 71)**2) ** 0.5
    flight_hr = dist_km / SHAHED_CRUISE_KMH
    obs_dt = pd.Timestamp(observation_utc)
    launch_dt = obs_dt - pd.Timedelta(hours=flight_hr)
    return {
        'sighting_at': obs_dt.isoformat(),
        'oblast': oblast,
        'probable_launch_site': site,
        'distance_km': round(dist_km, 1),
        'flight_time_hr': round(flight_hr, 2),
        'probable_launch_time_utc': launch_dt.isoformat(),
    }
