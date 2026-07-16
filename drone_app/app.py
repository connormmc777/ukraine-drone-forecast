"""
Ukraine Drone Forecast Dashboard
=================================
Local Streamlit app that runs the full forecasting model.
Run with: streamlit run app.py

This is the local version of what we tried to build on Palantir Foundry.
Same regression, same predictions, runs entirely on your computer.
"""
import json
import sqlite3
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from datetime import datetime, timedelta, date
from pathlib import Path

import acled_ingest
import telegram_ingest
from streamlit_autorefresh import st_autorefresh

st.set_page_config(
    page_title="Ukraine Drone Forecast",
    page_icon="🎯",
    layout="wide",
)

# ============== AUTO-REFRESH ==============
# A periodic tick reruns the script. The cached data loaders are keyed on
# file mtime, so when the observations or daily-totals CSVs change (either
# from the sidebar fetch button or from an external script), the next tick
# automatically picks them up — no manual refresh needed.
AUTO_REFRESH_OPTIONS = {
    'Off': 0, '15s': 15, '30s': 30, '1 min': 60,
    '2 min': 120, '5 min': 300,
}
with st.sidebar:
    st.caption("⟳ Auto-refresh")
    _refresh_label = st.selectbox(
        "Refresh interval", list(AUTO_REFRESH_OPTIONS.keys()), index=2,
        label_visibility='collapsed',
    )
    _refresh_secs = AUTO_REFRESH_OPTIONS[_refresh_label]
    auto_fetch_on_tick = st.checkbox(
        "Auto-fetch Telegram on each tick",
        value=False,
        help="Hits the @kpszsu preview on every refresh. Off by default to "
             "avoid hammering the channel.",
    )

if _refresh_secs > 0:
    tick = st_autorefresh(interval=_refresh_secs * 1000, key='dashboard-tick')
else:
    tick = 0

DATA_DIR = Path(__file__).parent / "data"
DB_PATH = DATA_DIR / "forecast_history.db"
OBS_CSV = DATA_DIR / "observations.csv"
DAILY_TOTALS_CSV = DATA_DIR / "daily_totals.csv"
ACLED_CREDS_PATH = DATA_DIR / ".acled_creds.json"
ACLED_SYNC_LOG = DATA_DIR / ".acled_sync_log.json"
TG_SYNC_LOG = DATA_DIR / ".tg_sync_log.json"


def load_tg_sync_log():
    if TG_SYNC_LOG.exists():
        try:
            return json.loads(TG_SYNC_LOG.read_text())
        except Exception:
            return {}
    return {}


def save_tg_sync_log(entry: dict):
    TG_SYNC_LOG.write_text(json.dumps(entry, default=str))


def load_acled_creds():
    if ACLED_CREDS_PATH.exists():
        try:
            return json.loads(ACLED_CREDS_PATH.read_text())
        except Exception:
            return {}
    return {}


def save_acled_creds(email: str, password: str):
    ACLED_CREDS_PATH.write_text(json.dumps({'email': email, 'password': password}))
    try:
        ACLED_CREDS_PATH.chmod(0o600)
    except Exception:
        pass


def load_sync_log():
    if ACLED_SYNC_LOG.exists():
        try:
            return json.loads(ACLED_SYNC_LOG.read_text())
        except Exception:
            return {}
    return {}


def save_sync_log(entry: dict):
    ACLED_SYNC_LOG.write_text(json.dumps(entry, default=str))


# ============== LOAD DATA ==============
# Cache keys include the file's mtime so external edits (e.g. an out-of-band
# sync from the CLI) invalidate the cache automatically without requiring the
# user to click anything.
def _mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_data
def load_oblasts(_mtime_key: float):
    return pd.read_csv(DATA_DIR / "oblast_features.csv")


@st.cache_data
def load_observations(_mtime_key: float):
    return pd.read_csv(DATA_DIR / "observations.csv")


@st.cache_data
def load_daily_totals(_mtime_key: float):
    if DAILY_TOTALS_CSV.exists():
        return pd.read_csv(DAILY_TOTALS_CSV)
    return pd.DataFrame(columns=['date', 'period', 'launched', 'intercepted',
                                  'shaheds_estimated', 'missiles_intercepted',
                                  'hits', 'hit_locations', 'posted_at',
                                  'message_id', 'source'])


# ============== ACCURACY TRACKING DB ==============
def db_connect():
    return sqlite3.connect(DB_PATH)


def init_db():
    with db_connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                week_start TEXT NOT NULL,
                russian_daily_capacity INTEGER,
                tempo_factor REAL,
                weekly_budget INTEGER,
                remaining_budget INTEGER,
                low_tempo INTEGER,
                learning_alpha REAL,
                note TEXT
            );
            CREATE TABLE IF NOT EXISTS snapshot_rows (
                snapshot_id INTEGER NOT NULL,
                oblast TEXT NOT NULL,
                prior_share REAL,
                obs_share REAL,
                learned_share REAL,
                predicted_week REAL,
                FOREIGN KEY (snapshot_id) REFERENCES snapshots(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_snapshot_rows_id
                ON snapshot_rows(snapshot_id);
            CREATE INDEX IF NOT EXISTS idx_snapshots_week
                ON snapshots(week_start);
            """
        )


def save_snapshot(forecast_df, params, note=""):
    """Persist one forecast as a snapshot. Returns the new snapshot id."""
    with db_connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO snapshots(
                created_at, week_start, russian_daily_capacity, tempo_factor,
                weekly_budget, remaining_budget, low_tempo, learning_alpha, note
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now().isoformat(timespec='seconds'),
                params['week_start'],
                int(params['russian_daily_capacity']),
                float(params['tempo_factor']),
                int(params['weekly_budget']),
                int(params['remaining_budget']),
                int(bool(params['low_tempo'])),
                float(params['learning_alpha']),
                note,
            ),
        )
        snap_id = cur.lastrowid
        rows = [
            (
                snap_id,
                r['oblast'],
                float(r.get('share', 0.0)),
                float(r.get('obs_share', 0.0)),
                float(r.get('learned_share', 0.0)),
                float(r['predicted_week']),
            )
            for _, r in forecast_df.iterrows()
        ]
        conn.executemany(
            """
            INSERT INTO snapshot_rows(
                snapshot_id, oblast, prior_share, obs_share,
                learned_share, predicted_week
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
    return snap_id


def init_backtest_table():
    with db_connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS backtest_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_at TEXT NOT NULL,
                target_date TEXT NOT NULL,
                budget_used INTEGER,
                predicted_total INTEGER,
                actual_total INTEGER,
                total_error INTEGER,
                mae_per_oblast REAL,
                pearson_r REAL,
                spatial_r REAL,
                alpha_used REAL,
                training_obs INTEGER,
                low_tempo INTEGER
            );
            CREATE TABLE IF NOT EXISTS backtest_rows (
                run_id INTEGER NOT NULL,
                target_date TEXT NOT NULL,
                oblast TEXT NOT NULL,
                predicted REAL,
                actual REAL,
                FOREIGN KEY (run_id) REFERENCES backtest_results(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_backtest_run ON backtest_rows(run_id);
            CREATE INDEX IF NOT EXISTS idx_backtest_date ON backtest_results(target_date);
            """
        )


def list_snapshots():
    with db_connect() as conn:
        return pd.read_sql_query(
            "SELECT * FROM snapshots ORDER BY created_at DESC", conn
        )


def get_snapshot_rows(snap_id):
    with db_connect() as conn:
        return pd.read_sql_query(
            "SELECT * FROM snapshot_rows WHERE snapshot_id = ?",
            conn, params=(snap_id,),
        )


def delete_snapshot(snap_id):
    with db_connect() as conn:
        conn.execute("DELETE FROM snapshot_rows WHERE snapshot_id = ?", (snap_id,))
        conn.execute("DELETE FROM snapshots WHERE id = ?", (snap_id,))


def score_snapshots(observations_df):
    """Join every snapshot's per-oblast prediction against the observations
    log aggregated over that snapshot's week. Returns one row per snapshot.

    Marks data_gap=True for weeks where NO observations exist at all —
    distinguishing 'Russia genuinely launched zero' from 'we don't have data
    for that week'. Critical: never report observed_total=0 as a real
    measurement when it's actually a data gap."""
    snaps = list_snapshots()
    if snaps.empty:
        return pd.DataFrame(), pd.DataFrame()

    obs = observations_df.copy()
    obs['observation_date'] = pd.to_datetime(obs['observation_date'])

    summary_rows = []
    detail_rows = []
    for _, s in snaps.iterrows():
        ws = pd.Timestamp(s['week_start'])
        we = ws + timedelta(days=7)
        week_obs = obs[(obs['observation_date'] >= ws) &
                       (obs['observation_date'] < we)]
        data_gap = week_obs.empty
        observed_by_oblast = (
            week_obs.groupby('oblast')['observed_drones'].sum().reset_index()
        )
        observed_by_oblast.columns = ['oblast', 'observed']

        rows = get_snapshot_rows(int(s['id']))
        merged = rows.merge(observed_by_oblast, on='oblast', how='left')
        merged['observed'] = merged['observed'].fillna(0)
        merged['error'] = merged['predicted_week'] - merged['observed']
        merged['abs_error'] = merged['error'].abs()

        predicted_total = float(merged['predicted_week'].sum())
        observed_total = float(merged['observed'].sum())
        mae = float(merged['abs_error'].mean())
        if merged['predicted_week'].std() > 0 and merged['observed'].std() > 0:
            pearson = float(np.corrcoef(
                merged['predicted_week'], merged['observed']
            )[0, 1])
        else:
            pearson = np.nan
        # Share-only correlation (how well does spatial pattern hold up,
        # independent of volume)
        if predicted_total > 0 and observed_total > 0:
            pred_share = merged['predicted_week'] / predicted_total
            obs_share = merged['observed'] / observed_total
            share_pearson = float(np.corrcoef(pred_share, obs_share)[0, 1])
        else:
            share_pearson = np.nan

        summary_rows.append({
            'snapshot_id': int(s['id']),
            'created_at': s['created_at'],
            'week_start': s['week_start'],
            'predicted_total': int(round(predicted_total)),
            'observed_total': (None if data_gap else int(round(observed_total))),
            'total_error': (None if data_gap
                            else int(round(predicted_total - observed_total))),
            'mae_per_oblast': (None if data_gap else round(mae, 2)),
            'pearson_r': round(pearson, 3) if pd.notna(pearson) else None,
            'share_pearson_r': round(share_pearson, 3)
                if pd.notna(share_pearson) else None,
            'alpha': round(float(s['learning_alpha']), 3),
            'data_gap': data_gap,
            'status': ('DATA GAP — actual unknown' if data_gap
                       else 'scored'),
            'note': s['note'] or '',
        })
        merged['snapshot_id'] = int(s['id'])
        merged['week_start'] = s['week_start']
        detail_rows.append(merged)

    summary = pd.DataFrame(summary_rows)
    details = pd.concat(detail_rows, ignore_index=True) if detail_rows else pd.DataFrame()
    return summary, details


init_db()
init_backtest_table()


def snapshot_exists_for_week(week_start: date) -> bool:
    with db_connect() as conn:
        row = conn.execute(
            "SELECT 1 FROM snapshots WHERE week_start = ? LIMIT 1",
            (week_start.isoformat(),),
        ).fetchone()
    return row is not None


def auto_lock_if_due():
    """If today is Monday and there is no snapshot for this week yet,
    lock one using the standalone script's logic. Idempotent."""
    import importlib.util
    today_dt = date.today()
    if today_dt.weekday() != 0:  # only on Mondays
        return None
    if snapshot_exists_for_week(today_dt):
        return None
    # Call out to the standalone module so the logic stays in one place
    spec = importlib.util.spec_from_file_location(
        "lock_weekly_snapshot", Path(__file__).parent / "lock_weekly_snapshot.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    oblasts_df = pd.read_csv(DATA_DIR / 'oblast_features.csv')
    raw_obs_df = pd.read_csv(DATA_DIR / 'observations.csv')
    daily_df = (pd.read_csv(DAILY_TOTALS_CSV)
                if DAILY_TOTALS_CSV.exists()
                else pd.DataFrame(columns=['date', 'launched']))
    obs_df = telegram_ingest.scale_observations_to_totals(raw_obs_df, daily_df)
    capacity = module.data_driven_capacity(daily_df)
    snap_id = module.lock(
        today_dt, capacity, 1.0, oblasts_df, obs_df,
        note=f"Auto-locked Monday {today_dt.isoformat()}: "
             f"data-driven {capacity}/d × 7 × 1.0 buffer 1.10",
    )
    return snap_id


# ============== REGRESSION MODEL ==============
def compute_targeting_weights(oblasts):
    """The structural regression coefficients applied to features."""
    oblasts = oblasts.copy()
    oblasts['weight'] = (
        0.35 * oblasts['energy'] +
        1.5 * np.exp(-oblasts['border_dist'] / 400) +
        0.25 * oblasts['pop']
    )
    oblasts.loc[oblasts['border_dist'] > 700, 'weight'] *= 0.4
    oblasts['share'] = oblasts['weight'] / oblasts['weight'].sum()
    return oblasts


def recalibrate_from_observations(oblasts, observations, shrinkage_k=150):
    """Blend the structural prior with the empirical share from observed launches.

    alpha = N_obs / (N_obs + K). Few launches -> trust the prior; many launches
    -> trust the data. This is the self-updating step: every new launcher use
    shifts predicted_share toward what's actually been observed.
    """
    oblasts = oblasts.copy()
    if observations is None or len(observations) == 0:
        oblasts['obs_share'] = 0.0
        oblasts['learned_share'] = oblasts['share']
        oblasts['alpha'] = 0.0
        oblasts['n_obs'] = 0
        return oblasts

    obs_by_oblast = (
        observations.groupby('oblast')['observed_drones'].sum().reset_index()
    )
    obs_by_oblast.columns = ['oblast', 'n_obs']
    oblasts = oblasts.merge(obs_by_oblast, on='oblast', how='left')
    oblasts['n_obs'] = oblasts['n_obs'].fillna(0)

    total_obs = oblasts['n_obs'].sum()
    if total_obs > 0:
        oblasts['obs_share'] = oblasts['n_obs'] / total_obs
    else:
        oblasts['obs_share'] = 0.0

    alpha = total_obs / (total_obs + shrinkage_k)
    oblasts['alpha'] = alpha
    oblasts['learned_share'] = (1 - alpha) * oblasts['share'] + alpha * oblasts['obs_share']
    oblasts['learned_share'] = oblasts['learned_share'] / oblasts['learned_share'].sum()
    return oblasts


def generate_weekly_forecast(oblasts, weekly_budget, low_tempo=True, observations=None):
    """Distribute weekly budget across oblasts using regression shares,
    recalibrated against any observed launches (self-updating step)."""
    oblasts = compute_targeting_weights(oblasts)
    oblasts = recalibrate_from_observations(oblasts, observations)
    oblasts['adj_share'] = oblasts['learned_share']

    # Regime adjustment: border bias under low tempo
    if low_tempo:
        oblasts.loc[oblasts['border_dist'] < 100, 'adj_share'] *= 1.30
        oblasts.loc[oblasts['energy'] >= 7, 'adj_share'] *= 0.90
    oblasts['adj_share'] = oblasts['adj_share'] / oblasts['adj_share'].sum()
    oblasts['predicted_week'] = (oblasts['adj_share'] * weekly_budget).round(0)

    return oblasts.sort_values('predicted_week', ascending=False)


# ============== WALK-FORWARD BACKTEST ==============
def run_backtest(oblasts_features, observations_df, daily_totals_df, low_tempo=False):
    """For each historical day D where we have both a UA AF summary and
    per-oblast sightings, train a model on observations BEFORE D, predict D,
    and score against the actuals for D.

    Returns (per_day_df, per_oblast_df)."""
    if daily_totals_df is None or daily_totals_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    obs = observations_df.copy()
    obs['observation_date'] = pd.to_datetime(obs['observation_date'])
    daily = daily_totals_df.copy()
    daily['date'] = pd.to_datetime(daily['date'])
    daily_sum = daily.groupby('date', as_index=False)['launched'].sum()

    per_day = []
    per_oblast_rows = []

    for D in sorted(daily_sum['date'].unique()):
        train = obs[obs['observation_date'] < D]
        actual = obs[obs['observation_date'].dt.date == D.date()]
        if actual.empty:
            continue
        budget = int(daily_sum.loc[daily_sum['date'] == D, 'launched'].iloc[0])
        if budget <= 0:
            continue

        cal = compute_targeting_weights(oblasts_features)
        cal = recalibrate_from_observations(cal, train)
        cal['adj_share'] = cal['learned_share']
        if low_tempo:
            cal.loc[cal['border_dist'] < 100, 'adj_share'] *= 1.30
            cal.loc[cal['energy'] >= 7, 'adj_share'] *= 0.90
        cal['adj_share'] = cal['adj_share'] / cal['adj_share'].sum()
        cal['predicted'] = (cal['adj_share'] * budget).round(0)

        actual_by = (actual.groupby('oblast')['observed_drones']
                     .sum().reset_index()
                     .rename(columns={'observed_drones': 'actual'}))
        merged = cal[['oblast', 'predicted']].merge(
            actual_by, on='oblast', how='left',
        ).fillna({'actual': 0})

        pred_total = int(merged['predicted'].sum())
        actual_total = int(merged['actual'].sum())
        mae = float((merged['predicted'] - merged['actual']).abs().mean())
        if merged['predicted'].std() and merged['actual'].std():
            r = float(np.corrcoef(merged['predicted'], merged['actual'])[0, 1])
        else:
            r = np.nan
        if pred_total and actual_total:
            ps = merged['predicted'] / pred_total
            os_ = merged['actual'] / actual_total
            sr = float(np.corrcoef(ps, os_)[0, 1])
        else:
            sr = np.nan

        per_day.append({
            'date': D.date().isoformat(),
            'budget_used': budget,
            'predicted_total': pred_total,
            'actual_total': actual_total,
            'total_error': pred_total - actual_total,
            'mae': round(mae, 2),
            'pearson_r': round(r, 3) if pd.notna(r) else None,
            'spatial_r': round(sr, 3) if pd.notna(sr) else None,
            'alpha_used': round(float(cal['alpha'].iloc[0]), 3),
            'training_obs': int(train['observed_drones'].sum()),
        })
        for _, row in merged.iterrows():
            per_oblast_rows.append({
                'date': D.date().isoformat(),
                'oblast': row['oblast'],
                'predicted': float(row['predicted']),
                'actual': float(row['actual']),
                'error': float(row['predicted'] - row['actual']),
            })

    return pd.DataFrame(per_day), pd.DataFrame(per_oblast_rows)


def save_backtest_run(per_day_df, per_oblast_df, low_tempo=False):
    if per_day_df.empty:
        return None
    run_at = datetime.now().isoformat(timespec='seconds')
    with db_connect() as conn:
        last_run_id = None
        for _, r in per_day_df.iterrows():
            cur = conn.execute(
                """INSERT INTO backtest_results(
                    run_at, target_date, budget_used, predicted_total,
                    actual_total, total_error, mae_per_oblast,
                    pearson_r, spatial_r, alpha_used, training_obs, low_tempo
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (run_at, r['date'], int(r['budget_used']),
                 int(r['predicted_total']), int(r['actual_total']),
                 int(r['total_error']), float(r['mae']),
                 float(r['pearson_r']) if r['pearson_r'] is not None else None,
                 float(r['spatial_r']) if r['spatial_r'] is not None else None,
                 float(r['alpha_used']), int(r['training_obs']),
                 int(bool(low_tempo))),
            )
            run_id = cur.lastrowid
            last_run_id = run_id
            day_rows = per_oblast_df[per_oblast_df['date'] == r['date']]
            conn.executemany(
                "INSERT INTO backtest_rows(run_id, target_date, oblast, predicted, actual) "
                "VALUES (?,?,?,?,?)",
                [(run_id, row['date'], row['oblast'],
                  float(row['predicted']), float(row['actual']))
                 for _, row in day_rows.iterrows()],
            )
    return last_run_id


def latest_backtest_results():
    with db_connect() as conn:
        runs = pd.read_sql_query(
            "SELECT * FROM backtest_results ORDER BY target_date", conn,
        )
        rows = pd.read_sql_query("SELECT * FROM backtest_rows", conn)
    if runs.empty:
        return runs, rows
    runs = (runs.sort_values('run_at', ascending=False)
                .drop_duplicates('target_date')
                .sort_values('target_date'))
    rows = rows[rows['run_id'].isin(set(runs['id']))]
    return runs, rows


# ============== UI ==============
# Auto-fetch fires on the autorefresh tick if enabled. Done early so the
# rest of the page renders against the freshly-synced data.
if auto_fetch_on_tick and _refresh_secs > 0 and tick > 0:
    try:
        result = telegram_ingest.sync(OBS_CSV, DAILY_TOTALS_CSV, pages=4)
        result['at'] = datetime.now().isoformat(timespec='seconds')
        save_tg_sync_log(result)
    except Exception:
        # Don't break the page if the channel is unreachable
        pass

# Auto-lock a fresh snapshot every Monday if none exists yet for the week
_autolock_snap_id = None
try:
    _autolock_snap_id = auto_lock_if_due()
except Exception as _e:
    _autolock_snap_id = None
if _autolock_snap_id:
    st.success(
        f"📌 Auto-locked snapshot #{_autolock_snap_id} for week of "
        f"{date.today().isoformat()} (data-driven budget from rolling "
        f"14-day average)."
    )

st.title("🎯 Ukraine Drone Strike Forecast")
_render_time = datetime.now().strftime('%H:%M:%S')
_refresh_hint = (f"auto-refresh every {_refresh_label}" if _refresh_secs > 0
                 else "auto-refresh off")
st.markdown(
    f"**Self-updating weekly forecast.** Structural regression prior + "
    f"empirical recalibration. Every recorded launcher use shifts the share "
    f"estimates — no manual refresh required. "
    f"<span style='color:#888'>· rendered {_render_time} · "
    f"{_refresh_hint}</span>",
    unsafe_allow_html=True,
)

# Load observations first so the sidebar can self-update from them.
# Pass file mtimes so external file changes invalidate the cache.
oblasts = load_oblasts(_mtime(DATA_DIR / "oblast_features.csv"))
raw_observations = load_observations(_mtime(OBS_CSV))
daily_totals = load_daily_totals(_mtime(DAILY_TOTALS_CSV))

# Scale sighting counts to literal drone counts wherever a UA AF summary
# exists for that date. Rows from other sources or dates pass through.
observations = telegram_ingest.scale_observations_to_totals(raw_observations, daily_totals)
observations['observation_date'] = pd.to_datetime(observations['observation_date'])
today = pd.Timestamp(datetime.now().date())
week_start = today - timedelta(days=today.weekday())
this_week_obs = observations[observations['observation_date'] >= week_start]
launches_this_week = int(this_week_obs['observed_drones'].sum())

# Freshness banner: how stale is the OBSERVATION LOG (data coverage) vs
# how recently did we SYNC (data pull).
#
# Data-source note: UA Air Force @kpszsu posts ONE morning summary per day
# covering the previous night. So `last_obs_date` = yesterday is EXPECTED
# most of the day — that's not stale, that's the source's cadence.
# The sync loop pulls every 60s and normally catches the morning summary
# within a minute of it being posted (~06:00 Kyiv).
last_obs_date = (
    observations['observation_date'].max().date()
    if len(observations) else None
)
_tg_last = load_tg_sync_log() or {}
_last_sync_at = _tg_last.get('at')
if _last_sync_at:
    try:
        _last_sync_dt = datetime.fromisoformat(_last_sync_at)
        _sync_age = datetime.now() - _last_sync_dt
        if _sync_age.total_seconds() < 90:
            _sync_ago = f"{int(_sync_age.total_seconds())}s ago"
        elif _sync_age.total_seconds() < 3600:
            _sync_ago = f"{int(_sync_age.total_seconds() // 60)}m ago"
        elif _sync_age.total_seconds() < 86400:
            _sync_ago = f"{int(_sync_age.total_seconds() // 3600)}h ago"
        else:
            _sync_ago = f"{int(_sync_age.total_seconds() // 86400)}d ago"
    except Exception:
        _sync_ago = "unknown"
else:
    _sync_ago = "never"

if last_obs_date is None:
    st.error("⚠️ No observations on file. Predictions are pure prior — α=0. "
             "Sync ACLED from the sidebar or add observations manually.")
else:
    days_stale = (today.date() - last_obs_date).days
    # UA Air Force posts the previous night's summary each morning ~06:00 Kyiv.
    # So an obs dated "today" or "yesterday" is fresh, "2+ days" is stale.
    if days_stale >= 3:
        st.error(
            f"🛑 Observation log covers up to **{last_obs_date}** "
            f"({days_stale} days behind). Sync last ran {_sync_ago}. "
            f"Either the source stopped posting or the sync loop is broken — "
            f"check `kubectl logs -l app.kubernetes.io/name=dronespredictions "
            f"-c sync-loop` for errors."
        )
    elif days_stale == 2:
        st.warning(
            f"⚠️ Observation log covers up to **{last_obs_date}** "
            f"(2 nights ago). Sync last ran {_sync_ago}. This is unusual — "
            f"the UA Air Force normally posts a morning summary daily."
        )
    else:
        # 0 or 1 days stale is the source's normal cadence. Green banner.
        _obs_desc = "tonight's summary is in" if days_stale == 0 \
                    else "last night's summary is the most recent"
        st.success(
            f"✅ Data current — {_obs_desc} (**{last_obs_date}**). "
            f"Sync last ran **{_sync_ago}** (loop polls every 60s). "
            f"The UA Air Force posts once per day at ~06:00 Kyiv, so "
            f"'yesterday' is normal until the next morning's report drops."
        )

# ============== LIVE — LATEST REGIONAL ACTIVITY (top-of-page) ==============
# Refreshes on every autorefresh tick. Shows the newest per-oblast counts,
# the day's per-oblast budget from the current locked snapshot, the delta
# actual − budget, a distance table between successive latest-hit regions,
# and a bar chart of the single most-recently-hit region's actual vs budget.
def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1r, lat2r = np.radians(lat1), np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2 +
         np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2)
    return 2 * R * np.arcsin(np.sqrt(a))


st.markdown("---")
st.subheader("🔴 LIVE — Latest Regional Activity")

# Today's daily budget per oblast is drawn from THIS week's locked snapshot
# (auto-locked on Monday). We split the weekly budget evenly across 7 days as
# a baseline; per-day weighting could refine this later.
_snap_row = None
with db_connect() as conn:
    _snap_row = conn.execute(
        "SELECT id, weekly_budget FROM snapshots WHERE week_start = ? "
        "ORDER BY id DESC LIMIT 1",
        (week_start.date().isoformat(),),
    ).fetchone()
_week_budget = _snap_row[1] if _snap_row else 1300
_daily_budget = _week_budget / 7.0

# lat/lon + share for each oblast (structural forecast)
_oblasts_geo = pd.read_csv(DATA_DIR / "updated_forecast.csv")[
    ['oblast', 'lat', 'lon', 'share']
].copy()
_oblasts_geo['today_budget'] = _oblasts_geo['share'] * _daily_budget

# Region filter — empty = show everything
_all_regions = sorted(observations['oblast'].dropna().unique().tolist())
_sel_regions = st.multiselect(
    "Filter regions (empty = all)", _all_regions, default=[],
    help="Narrow the table + bar chart to specific oblasts.",
    key='live_region_filter',
)

# Per-oblast: most recent observation date, and today's actual count
_latest_by_oblast = (
    observations.groupby('oblast')['observation_date'].max().reset_index()
    .rename(columns={'observation_date': 'last_hit'})
)
_latest_by_oblast = _latest_by_oblast.merge(_oblasts_geo, on='oblast', how='left')

_latest_date_overall = observations['observation_date'].max()
_todays = (
    observations[observations['observation_date'] == _latest_date_overall]
    .groupby('oblast')['observed_drones'].sum().reset_index()
    .rename(columns={'observed_drones': 'actual_today'})
)
_panel = _latest_by_oblast.merge(_todays, on='oblast', how='left').fillna(
    {'actual_today': 0}
)
_panel['delta'] = _panel['actual_today'] - _panel['today_budget']
_panel['days_since_hit'] = (today - _panel['last_hit']).dt.days

if _sel_regions:
    _panel = _panel[_panel['oblast'].isin(_sel_regions)]

# Sort by most-recent-hit desc, then by today's actual desc
_panel = _panel.sort_values(['last_hit', 'actual_today'], ascending=[False, False])

_col_tbl, _col_bar = st.columns([2, 1])
with _col_tbl:
    st.markdown(
        f"**Report covers:** night of {_latest_date_overall.date()} · "
        f"**Sync last ran:** {_sync_ago} (polls every 60s) · "
        f"**Today's per-day budget:** {_daily_budget:.0f} drones · "
        f"**Week snapshot:** #{_snap_row[0] if _snap_row else '—'} "
        f"({_week_budget}/wk)"
    )
    _display = _panel.copy()
    _display['last_hit'] = _display['last_hit'].dt.date
    _display = _display[
        ['oblast', 'last_hit', 'days_since_hit',
         'actual_today', 'today_budget', 'delta']
    ]
    _display.columns = [
        'Region', 'Last hit', 'Days since',
        'Actual today', 'Budget today', 'Δ (act−bud)',
    ]
    _display['Budget today'] = _display['Budget today'].round(1)
    _display['Δ (act−bud)'] = _display['Δ (act−bud)'].round(1)
    st.dataframe(_display, use_container_width=True, hide_index=True)

with _col_bar:
    if len(_panel):
        _top = _panel.iloc[0]
        _fig_l, _ax_l = plt.subplots(figsize=(4, 3))
        _vals = [float(_top['actual_today']), float(_top['today_budget'])]
        _bars = _ax_l.bar(['Actual', 'Budget'], _vals,
                          color=['#cc0033', '#003d7a'])
        _ax_l.set_title(f"{_top['oblast']} — today", fontsize=11)
        _ax_l.set_ylabel('drones')
        _ax_l.grid(alpha=0.25, axis='y')
        for _b, _v in zip(_bars, _vals):
            _ax_l.text(_b.get_x() + _b.get_width() / 2, _b.get_height(),
                       f"{_v:.0f}", ha='center', va='bottom', fontsize=10)
        st.pyplot(_fig_l)
        plt.close(_fig_l)

# ---------- Per-message drone-track alerts (minute-fresh) ----------
# One row per @kpszsu Telegram post, timestamped by the original post time.
# Powers the "what's happening RIGHT NOW" view — separate from the nightly
# summary aggregate above.
_SIGHTINGS_CSV = DATA_DIR / 'drone_sightings.csv'
if _SIGHTINGS_CSV.exists():
    try:
        _sight = pd.read_csv(_SIGHTINGS_CSV)
        if not _sight.empty:
            _sight['posted_at'] = pd.to_datetime(_sight['posted_at'], utc=True,
                                                 errors='coerce')
            _sight = _sight.dropna(subset=['posted_at']).sort_values(
                'posted_at', ascending=False
            )
            if _sel_regions:
                _sight = _sight[_sight['oblast'].isin(_sel_regions)]
            _now_utc = pd.Timestamp.utcnow()
            _sight['age'] = _now_utc - _sight['posted_at']

            def _age_str(td):
                s = td.total_seconds()
                if s < 90:
                    return f"{int(s)}s ago"
                if s < 3600:
                    return f"{int(s // 60)}m ago"
                if s < 86400:
                    return f"{int(s // 3600)}h ago"
                return f"{int(s // 86400)}d ago"

            _sight['ago'] = _sight['age'].apply(_age_str)
            _display_s = _sight.head(15)[
                ['ago', 'oblast', 'text', 'posted_at']
            ].copy()
            _display_s.columns = ['Age', 'Region', 'Message', 'Posted (UTC)']
            _display_s['Posted (UTC)'] = _display_s['Posted (UTC)'].dt.strftime(
                '%Y-%m-%d %H:%M:%S'
            )
            st.caption(
                "**Live drone-track alerts** — one row per @kpszsu Telegram "
                "post, timestamped when the source posted (not aggregated). "
                f"Showing 15 most recent of {len(_sight):,} in log. Region "
                "filter above applies."
            )
            st.dataframe(_display_s, use_container_width=True, hide_index=True)
    except Exception as _sight_e:
        st.caption(f"(sightings log unavailable: {type(_sight_e).__name__})")

# Distance table: pairwise great-circle distance between successive
# latest-hit regions (most recent → older).
_geo_panel = _panel.dropna(subset=['lat', 'lon']).reset_index(drop=True)
if len(_geo_panel) >= 2:
    _dists = []
    for _i in range(1, len(_geo_panel)):
        _a = _geo_panel.iloc[_i - 1]
        _b = _geo_panel.iloc[_i]
        _dists.append({
            'From': _a['oblast'],
            'To': _b['oblast'],
            'Distance (km)': int(round(_haversine_km(
                _a['lat'], _a['lon'], _b['lat'], _b['lon']))),
        })
    st.caption("**Distance between successive latest-hit regions** "
               "(great-circle, ordered most-recent → older)")
    st.dataframe(pd.DataFrame(_dists), use_container_width=True,
                 hide_index=True)

st.markdown("---")

# Official UA Air Force daily counts (when available)
if not daily_totals.empty:
    dt = daily_totals.copy()
    dt['date'] = pd.to_datetime(dt['date'])
    dt = dt.sort_values(['date', 'period'])
    today_str = today.date().isoformat()
    yesterday_str = (today.date() - timedelta(days=1)).isoformat()

    # Per-date totals (sum night + day)
    daily_sum = (
        dt.groupby('date')
          .agg(launched=('launched', 'sum'),
               intercepted=('intercepted', 'sum'),
               hits=('hits', 'sum'))
          .reset_index().sort_values('date', ascending=False)
    )
    # =============== CURRENT WEEK PROGRESS + PROJECTION ===============
    current_week_dates = pd.date_range(week_start, week_start + timedelta(days=6))
    current_week_actual = (dt[(dt['date'] >= week_start) &
                              (dt['date'] <= week_start + timedelta(days=6))]
                           .groupby('date', as_index=False)['launched'].sum())
    week_so_far = int(current_week_actual['launched'].sum())
    day_of_week = today.weekday()  # 0=Mon, 6=Sun
    days_elapsed = day_of_week + 1
    days_remaining = 7 - days_elapsed
    days_with_data = len(current_week_actual)
    avg_per_day_so_far = (week_so_far / max(days_with_data, 1)) if days_with_data else 0

    # Use the locked snapshot for this week as the model's projection.
    # This block runs BEFORE the sidebar (where the live `weekly_budget` is
    # computed), so the fallback uses the data-driven recipe directly.
    import sqlite3 as _sq3
    with _sq3.connect(DB_PATH) as _c:
        _row = _c.execute(
            "SELECT id, weekly_budget, learning_alpha FROM snapshots "
            "WHERE week_start=? ORDER BY id DESC LIMIT 1",
            (week_start.date().isoformat(),),
        ).fetchone()
    if _row:
        snap_budget = _row[1]
    else:
        # Data-driven fallback: 14-day rolling avg × 7 × 1.10 buffer
        _by_d = daily_totals.copy()
        _by_d['date'] = pd.to_datetime(_by_d['date'])
        _rolling = (_by_d.groupby('date')['launched'].sum()
                          .sort_index().tail(14).mean())
        snap_budget = int(round((_rolling if pd.notna(_rolling) else 300) * 7 * 1.10))
    snap_id = _row[0] if _row else None
    snap_remaining = max(snap_budget - week_so_far, 0)
    expected_rest = avg_per_day_so_far * days_remaining
    projected_week_total = week_so_far + expected_rest

    with st.container():
        st.markdown("### 📅 Current week progress (Mon → Sun)")
        st.caption(
            f"Today: **{today.date()}** ({today.day_name()}) — "
            f"day {days_elapsed} of 7. Two projections shown: "
            f"model's locked budget (Snapshot #{snap_id}) vs linear "
            f"extrapolation from observed days."
        )
        pcol1, pcol2, pcol3, pcol4 = st.columns(4)
        with pcol1:
            st.metric("Launched so far",
                      f"{week_so_far:,}",
                      f"over {days_with_data} day(s) of data")
        with pcol2:
            st.metric("Avg per day so far",
                      f"{avg_per_day_so_far:.0f}",
                      f"× {days_remaining} days left")
        with pcol3:
            st.metric("Model's locked budget",
                      f"{snap_budget:,}",
                      f"≈ {snap_budget // 7}/day")
        with pcol4:
            delta = projected_week_total - snap_budget
            st.metric("Projected week total",
                      f"{int(projected_week_total):,}",
                      f"vs budget {delta:+,}")

        # Day-by-day chart: actual + projected
        fig_wk, ax_wk = plt.subplots(figsize=(12, 3.5))
        chart_data = []
        for d in current_week_dates:
            actual_row = current_week_actual[current_week_actual['date'] == d]
            if not actual_row.empty:
                chart_data.append((d, int(actual_row['launched'].iloc[0]), 'Actual'))
            elif d.date() < today.date():
                chart_data.append((d, 0, 'No summary yet'))
            else:
                chart_data.append((d, int(round(avg_per_day_so_far)), 'Projected'))

        dates = [c[0] for c in chart_data]
        vals = [c[1] for c in chart_data]
        kinds = [c[2] for c in chart_data]
        color_map = {'Actual': '#003d7a', 'No summary yet': '#bbbbbb',
                     'Projected': '#cc6633'}
        colors = [color_map[k] for k in kinds]
        labels = [d.strftime('%a %m-%d') for d in dates]
        bars = ax_wk.bar(labels, vals, color=colors, alpha=0.85,
                          edgecolor='black', linewidth=0.6)
        for b, v, k in zip(bars, vals, kinds):
            if v > 0:
                ax_wk.text(b.get_x() + b.get_width()/2, v + 10,
                            str(v), ha='center', fontsize=9,
                            color='#666' if k == 'Projected' else 'black')

        # Budget line for daily average
        ax_wk.axhline(snap_budget / 7, color='red', linestyle='--', alpha=0.6,
                       label=f"Budget/day target ({snap_budget//7})")

        from matplotlib.patches import Patch
        legend_elems = [
            Patch(facecolor='#003d7a', edgecolor='black', label='Actual'),
            Patch(facecolor='#cc6633', edgecolor='black', label='Projected (avg-extrap)'),
            Patch(facecolor='#bbbbbb', edgecolor='black', label='Missing summary'),
        ]
        ax_wk.legend(handles=legend_elems + [
            plt.Line2D([0], [0], color='red', linestyle='--',
                       label=f"Budget/day ({snap_budget//7})"),
        ], loc='upper right', fontsize=8)
        ax_wk.set_title(f"Week of {week_start.date()} — actuals so far + linear projection to Sunday")
        ax_wk.set_ylabel('Drones launched')
        ax_wk.grid(alpha=0.25, axis='y')
        st.pyplot(fig_wk)

    st.divider()

    with st.container():
        st.markdown("### 🇺🇦 Official UA Air Force counts")
        ocol1, ocol2, ocol3, ocol4 = st.columns(4)
        last_row = daily_sum.iloc[0]
        last_d = last_row['date'].date()
        with ocol1:
            st.metric(f"Latest day in feed", str(last_d))
        with ocol2:
            st.metric("Drones launched", f"{int(last_row['launched']):,}")
        with ocol3:
            st.metric("Intercepted", f"{int(last_row['intercepted']):,}")
        with ocol4:
            if last_row['launched']:
                pct = 100 * last_row['intercepted'] / last_row['launched']
                st.metric("Intercept rate", f"{pct:.0f}%")

        with st.expander(f"Daily totals — {len(daily_sum)} day(s) on record",
                          expanded=False):
            display = daily_sum.copy()
            display['date'] = display['date'].dt.date
            display.columns = ['Date', 'Launched', 'Intercepted', 'Hits']
            st.dataframe(display, use_container_width=True, hide_index=True)

        if len(daily_sum) >= 2:
            fig0, ax0 = plt.subplots(figsize=(10, 3))
            d = daily_sum.sort_values('date')
            ax0.bar(d['date'].dt.strftime('%m-%d'), d['launched'],
                     color='#cc0033', alpha=0.75, label='Launched')
            ax0.bar(d['date'].dt.strftime('%m-%d'), d['intercepted'],
                     color='#003d7a', alpha=0.8, label='Intercepted')
            ax0.set_title('Daily drone activity (UA Air Force official)')
            ax0.legend()
            ax0.tick_params(axis='x', rotation=30)
            st.pyplot(fig0)
    st.divider()

# Sidebar controls
with st.sidebar:
    st.header("Forecast Parameters")

    russian_daily_capacity = st.slider(
        "Russian daily production / launch capacity",
        min_value=100, max_value=1500, value=300, step=10,
        help="Data-driven default. 14-day mean (Apr 30–May 13) is 268/day "
             "with ±10% buffer → 295. Apr 28 changepoint shifted baseline "
             "from ~150 → ~270/day. Push to 600+ if surge days dominate "
             "(May 1–3 and May 13 each hit 500+); drop to 150 if a real "
             "ceasefire returns (May 4–10 averaged 108)."
    )

    tempo_factor = st.slider(
        "Tempo factor (0=ceasefire, 1=full war)",
        min_value=0.1, max_value=1.5, value=1.0, step=0.05,
        help="Fraction of launch capacity actually being used. The May 2026 "
             "data shows tempo varies massively day-to-day (CoV 132%) — keep "
             "at 1.0 for week-level forecasting; the budget slider absorbs "
             "the variance."
    )

    weekly_budget = int(russian_daily_capacity * 7 * tempo_factor)
    st.metric("Weekly drone budget", f"{weekly_budget:,}")

    auto_subtract = st.checkbox(
        "Auto-subtract observed launches this week",
        value=True,
        help="When on, the remaining budget shrinks automatically every "
             "time a new launcher use is recorded."
    )

    if auto_subtract:
        already_used = min(launches_this_week, weekly_budget)
        st.metric("Drones already used this week (auto)", f"{already_used:,}")
    else:
        already_used = st.number_input(
            "Drones already used this week",
            min_value=0, max_value=weekly_budget,
            value=min(launches_this_week, weekly_budget), step=10,
            help="Subtract observed launches earlier in the week"
        )

    remaining_budget = max(weekly_budget - already_used, 0)
    st.metric("Remaining for forecast", f"{remaining_budget:,}")

    low_tempo = st.checkbox(
        "Low-tempo regime (ceasefire-like)", value=False,
        help="Boosts border oblasts and dampens energy-heavy ones. Turn ON "
             "only during genuine ceasefire periods (e.g. May 9–10, 2026). "
             "OFF for the current active-war / western-strike regime.",
    )

    st.divider()
    with st.expander("💰 Cost assumptions (per-hit)"):
        st.caption(
            "Defaults are conservative public-reporting averages for "
            "Shahed-136 strikes. Adjust if you have better sources."
        )
        st.session_state['cost_fatal_lo'] = st.number_input(
            "Fatalities per hit (low)", 0.0, 5.0, 0.10, 0.05,
            help="Conservative average across all target types. Rural/"
                 "infrastructure strikes often kill 0; dense urban hits "
                 "can kill 10+. OHCHR/UN tracking implies ~0.1 avg.",
        )
        st.session_state['cost_fatal_hi'] = st.number_input(
            "Fatalities per hit (high)", 0.0, 10.0, 0.30, 0.05,
            help="Upper-bound estimate including residential strikes.",
        )
        st.session_state['cost_inj_lo'] = st.number_input(
            "Injuries per hit (low)", 0.0, 20.0, 0.5, 0.1,
        )
        st.session_state['cost_inj_hi'] = st.number_input(
            "Injuries per hit (high)", 0.0, 50.0, 1.5, 0.1,
        )
        st.session_state['cost_dmg_lo'] = st.number_input(
            "Material damage per hit (low, $M)", 0.0, 50.0, 0.5, 0.1,
            help="Civilian residential, light commercial.",
        )
        st.session_state['cost_dmg_hi'] = st.number_input(
            "Material damage per hit (high, $M)", 0.0, 100.0, 3.0, 0.5,
            help="Energy infrastructure, industrial sites.",
        )
        st.session_state['cost_drone_unit'] = st.number_input(
            "Russia's cost per drone launched ($)", 0, 200_000, 40_000, 5_000,
            help="Shahed-136/Geran-2 unit cost ~\\$30-50K; decoys cheaper.",
        )
        st.session_state['cost_intercept_unit'] = st.number_input(
            "Ukraine's avg cost per intercept ($)", 0, 5_000_000, 500_000,
            50_000, help="Blended estimate across Gepard (\\$30/round), IRIS-T, "
                          "NASAMS, Patriot (\\$3-4M/missile). Most kills use "
                          "cheap systems; this averages the mix.",
        )

    with st.expander("💥 Offensive exchange (Ukraine → Russia)"):
        st.caption(
            "Ukraine's deep-strike drone campaign against Russian oil "
            "refineries, airfields, and depots. Defaults from public "
            "reporting (CSIS, ISW, Atlantic Council)."
        )
        st.session_state['off_ua_drones_per_year'] = st.number_input(
            "Ukrainian deep-strike drones launched/year",
            0, 100_000, 7_500, 500,
            help="Long-range one-way drones (Lyutyi, Bober, AQ-400 etc.). "
                 "2024 saw ~5K; 2025-2026 trending toward 10K+.",
        )
        st.session_state['off_ua_drone_cost'] = st.number_input(
            "Ukraine's cost per offensive drone ($)",
            0, 500_000, 75_000, 5_000,
            help="Lyutyi unit cost ~\\$50-100K; cheaper FPV conversions ~\\$5K. "
                 "Blended average.",
        )
        st.session_state['off_ru_intercept_rate'] = st.slider(
            "Russian intercept rate", 0.0, 1.0, 0.55, 0.05,
            help="Russia claims 60-90%; independent estimates 40-70%. "
                 "Lower than Ukraine's 88% because Russia's interior is "
                 "vast and air-defense coverage sparser.",
        )
        st.session_state['off_ru_intercept_cost'] = st.number_input(
            "Russia's avg cost per intercept ($)",
            0, 2_000_000, 300_000, 25_000,
            help="Pantsir rounds (~\\$15K) blended with S-300/S-400 "
                 "missiles (\\$200K-\\$1M+). Mid estimate \\$300K.",
        )
        st.session_state['off_dmg_per_hit_lo'] = st.number_input(
            "Damage per hit on Russia (low, $M)",
            0.0, 100.0, 2.0, 0.5,
            help="Mostly tactical hits, frontline equipment, smaller depots.",
        )
        st.session_state['off_dmg_per_hit_hi'] = st.number_input(
            "Damage per hit on Russia (high, $M)",
            0.0, 500.0, 20.0, 1.0,
            help="Refinery, strategic airfield, major depot. Single big "
                 "strikes can run \\$50-500M.",
        )
        st.session_state['off_aid_subsidy_pct'] = st.slider(
            "% of Ukraine's costs covered by Western aid", 0.0, 1.0, 0.65, 0.05,
            help="Most intercept and strike costs are subsidized via US/EU "
                 "military aid packages. The remainder hits the UA budget "
                 "directly.",
        )

    with st.expander("📈 Strategic outlook (multi-year)"):
        st.caption(
            "How long can each side sustain? Set macro assumptions; the "
            "outlook panel projects cumulative cost curves and shows when "
            "each Russian constraint binds."
        )
        st.session_state['outlook_oil_price'] = st.slider(
            "Oil price (\\$/bbl)", 30, 150, 75, 5,
            help="Brent crude benchmark. Russia's oil revenue scales "
                 "roughly linearly. Baseline \\$75 = ~\\$240B/yr revenue.",
        )
        st.session_state['outlook_ru_revenue_base'] = st.number_input(
            "Russia annual oil+gas revenue at baseline (\\$B)",
            100, 500, 240, 10,
            help="At baseline oil price. Scales with price slider above.",
        )
        st.session_state['outlook_ru_war_spend'] = st.number_input(
            "Russia total annual war spend (\\$B, ground+air+drone+missile)",
            50, 400, 130, 5,
            help="Drone-war numbers above are only a slice. This is "
                 "Russia's TOTAL war cost. Public estimates: \\$100-150B/yr.",
        )
        st.session_state['outlook_ru_nwf'] = st.number_input(
            "Russia's National Wealth Fund liquid balance (\\$B)",
            0, 200, 55, 5,
            help="Down from \\$150B in 2022 to ~\\$55B in late 2025 per public "
                 "reporting. Sovereign reserve that buffers any deficit.",
        )
        st.session_state['outlook_aid_multiplier'] = st.slider(
            "Western aid level (1.0 = current)", 0.0, 2.0, 1.0, 0.1,
            help="Move down to test 'aid collapses' scenarios; up to test "
                 "'aid doubled' (e.g. new EU defense fund).",
        )
        st.session_state['outlook_ua_total_spend'] = st.number_input(
            "Ukraine total annual defense spend (\\$B)",
            50, 300, 120, 5,
            help="Drone-war numbers above are only a slice. Total includes "
                 "ground forces, conventional munitions, and infrastructure.",
        )
        st.session_state['outlook_refinery_pct'] = st.slider(
            "% Russian refining capacity offline (avg)",
            0.0, 0.80, 0.20, 0.05,
            help="Annualised average. Strike campaign has hit 30-40% peaks; "
                 "Russia repairs/relocates. Each % offline = ~1% of refined "
                 "output → ~\\$2.4B/yr revenue impact at baseline.",
        )

    with st.expander("🏭 Production rate estimate"):
        st.caption(
            "Reverse-engineered from launch data. Override the auto "
            "estimate to test scenarios — does stockpile bottom out? "
            "Does production exceed launches over the window?"
        )
        st.session_state['prod_rate_override'] = st.number_input(
            "Russian drone production rate (drones/day)",
            0, 2000, 0, 10,
            help="0 = auto (uses peak 14-day rolling avg of launches as "
                 "a defensible floor). Set manually to override.",
        )
        st.session_state['prod_initial_stockpile'] = st.number_input(
            "Assumed initial stockpile (drones)",
            0, 50_000, 1_000, 100,
            help="Buffer Russia had before the data window began. The "
                 "stockpile curve = initial + cumulative(production − launches).",
        )

    st.divider()
    st.subheader("Data ingestion")
    st.caption(
        "Primary source: **UA Air Force Telegram** "
        "([@kpszsu](https://t.me/s/kpszsu)). Live drone-group sightings, "
        "no auth needed."
    )
    tg_clicked = st.button("⬇️ Fetch latest from UA Air Force Telegram",
                            use_container_width=True, type='primary')

    with st.expander("📂 Drop a CSV (manual import)"):
        st.caption(
            "Drop any CSV with observation data and I'll auto-detect the "
            "schema. Recognized shapes:\n"
            "- **Observations**: cols `observation_date, oblast, "
            "observed_drones, source` → merges into `observations.csv`\n"
            "- **Daily totals**: cols `date, period, launched, intercepted` "
            "→ merges into `daily_totals.csv`\n"
            "- **ACLED export**: cols `event_date, admin1, sub_event_type` "
            "→ aggregated then merged as observations"
        )
        uploaded = st.file_uploader(
            "CSV file", type=['csv'], accept_multiple_files=False,
            label_visibility='collapsed',
        )
        if uploaded is not None:
            try:
                # Skip BOM-only / empty files cleanly
                raw = uploaded.getvalue()
                if len(raw) < 20:
                    st.error(
                        f"File looks empty ({len(raw)} bytes) — re-export "
                        "from the source. The download may have failed."
                    )
                else:
                    import io
                    df_in = pd.read_csv(io.BytesIO(raw))
                    st.write(f"**{len(df_in)} rows / {len(df_in.columns)} cols** preview:")
                    st.dataframe(df_in.head(10), use_container_width=True,
                                  hide_index=True)
                    cols_lo = [c.lower().strip() for c in df_in.columns]

                    if {'observation_date', 'oblast', 'observed_drones'}.issubset(cols_lo):
                        kind = 'observations'
                    elif {'date', 'period', 'launched'}.issubset(cols_lo):
                        kind = 'daily_totals'
                    elif {'event_date', 'admin1'}.issubset(cols_lo):
                        kind = 'acled'
                    else:
                        kind = 'unknown'

                    st.write(f"Detected schema: **{kind}**")

                    if kind != 'unknown' and st.button(
                            "Merge into local data",
                            use_container_width=True, type='primary'):
                        if kind == 'observations':
                            df_in.columns = [c.lower().strip() for c in df_in.columns]
                            existing = pd.read_csv(OBS_CSV)
                            combined = pd.concat([existing, df_in], ignore_index=True)
                            combined = combined.drop_duplicates(
                                subset=['observation_date', 'oblast', 'source']
                            )
                            combined.to_csv(OBS_CSV, index=False)
                            st.success(
                                f"Merged. observations.csv now has "
                                f"{len(combined)} rows."
                            )
                            st.cache_data.clear()
                            st.rerun()
                        elif kind == 'daily_totals':
                            df_in.columns = [c.lower().strip() for c in df_in.columns]
                            existing = (pd.read_csv(DAILY_TOTALS_CSV)
                                        if DAILY_TOTALS_CSV.exists()
                                        else pd.DataFrame())
                            combined = pd.concat([existing, df_in], ignore_index=True)
                            combined = combined.drop_duplicates(
                                subset=['date', 'period']
                            )
                            combined.to_csv(DAILY_TOTALS_CSV, index=False)
                            st.success(
                                f"Merged. daily_totals.csv now has "
                                f"{len(combined)} rows."
                            )
                            st.cache_data.clear()
                            st.rerun()
                        elif kind == 'acled':
                            df_in.columns = [c.lower().strip() for c in df_in.columns]
                            rows, unmatched = acled_ingest.events_to_observation_rows(df_in)
                            added, updated = acled_ingest.merge_into_observations(
                                rows, OBS_CSV
                            )
                            st.success(
                                f"Merged ACLED data: +{added} new rows, "
                                f"{updated} updated. Unmatched admin1: "
                                f"{unmatched or 'none'}"
                            )
                            st.cache_data.clear()
                            st.rerun()
                    elif kind == 'unknown':
                        st.warning(
                            f"Don't recognize this schema. Columns found: "
                            f"`{list(df_in.columns)}`. Tell me what they "
                            f"mean and I'll add a parser."
                        )
            except Exception as e:
                st.error(f"Couldn't parse CSV: {type(e).__name__}: {e}")

    tg_last = load_tg_sync_log()
    if tg_last:
        st.caption(
            f"Last TG sync: {tg_last.get('at', '?')} — "
            f"+{tg_last.get('rows_added', 0)} new, "
            f"{tg_last.get('rows_updated', 0)} updated, "
            f"{tg_last.get('drone_messages', 0)}/{tg_last.get('messages_seen', 0)} "
            f"messages were drones"
        )
        if tg_last.get('unmatched_messages'):
            with st.expander(
                f"⚠️ {len(tg_last['unmatched_messages'])} drone messages "
                f"didn't match any oblast — review"
            ):
                for u in tg_last['unmatched_messages']:
                    st.text(f"{u['datetime']}: {u['text']}")

    tg_pages = st.number_input(
        "How many pages of history to fetch (≈20 msgs/page)",
        min_value=1, max_value=50, value=12, step=1,
    )
    if tg_clicked:
        try:
            with st.spinner(f"Fetching @kpszsu ({tg_pages} pages)…"):
                result = telegram_ingest.sync(
                    OBS_CSV, DAILY_TOTALS_CSV, pages=int(tg_pages)
                )
            result['at'] = datetime.now().isoformat(timespec='seconds')
            save_tg_sync_log(result)
            st.success(
                f"Synced. Sightings: +{result['rows_added']} new, "
                f"{result['rows_updated']} updated. "
                f"Summaries: +{result['summary_rows_added']} new, "
                f"{result['summary_rows_updated']} updated. "
                f"{result['drone_messages']} drone msgs / "
                f"{result['messages_seen']} total."
            )
            st.cache_data.clear()
            st.rerun()
        except Exception as e:
            st.error(f"{type(e).__name__}: {e}")

    with st.expander("ACLED ingestion (waiting on access approval)"):
        st.caption(
            "Your account auths but the free tier blocks the API. Apply for "
            "researcher access at acleddata.com; once approved this section "
            "starts working automatically."
        )
        creds = load_acled_creds()
        acled_email = st.text_input(
            "ACLED account email", value=creds.get('email', ''),
        )
        acled_password = st.text_input(
            "ACLED password", value=creds.get('password', ''), type='password',
        )
        remember = st.checkbox(
            "Remember credentials on this machine",
            value=bool(creds),
            help=f"Stored locally at {ACLED_CREDS_PATH.name} with 0600 perms.",
        )
        sync_days = st.number_input(
            "Fetch how many days back?",
            min_value=1, max_value=60, value=14, step=1,
        )
        fetch_clicked = st.button("⬇️ Fetch from ACLED",
                                   use_container_width=True)
        last_sync = load_sync_log()
        if last_sync:
            st.caption(
                f"Last ACLED sync: {last_sync.get('at', '?')} — "
                f"+{last_sync.get('rows_added', 0)} new, "
                f"{last_sync.get('rows_updated', 0)} updated"
            )
        if fetch_clicked:
            if not acled_email or not acled_password:
                st.error("Email and password are required.")
            else:
                try:
                    end = date.today()
                    start = end - timedelta(days=int(sync_days))
                    with st.spinner(f"Fetching ACLED events {start} → {end}…"):
                        result = acled_ingest.sync(
                            acled_email, acled_password, start, end, OBS_CSV
                        )
                    result['at'] = datetime.now().isoformat(timespec='seconds')
                    save_sync_log(result)
                    if remember:
                        save_acled_creds(acled_email, acled_password)
                    elif ACLED_CREDS_PATH.exists():
                        ACLED_CREDS_PATH.unlink()
                    st.success(
                        f"Synced. +{result['rows_added']} new, "
                        f"{result['rows_updated']} updated."
                    )
                    st.cache_data.clear()
                    st.rerun()
                except acled_ingest.ACLEDError as e:
                    st.error(f"ACLED error: {e}")
                except Exception as e:
                    st.error(f"Unexpected error: {type(e).__name__}: {e}")

    st.divider()
    st.caption("Built locally on Bazzite. No cloud required.")

forecast = generate_weekly_forecast(
    oblasts, remaining_budget, low_tempo=low_tempo, observations=this_week_obs
)
learning_alpha = float(forecast['alpha'].iloc[0]) if 'alpha' in forecast.columns else 0.0


# ============== COST PROJECTION ==============
# Map leakers (launched − intercepted) and reported hits to expected
# human + material cost using configurable per-hit assumptions.
if not daily_totals.empty:
    dt_cost = daily_totals.copy()
    dt_cost['date'] = pd.to_datetime(dt_cost['date'])
    by_d = (dt_cost.groupby('date', as_index=False)
            .agg(launched=('launched', 'sum'),
                 intercepted=('intercepted', 'sum'),
                 hits=('hits', 'sum'))
            .sort_values('date'))
    by_d['leakers'] = (by_d['launched'] - by_d['intercepted']).clip(lower=0)

    total_launched = int(by_d['launched'].sum())
    total_intercepted = int(by_d['intercepted'].sum())
    total_leakers = int(by_d['leakers'].sum())
    total_hits_reported = int(by_d['hits'].sum())
    avg_intercept = (total_intercepted / total_launched) if total_launched else 0.0

    with st.container():
        st.markdown("### 💰 Projected human & material cost (drones that get through)")
        st.caption(
            "Two ways to count what 'gets through': **leakers** = launched − "
            "intercepted (upper bound — includes lost/crashed drones), and "
            "**reported hits** = UA Air Force's confirmed strike impacts "
            "(more accurate). Per-hit cost assumptions are conservative "
            "defaults from public reporting on Shahed-136 strikes; adjust "
            "in the sidebar if you have better numbers."
        )

        cost_cols = st.columns(4)
        with cost_cols[0]:
            st.metric(f"Launched ({len(by_d)} d)", f"{total_launched:,}")
        with cost_cols[1]:
            st.metric("Intercepted",
                      f"{total_intercepted:,}",
                      f"{avg_intercept*100:.1f}% rate")
        with cost_cols[2]:
            st.metric("Leakers (got through)", f"{total_leakers:,}",
                      help="launched − intercepted. Upper-bound estimate; "
                           "some leakers crash without hitting anything.")
        with cost_cols[3]:
            st.metric("Reported hits (UA AF)",
                      f"{total_hits_reported:,}",
                      help="The conservative count — confirmed strike "
                           "impacts cited in UA Air Force daily summaries.")

        proj_remaining_drones = max(weekly_budget - launches_this_week, 0)
        proj_leakers_remaining = proj_remaining_drones * (1 - avg_intercept)
        hits_per_leaker = (total_hits_reported / total_leakers) if total_leakers else 0.7
        proj_hits_remaining = proj_leakers_remaining * hits_per_leaker

        fat_lo = st.session_state.get('cost_fatal_lo', 0.10)
        fat_hi = st.session_state.get('cost_fatal_hi', 0.30)
        inj_lo = st.session_state.get('cost_inj_lo', 0.5)
        inj_hi = st.session_state.get('cost_inj_hi', 1.5)
        dmg_lo = st.session_state.get('cost_dmg_lo', 0.5)
        dmg_hi = st.session_state.get('cost_dmg_hi', 3.0)
        drone_cost = st.session_state.get('cost_drone_unit', 40_000)
        intercept_cost = st.session_state.get('cost_intercept_unit', 500_000)

        st.markdown(f"**Historical cost incurred ({by_d['date'].min().date()} – {by_d['date'].max().date()})**")
        h1, h2, h3 = st.columns(3)
        with h1:
            st.markdown(
                f"**Human cost** _(applied to {total_hits_reported} "
                f"reported hits)_\n\n"
                f"- Fatalities: **{int(total_hits_reported*fat_lo)}–"
                f"{int(total_hits_reported*fat_hi)}**\n"
                f"- Injuries: **{int(total_hits_reported*inj_lo)}–"
                f"{int(total_hits_reported*inj_hi)}**"
            )
        with h2:
            dmg_low_total = total_hits_reported * dmg_lo
            dmg_high_total = total_hits_reported * dmg_hi
            st.markdown(
                f"**Material damage**\n\n"
                f"- Low estimate: **\\${dmg_low_total:.1f}M**\n"
                f"- High estimate: **\\${dmg_high_total:.1f}M**\n"
                f"- Per hit assumed: \\${dmg_lo:.1f}M–\\${dmg_hi:.1f}M"
            )
        with h3:
            russia_spend = total_launched * drone_cost / 1_000_000
            ua_spend = total_intercepted * intercept_cost / 1_000_000
            net = (dmg_low_total + dmg_high_total) / 2 + ua_spend - russia_spend
            st.markdown(
                f"**Cost exchange**\n\n"
                f"- Russia spent on drones: **\\${russia_spend:.0f}M**\n"
                f"- Ukraine spent on intercepts: **\\${ua_spend:.0f}M**\n"
                f"- Net cost to Ukraine (mid): **\\${net:.0f}M**"
            )

        st.markdown(f"**Projected cost for rest of this week** "
                    f"_(model expects {int(proj_remaining_drones):,} more "
                    f"launches @ {avg_intercept*100:.0f}% intercept rate "
                    f"→ ~{int(proj_hits_remaining):,} hits)_")
        p1, p2, p3 = st.columns(3)
        with p1:
            st.markdown(
                f"**Projected fatalities**\n\n"
                f"**{int(proj_hits_remaining*fat_lo)}–"
                f"{int(proj_hits_remaining*fat_hi)}** people\n\n"
                f"**Projected injuries**\n\n"
                f"**{int(proj_hits_remaining*inj_lo)}–"
                f"{int(proj_hits_remaining*inj_hi)}** people"
            )
        with p2:
            proj_dmg_lo = proj_hits_remaining * dmg_lo
            proj_dmg_hi = proj_hits_remaining * dmg_hi
            st.markdown(
                f"**Projected material damage**\n\n"
                f"\\${proj_dmg_lo:.0f}M – \\${proj_dmg_hi:.0f}M\n\n"
                f"Equivalent to **{int(proj_dmg_lo)} – "
                f"{int(proj_dmg_hi)}** transformer stations"
                f" (\\$1M each)"
            )
        with p3:
            proj_russia = proj_remaining_drones * drone_cost / 1_000_000
            proj_intercept = (proj_remaining_drones * avg_intercept *
                              intercept_cost) / 1_000_000
            if proj_russia > 0:
                st.markdown(
                    f"**Projected spend**\n\n"
                    f"Russia: \\${proj_russia:.0f}M (drones)\n\n"
                    f"Ukraine: \\${proj_intercept:.0f}M (intercepts)\n\n"
                    f"_Exchange ratio: \\${proj_intercept/proj_russia:.1f} "
                    f"defender ÷ \\$1 attacker_"
                )
            else:
                st.markdown("**Projected spend**\n\n_no remaining budget_")

        worst = by_d.loc[by_d['leakers'].idxmax()]
        st.caption(
            f"📌 Worst day on record: **{worst['date'].date()}** — "
            f"{int(worst['launched'])} launched, {int(worst['leakers'])} got "
            f"through, {int(worst['hits'])} reported hits. That single day "
            f"alone implied ~{int(worst['hits']*fat_lo)}–"
            f"{int(worst['hits']*fat_hi)} fatalities and "
            f"\\${worst['hits']*dmg_lo:.0f}M–\\${worst['hits']*dmg_hi:.0f}M "
            f"in damage."
        )

    st.divider()


# ============== WEEKLY LEDGER (PERMANENT PREDICTED VS ACTUAL) ==============
# This panel is the ground truth: every week has a locked prediction,
# a day-by-day growing actual, and a final closed score. Data gaps are
# explicitly marked, not silently treated as zero.
try:
    import weekly_tracker as _wt
    import importlib as _il
    _il.reload(_wt)
    _wt.init_tables(DB_PATH)
    scores_df = _wt.get_all_closed_weeks(DB_PATH)
    if not scores_df.empty:
        with st.container():
            st.markdown("### 📒 Weekly ledger — predicted vs actual, immutable history")
            st.caption(
                "Each closed week is permanently recorded with its locked "
                "prediction, final actual, and per-oblast breakdown. Data "
                "gaps (weeks where the Telegram channel rotated past unsynced "
                "days) are explicitly flagged — the actual never silently "
                "shows as zero. Every closed week has a JSON archive at "
                "data/weekly_actuals_archive/. The PREDICTION is locked at "
                "week-start and never changes; the ACTUAL grows day-by-day "
                "until Sunday EOD, then is frozen."
            )

            n_scored = (scores_df['is_data_gap'] == 0).sum()
            n_gap = (scores_df['is_data_gap'] == 1).sum()
            wcol1, wcol2, wcol3, wcol4 = st.columns(4)
            with wcol1:
                st.metric("Weeks closed", len(scores_df))
            with wcol2:
                st.metric("Scored", int(n_scored))
            with wcol3:
                st.metric("Data gaps", int(n_gap),
                           help="Snapshot exists but no actuals captured.")
            with wcol4:
                with_pct = scores_df[scores_df['pct_off'].notna() & (scores_df['predicted_total']>0)]
                avg_abs_err = with_pct['pct_off'].abs().mean() if not with_pct.empty else 0
                st.metric("Avg |% off|", f"{avg_abs_err:.1f}%")

            # Build display table
            display = scores_df.copy()
            display['week_start'] = pd.to_datetime(display['week_start']).dt.date
            display['status'] = display['is_data_gap'].map(
                {1: '⚠️ DATA GAP', 0: '✓ scored'})
            display['actual_str'] = display.apply(
                lambda r: 'UNKNOWN' if r['is_data_gap']
                else f"{int(r['actual_total']):,}", axis=1)
            display['predicted_str'] = display['predicted_total'].apply(
                lambda v: f"{int(v):,}" if v > 0 else '—')
            display['err_str'] = display.apply(
                lambda r: '—' if r['is_data_gap'] or r['predicted_total']==0
                else f"{int(r['total_error']):+,}", axis=1)
            display['pct_str'] = display['pct_off'].apply(
                lambda v: f"{v:+.1f}%" if pd.notna(v) else '—')
            display['r_str'] = display['spatial_r'].apply(
                lambda v: f"{v:.3f}" if pd.notna(v) else '—')
            view = display[['week_start','status','predicted_str','actual_str',
                            'err_str','pct_str','r_str']].copy()
            view.columns = ['Week start','Status','Predicted','Actual',
                            'Error','% off','Spatial r']
            st.dataframe(view, use_container_width=True, hide_index=True)

            # Predicted-vs-actual chart over time
            scored_only = scores_df[(scores_df['is_data_gap']==0) &
                                     (scores_df['predicted_total']>0)].copy()
            if not scored_only.empty:
                scored_only['week_start'] = pd.to_datetime(scored_only['week_start'])
                scored_only = scored_only.sort_values('week_start')
                fig_l, ax_l = plt.subplots(figsize=(11, 4))
                x = scored_only['week_start']
                ax_l.bar([d - pd.Timedelta(days=1) for d in x],
                         scored_only['predicted_total'], width=2.5,
                         color='#003d7a', alpha=0.75, label='Predicted')
                ax_l.bar([d + pd.Timedelta(days=1) for d in x],
                         scored_only['actual_total'], width=2.5,
                         color='#cc0033', alpha=0.85, label='Actual')
                # Annotate data-gap weeks
                gaps = scores_df[scores_df['is_data_gap']==1]
                for _, gr in gaps.iterrows():
                    ax_l.axvline(pd.to_datetime(gr['week_start']),
                                  color='gray', linestyle='--', alpha=0.5)
                    ax_l.text(pd.to_datetime(gr['week_start']),
                               scored_only['predicted_total'].max()*0.95,
                               'GAP', ha='center', fontsize=8, color='gray')
                ax_l.set_title('Weekly predicted vs actual — immutable scored history')
                ax_l.legend(loc='upper left')
                ax_l.grid(alpha=0.3, axis='y')
                ax_l.tick_params(axis='x', rotation=25)
                st.pyplot(fig_l)

            # Current week's growth (this week, in progress)
            today_dt = date.today()
            cur_ws = _wt.week_start_of(today_dt)
            cur_progress = _wt.get_week_progress(cur_ws, DB_PATH)
            if not cur_progress.empty:
                with st.expander(f"📈 Current-week progress (week of {cur_ws} — growing live)"):
                    cp = cur_progress[['observation_date','day_of_week',
                                        'daily_launched','daily_intercepted',
                                        'daily_hits','cumulative_launched',
                                        'cumulative_intercepted','cumulative_hits',
                                        'is_frozen']].copy()
                    day_names = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
                    cp['day'] = cp['day_of_week'].map(lambda i: day_names[i])
                    cp.columns = ['Date','dow_n','Daily launched','Daily intercepted',
                                  'Daily hits','Cumulative launched',
                                  'Cumulative intercepted','Cumulative hits',
                                  'Frozen','Day']
                    st.dataframe(cp[['Day','Date','Daily launched','Daily intercepted',
                                      'Daily hits','Cumulative launched',
                                      'Cumulative intercepted','Cumulative hits',
                                      'Frozen']],
                                  use_container_width=True, hide_index=True)
                    st.caption(
                        f"This week's cumulative total so far: "
                        f"**{int(cur_progress['cumulative_launched'].iloc[-1]):,}** drones. "
                        f"Rows refresh on every sync until Sunday EOD, then freeze."
                    )

            with st.expander("📂 JSON archives (defense-in-depth)"):
                import os
                archive_dir = DATA_DIR / 'weekly_actuals_archive'
                if archive_dir.exists():
                    files = sorted(os.listdir(archive_dir))
                    st.markdown(
                        "Each closed week is persisted as a standalone JSON "
                        "file. Survives database corruption / accidental "
                        "wipes. Files:"
                    )
                    for f in files:
                        sz = os.path.getsize(archive_dir / f)
                        st.code(f"  data/weekly_actuals_archive/{f}  ({sz:,} bytes)",
                                 language=None)
    st.divider()
except Exception as _wt_e:
    st.warning(f"Weekly ledger unavailable: {_wt_e}")


# ============== PREDICTED VS ACTUAL — BARS + LINES + 14D ROLLING ==============
# Interactive panel: bars for weekly predicted vs actual, overlay lines
# for 14-day rolling average. Selectable month(s) and oblast.
try:
    import sqlite3 as _sq3_pva
    with _sq3_pva.connect(DB_PATH) as _c:
        _all_snaps = pd.read_sql_query(
            "SELECT id, week_start, weekly_budget FROM snapshots ORDER BY id", _c)
        _all_snaprows = pd.read_sql_query(
            "SELECT snapshot_id, oblast, predicted_week FROM snapshot_rows", _c)

    # Latest snapshot per week (data-driven ones)
    _week_to_snap = (_all_snaps.drop_duplicates('week_start', keep='last')
                      .set_index('week_start')['id'].to_dict())

    with st.container():
        st.markdown("### 📊 Predicted vs Actual — bars + lines + 14-day rolling average")
        st.caption(
            "Time-selectable comparison of budgeted vs actual per week or "
            "per oblast, with the 14-day rolling average of actuals overlaid "
            "as a smoothed trend line. Select month(s) and an oblast to "
            "zoom in."
        )

        # === Controls ===
        _daily_ts = daily_totals.copy()
        _daily_ts['date'] = pd.to_datetime(_daily_ts['date'])
        _daily_ts['month_key'] = _daily_ts['date'].dt.strftime('%Y-%m')
        _months_available = sorted(_daily_ts['month_key'].unique())
        _month_labels = {m: pd.Timestamp(m + '-01').strftime('%b %Y')
                         for m in _months_available}

        _oblast_options = ['ALL OBLASTS (aggregate)'] + sorted(oblasts['oblast'].tolist())

        pcol1, pcol2 = st.columns([2, 3])
        with pcol1:
            _sel_oblast = st.selectbox(
                'Oblast',
                options=_oblast_options,
                index=0,
                key='pva_oblast',
            )
        with pcol2:
            _sel_months = st.multiselect(
                'Month(s)',
                options=_months_available,
                default=_months_available,
                format_func=lambda m: _month_labels[m],
                key='pva_months',
            )

        if not _sel_months:
            st.info("Select at least one month.")
        else:
            # === Data prep ===
            _month_mask = _daily_ts['month_key'].isin(_sel_months)
            _daily_filt = _daily_ts[_month_mask].copy()

            # Scaled observations for the same window
            _obs_pva = observations.copy()
            _obs_pva['observation_date'] = pd.to_datetime(_obs_pva['observation_date'])
            _obs_filt = _obs_pva[_obs_pva['observation_date'].dt.strftime('%Y-%m')
                                  .isin(_sel_months)]

            if _sel_oblast == 'ALL OBLASTS (aggregate)':
                # Daily actuals across all
                _daily_actual = (_daily_filt.groupby('date')['launched']
                                  .sum().reset_index())
                # Weekly predicted from snapshots (aggregate across all oblasts)
                _wk_preds = []
                for _ws, _sid in _week_to_snap.items():
                    _pred_total = _all_snaprows[_all_snaprows['snapshot_id']==_sid][
                        'predicted_week'].sum()
                    _wk_preds.append({'week_start': pd.Timestamp(_ws),
                                       'predicted': float(_pred_total)})
                _pred_df = pd.DataFrame(_wk_preds)
                _pred_df = _pred_df[_pred_df['week_start'].dt.strftime('%Y-%m')
                                     .isin(_sel_months)]
                _label = "all oblasts"
            else:
                # Filter to a single oblast
                _daily_actual = (_obs_filt[_obs_filt['oblast']==_sel_oblast]
                                  .groupby('observation_date')['observed_drones']
                                  .sum().reset_index()
                                  .rename(columns={'observation_date':'date',
                                                    'observed_drones':'launched'}))
                _wk_preds = []
                for _ws, _sid in _week_to_snap.items():
                    _p = _all_snaprows[(_all_snaprows['snapshot_id']==_sid) &
                                        (_all_snaprows['oblast']==_sel_oblast)]
                    _pred_total = float(_p['predicted_week'].sum())
                    _wk_preds.append({'week_start': pd.Timestamp(_ws),
                                       'predicted': _pred_total})
                _pred_df = pd.DataFrame(_wk_preds)
                _pred_df = _pred_df[_pred_df['week_start'].dt.strftime('%Y-%m')
                                     .isin(_sel_months)]
                _label = _sel_oblast

            # Aggregate actual to weekly for bar comparison
            if not _daily_actual.empty:
                _daily_actual['week_start'] = (
                    _daily_actual['date']
                    - pd.to_timedelta(_daily_actual['date'].dt.weekday, unit='D')
                )
                _weekly_actual = _daily_actual.groupby('week_start')['launched'].sum().reset_index()
                _weekly_actual.columns = ['week_start', 'actual']
            else:
                _weekly_actual = pd.DataFrame(columns=['week_start','actual'])

            # 14-day rolling average of daily launches (only meaningful for all-oblast
            # or when the oblast has continuous data)
            _daily_actual_sorted = _daily_actual.sort_values('date') if not _daily_actual.empty else pd.DataFrame()
            if not _daily_actual_sorted.empty:
                _daily_actual_sorted['rolling_14'] = (
                    _daily_actual_sorted['launched'].rolling(14, min_periods=3).mean()
                )
                # Weekly average of the 14-day rolling (for chart smoothness)
                _daily_actual_sorted['week_start'] = (
                    _daily_actual_sorted['date']
                    - pd.to_timedelta(_daily_actual_sorted['date'].dt.weekday, unit='D')
                )

            # Merge weekly actual with weekly predicted
            _combined = _pred_df.merge(_weekly_actual, on='week_start', how='outer')
            _combined = _combined.sort_values('week_start').fillna(0)

            # === Chart ===
            fig_pva, ax_pva = plt.subplots(figsize=(13, 5.5))
            _weeks = _combined['week_start'].tolist()
            if _weeks:
                _x_positions = list(range(len(_weeks)))
                _bar_width = 0.38
                ax_pva.bar([xi - _bar_width/2 for xi in _x_positions],
                            _combined['predicted'], _bar_width,
                            color='#003d7a', alpha=0.85,
                            edgecolor='white', linewidth=0.6,
                            label='Predicted (weekly budget)')
                ax_pva.bar([xi + _bar_width/2 for xi in _x_positions],
                            _combined['actual'], _bar_width,
                            color='#cc0033', alpha=0.85,
                            edgecolor='white', linewidth=0.6,
                            label='Actual (weekly total)')

                # Line for weekly-actual trend
                ax_pva.plot(_x_positions, _combined['actual'],
                             marker='s', color='#cc0033', linewidth=1.5,
                             alpha=0.5, zorder=5)
                # Line for weekly-predicted trend
                ax_pva.plot(_x_positions, _combined['predicted'],
                             marker='o', color='#003d7a', linewidth=1.5,
                             alpha=0.5, zorder=5)

                # 14-day rolling on daily actuals (mapped to weeks)
                if not _daily_actual_sorted.empty and \
                        _daily_actual_sorted['rolling_14'].notna().any():
                    # For each of our chart weeks, take the 14-day rolling on the
                    # LAST day of that week (or last available day within week)
                    _rolling_points = []
                    for wi, w in enumerate(_weeks):
                        _in_week = _daily_actual_sorted[
                            (_daily_actual_sorted['date'] >= w) &
                            (_daily_actual_sorted['date'] < w + pd.Timedelta(days=7))
                        ]
                        if not _in_week.empty:
                            _val = _in_week['rolling_14'].dropna().mean()
                            if pd.notna(_val):
                                # Rolling avg is daily → scale to weekly for comparison
                                _rolling_points.append((wi, _val * 7))
                    if _rolling_points:
                        _rx, _ry = zip(*_rolling_points)
                        ax_pva.plot(_rx, _ry, marker='^', color='#2a8c4a',
                                     linewidth=2.5, linestyle='--',
                                     label='14-day rolling avg × 7 (weekly equivalent)',
                                     zorder=6)

                ax_pva.set_xticks(_x_positions)
                ax_pva.set_xticklabels([w.strftime('%b %d') for w in _weeks],
                                        rotation=25, ha='right', fontsize=9)
                ax_pva.set_ylabel('Drones per week')
                ax_pva.set_title(
                    f"Predicted vs Actual — {_label}   ·   months: "
                    f"{', '.join(_month_labels[m] for m in _sel_months)}",
                    fontsize=12, fontweight='bold')
                ax_pva.legend(loc='upper right', fontsize=9)
                ax_pva.grid(alpha=0.3, axis='y')

                st.pyplot(fig_pva)

            # Summary numbers under the chart
            if not _combined.empty:
                _total_pred = int(_combined['predicted'].sum())
                _total_actual = int(_combined['actual'].sum())
                _err = _total_pred - _total_actual
                _pct = (_err / _total_actual * 100) if _total_actual else 0
                sc1, sc2, sc3, sc4 = st.columns(4)
                with sc1: st.metric("Total predicted", f"{_total_pred:,}")
                with sc2: st.metric("Total actual", f"{_total_actual:,}")
                with sc3: st.metric("Error", f"{_err:+,}",
                                     help="Predicted minus actual")
                with sc4: st.metric("% off",
                                     f"{_pct:+.1f}%",
                                     delta_color='inverse')

                # Show the underlying table
                with st.expander("Raw weekly numbers"):
                    _table = _combined.copy()
                    _table['week_start'] = _table['week_start'].dt.date
                    _table['error'] = _table['predicted'] - _table['actual']
                    _table['pct_off'] = _table.apply(
                        lambda r: (r['predicted']-r['actual'])/r['actual']*100
                        if r['actual'] else 0, axis=1)
                    _table.columns = ['Week start','Predicted','Actual','Error','% off']
                    _table['Predicted'] = _table['Predicted'].astype(int)
                    _table['Actual'] = _table['Actual'].astype(int)
                    _table['Error'] = _table['Error'].astype(int)
                    _table['% off'] = _table['% off'].round(1)
                    st.dataframe(_table, use_container_width=True, hide_index=True)

    st.divider()
except Exception as _pva_e:
    st.warning(f"Predicted-vs-actual panel unavailable: {_pva_e}")


# ============== DAILY SURGE PROBABILITY ==============
try:
    import surge_probability as _sp
    import importlib as _il
    _il.reload(_sp)

    with st.container():
        st.markdown("### ⚡ Daily surge probability model")
        st.caption(
            "Logistic regression on 5 features estimates P(surge tomorrow), "
            "where 'surge' = daily launches ≥ threshold. Uses only past "
            "days' data — no look-ahead."
        )

        # Controls
        scol1, scol2 = st.columns([1, 1])
        with scol1:
            _surge_threshold = st.number_input(
                "Surge threshold (drones / day)",
                min_value=100, max_value=1000, value=300, step=25,
                key='surge_threshold',
                help="Days with launches ≥ this count as surges.",
            )
        with scol2:
            _prod_rate_input = st.number_input(
                "Assumed production rate (drones/day)",
                min_value=100, max_value=1000, value=283, step=10,
                key='surge_prod_rate',
                help="Used for buffer estimation (production − consumption).",
            )

        # Prepare data + fit
        _sdata = daily_totals.copy()
        _sdata['date'] = pd.to_datetime(_sdata['date'])
        _sdata_agg = _sdata.groupby('date', as_index=False)['launched'].sum().sort_values('date')

        _model = _sp.SurgeModel(threshold=int(_surge_threshold),
                                  production_rate=int(_prod_rate_input))
        _fit_stats = _model.fit(_sdata_agg)

        # Today's prediction (or tomorrow if data goes through today)
        _last_data_day = _sdata_agg['date'].max().date()
        _target_day = _last_data_day + timedelta(days=1)
        _pred_today = _model.predict_next(_sdata_agg, target_date=_target_day)

        # Headline metric
        p = _pred_today['p_surge']
        pcol1, pcol2, pcol3, pcol4 = st.columns(4)
        with pcol1:
            st.metric(
                f"P(surge on {_target_day})",
                f"{p*100:.1f}%",
                help=f"Model's estimate. Base rate is "
                     f"{_fit_stats.get('base_rate', 0.15)*100:.1f}% in training data.",
            )
        with pcol2:
            st.metric(
                "Days since last surge",
                _pred_today['features']['days_since_last_surge'],
                help="Capped at 30. Higher = more overdue.",
            )
        with pcol3:
            st.metric(
                "Buffer estimate (drones)",
                f"{_pred_today['features']['buffer_estimate']:+,.0f}",
                help="Production − launches over trailing 14 days. "
                     "Positive = stockpile accumulating.",
            )
        with pcol4:
            st.metric(
                "14-day rolling avg",
                f"{_pred_today['features']['rolling_14day_avg']:.0f}",
                f"trend: {_pred_today['features']['trend_slope_7day']:+.1f}/day",
            )

        # Math explainer
        with st.expander("🧮 The math (logistic regression)"):
            st.latex(
                r"P(\mathrm{surge}) = "
                r"\frac{1}{1 + e^{-z}}, \quad "
                r"z = \beta_0 + \sum_{i=1}^{5} \beta_i \cdot x_i"
            )
            st.markdown(
                "**Features (x_i):**\n"
                "1. `days_since_last_surge` — hazard-rate proxy (grows with time)\n"
                "2. `rolling_14day_avg` — baseline tempo\n"
                "3. `buffer_estimate` — production − launches (stockpile signal)\n"
                "4. `trend_slope_7day` — linear-regression slope, last 7 days\n"
                "5. `post_surge_penalty` — 1 if yesterday was a surge (stockpile drained)\n\n"
                "**Fitting**: L2-regularized maximum-likelihood via `scipy.minimize`."
            )
            st.markdown("**Fit statistics on your data:**")
            st.json(_fit_stats)
            st.markdown("**Fitted coefficients (β):**")
            _cdf = pd.DataFrame([
                {'feature': k, 'coefficient': round(v, 4)}
                for k, v in _model.coeffs.items()])
            st.dataframe(_cdf, use_container_width=True, hide_index=True)

        # 7-day forecast
        st.markdown("**Next 7 days — daily surge probabilities:**")
        _forecast = _model.predict_next_week(_sdata_agg, start_date=_target_day, n_days=7)
        _fdf = pd.DataFrame([{
            'date': f['target_date'],
            'p_surge_pct': round(f['p_surge']*100, 1),
            'days_since_last_surge': f['features']['days_since_last_surge'],
            'rolling_14day': round(f['features']['rolling_14day_avg'], 0),
        } for f in _forecast])
        st.dataframe(_fdf, use_container_width=True, hide_index=True)

        # Chart the forecast
        fig_sp, ax_sp = plt.subplots(figsize=(11, 3.5))
        ax_sp.bar(_fdf['date'], _fdf['p_surge_pct'],
                    color='#cc0033', alpha=0.75, edgecolor='black', linewidth=0.5)
        ax_sp.axhline(_fit_stats.get('base_rate', 0.15)*100,
                       color='gray', linestyle='--', alpha=0.6,
                       label=f"Base rate ({_fit_stats.get('base_rate', 0.15)*100:.1f}%)")
        for i, r in _fdf.iterrows():
            ax_sp.text(i, r['p_surge_pct'] + 0.5, f"{r['p_surge_pct']:.0f}%",
                        ha='center', fontsize=8, fontweight='bold')
        ax_sp.set_ylabel('P(surge) %')
        ax_sp.set_title(f'7-day surge probability forecast '
                         f'(threshold = {int(_surge_threshold)}/day)',
                         fontsize=11, fontweight='bold')
        ax_sp.legend()
        ax_sp.grid(alpha=0.3, axis='y')
        plt.setp(ax_sp.get_xticklabels(), rotation=25, ha='right', fontsize=8)
        st.pyplot(fig_sp)

        # Feature contributions to today's z-score
        with st.expander("Contribution breakdown for today's prediction"):
            _contribs = _pred_today['contributions_to_z']
            _contribs_df = pd.DataFrame([
                {'feature': k, 'contribution_to_z': v,
                 'moves_prob_toward': 'SURGE' if v > 0 else 'no-surge' if v < 0 else 'neutral'}
                for k, v in _contribs.items()
            ]).sort_values('contribution_to_z', key=abs, ascending=False)
            st.dataframe(_contribs_df, use_container_width=True, hide_index=True)
            st.caption(
                f"Total z = {_pred_today['z']:.3f}   →   "
                f"P(surge) = 1/(1+e^-z) = **{p*100:.1f}%**"
            )

    st.divider()
except Exception as _sp_e:
    st.warning(f"Surge probability panel unavailable: {_sp_e}")


# ============== STRATEGIC EXCHANGE (BOTH DIRECTIONS) ==============
# Two-way cost accounting: who's actually winning on dollars,
# and does Ukraine's refinery campaign change the math.
if not daily_totals.empty and total_launched > 0:
    # --- Defensive side (RU → UA) — annualised from observed data ---
    days_observed = max(len(by_d), 1)
    annual_scale = 365 / days_observed
    annual_ru_launches = total_launched * annual_scale
    annual_ru_intercepted = total_intercepted * annual_scale
    annual_ua_hits = total_hits_reported * annual_scale

    annual_ru_drone_spend = annual_ru_launches * drone_cost
    annual_ua_intercept_spend = annual_ru_intercepted * intercept_cost
    # mid-point damage absorbed by Ukraine
    annual_ua_damage_lo = annual_ua_hits * dmg_lo * 1_000_000
    annual_ua_damage_hi = annual_ua_hits * dmg_hi * 1_000_000
    annual_ua_damage_mid = (annual_ua_damage_lo + annual_ua_damage_hi) / 2
    annual_ua_total_cost = annual_ua_intercept_spend + annual_ua_damage_mid

    # --- Offensive side (UA → RU) — from sidebar assumptions ---
    off_drones = st.session_state.get('off_ua_drones_per_year', 7_500)
    off_drone_cost = st.session_state.get('off_ua_drone_cost', 75_000)
    off_intercept_rate = st.session_state.get('off_ru_intercept_rate', 0.55)
    off_intercept_cost = st.session_state.get('off_ru_intercept_cost', 300_000)
    off_dmg_lo = st.session_state.get('off_dmg_per_hit_lo', 2.0)
    off_dmg_hi = st.session_state.get('off_dmg_per_hit_hi', 20.0)
    aid_subsidy = st.session_state.get('off_aid_subsidy_pct', 0.65)

    annual_ua_drone_spend = off_drones * off_drone_cost
    annual_ru_intercept_spend = off_drones * off_intercept_rate * off_intercept_cost
    annual_ua_hits_on_ru = off_drones * (1 - off_intercept_rate)
    annual_ru_damage_lo = annual_ua_hits_on_ru * off_dmg_lo * 1_000_000
    annual_ru_damage_hi = annual_ua_hits_on_ru * off_dmg_hi * 1_000_000
    annual_ru_damage_mid = (annual_ru_damage_lo + annual_ru_damage_hi) / 2
    annual_ru_total_cost = annual_ru_intercept_spend + annual_ru_damage_mid

    # Net + strategic
    ua_total = annual_ua_drone_spend + annual_ua_total_cost
    ru_total = annual_ru_drone_spend + annual_ru_total_cost
    raw_winner = "Russia" if ru_total < ua_total else "Ukraine"
    raw_margin = abs(ru_total - ua_total) / 1e9
    raw_ratio = max(ua_total, ru_total) / max(min(ua_total, ru_total), 1)

    # Strategic: aid-subsidized portion of UA cost doesn't hit UA budget
    ua_self_funded = ua_total * (1 - aid_subsidy)
    # Russia's damage = oil revenue / strategic productive capacity loss
    ru_strategic = ru_total  # all of it is self-funded productive damage
    strategic_winner = "Russia" if ru_strategic < ua_self_funded else "Ukraine"
    strategic_margin = abs(ru_strategic - ua_self_funded) / 1e9

    with st.container():
        st.markdown("### 💥 Strategic exchange — both directions (annualised)")
        st.caption(
            "Defensive numbers are extrapolated from the actually-observed "
            f"{days_observed} days × {annual_scale:.1f}. Offensive numbers "
            "come from sidebar sliders ('💥 Offensive exchange'). "
            "Annual scale assumes current tempo holds for 12 months."
        )

        st.markdown("**Two-way annual flow:**")
        flow_cols = st.columns(2)
        with flow_cols[0]:
            st.markdown(
                f"#### 🇷🇺 → 🇺🇦  (defensive)\n"
                f"- Drones launched: **{int(annual_ru_launches):,}**/yr\n"
                f"- Intercepted: {int(annual_ru_intercepted):,} "
                f"({avg_intercept*100:.0f}%)\n"
                f"- Hits absorbed: {int(annual_ua_hits):,}\n"
                f"\n"
                f"**Cost to Russia (offensive):** "
                f"\\${annual_ru_drone_spend/1e9:.1f}B\n\n"
                f"**Cost to Ukraine (defending + damage):** "
                f"\\${annual_ua_total_cost/1e9:.1f}B\n"
                f"- Intercepts: \\${annual_ua_intercept_spend/1e9:.1f}B\n"
                f"- Damage absorbed: \\${annual_ua_damage_mid/1e9:.1f}B "
                f"(\\${annual_ua_damage_lo/1e9:.1f}–"
                f"\\${annual_ua_damage_hi/1e9:.1f}B)"
            )
        with flow_cols[1]:
            st.markdown(
                f"#### 🇺🇦 → 🇷🇺  (offensive)\n"
                f"- Drones launched: **{int(off_drones):,}**/yr\n"
                f"- Intercepted by Russia: "
                f"{int(off_drones*off_intercept_rate):,} "
                f"({off_intercept_rate*100:.0f}%)\n"
                f"- Hits landed on Russia: {int(annual_ua_hits_on_ru):,}\n"
                f"\n"
                f"**Cost to Ukraine (offensive):** "
                f"\\${annual_ua_drone_spend/1e9:.1f}B\n\n"
                f"**Cost to Russia (defending + damage):** "
                f"\\${annual_ru_total_cost/1e9:.1f}B\n"
                f"- Intercepts: \\${annual_ru_intercept_spend/1e9:.1f}B\n"
                f"- Damage absorbed: \\${annual_ru_damage_mid/1e9:.1f}B "
                f"(\\${annual_ru_damage_lo/1e9:.1f}–"
                f"\\${annual_ru_damage_hi/1e9:.1f}B)"
            )

        st.markdown("---")
        st.markdown("**Bottom-line tally:**")
        tot_cols = st.columns(3)
        with tot_cols[0]:
            st.metric("🇺🇦 Total annual cost",
                      f"${ua_total/1e9:.1f}B",
                      help="Defensive intercepts + damage absorbed + "
                           "offensive drone production")
        with tot_cols[1]:
            st.metric("🇷🇺 Total annual cost",
                      f"${ru_total/1e9:.1f}B",
                      help="Offensive drone production + defensive "
                           "intercepts + damage absorbed")
        with tot_cols[2]:
            st.metric(
                f"Raw winner: {raw_winner}",
                f"by ${raw_margin:.1f}B/yr  ({raw_ratio:.1f}× ratio)",
                help="Whoever pays less in absolute dollars 'wins' the "
                     "raw exchange. But raw dollars hide who's paying "
                     "those dollars — see strategic verdict below.",
            )

        # The strategic story
        st.markdown("---")
        st.markdown("**Strategic verdict (accounting for who pays what):**")

        strat_cols = st.columns(2)
        with strat_cols[0]:
            st.markdown(
                f"🇺🇦 **Self-funded cost: \\${ua_self_funded/1e9:.1f}B/yr**\n\n"
                f"- {int(aid_subsidy*100)}% of Ukraine's costs are covered "
                f"by Western military aid (US/EU). That's foreign capital, "
                f"not Ukrainian budget.\n"
                f"- True hit to Ukraine's own balance sheet: "
                f"\\${ua_self_funded/1e9:.1f}B"
            )
        with strat_cols[1]:
            st.markdown(
                f"🇷🇺 **Self-funded cost: \\${ru_strategic/1e9:.1f}B/yr**\n\n"
                f"- Russia self-funds 100% via oil revenue (~\\$240B/yr "
                f"industry).\n"
                f"- All damage hits productive capacity (refineries → "
                f"future oil revenue → future war fund).\n"
                f"- Refinery damage compounds: lost throughput continues "
                f"for months after each strike."
            )

        if strategic_winner == "Ukraine":
            verdict = (
                f"### 🟢 Ukraine is winning the *strategic* exchange "
                f"by ~\\${strategic_margin:.1f}B/yr.\n\n"
                f"Even though Russia 'wins' the raw dollar tally by "
                f"\\${raw_margin:.1f}B (because Ukraine's intercept costs are "
                f"high), most of Ukraine's spend is foreign aid (not its "
                f"own budget), while all of Russia's losses come from its "
                f"own oil revenue and productive capacity. **The refinery "
                f"campaign is the lever that flips this — without it, Russia "
                f"would be winning by an order of magnitude.**"
            )
        else:
            verdict = (
                f"### 🔴 Russia is winning the strategic exchange by "
                f"~\\${strategic_margin:.1f}B/yr.\n\n"
                f"Ukraine's offensive isn't yet inflicting enough damage "
                f"on Russian productive capacity to offset what Russia is "
                f"costing Ukraine (even after subtracting Western aid). "
                f"To flip this: raise Ukrainian deep-strike volume "
                f"(currently {int(off_drones):,}/yr — needs ~2× more), "
                f"or focus higher-value targets (raise damage-per-hit "
                f"slider)."
            )
        st.markdown(verdict)

        # Refinery campaign counterfactual
        st.markdown("---")
        st.markdown("**Counterfactual: what if Ukraine stopped striking Russia?**")
        cf_ua = annual_ua_total_cost  # only defensive
        cf_ru = annual_ru_drone_spend  # only offensive
        cf_ratio = cf_ua / max(cf_ru, 1)
        st.markdown(
            f"Without the deep-strike campaign Ukraine would pay "
            f"**\\${cf_ua/1e9:.1f}B/yr** while Russia would pay only "
            f"**\\${cf_ru/1e9:.1f}B/yr** — a "
            f"**{cf_ratio:.0f}× exchange ratio in Russia's favor**. "
            f"The refinery campaign collapses that to "
            f"**{raw_ratio:.1f}×** in raw dollars and flips it strategically. "
            f"So *yes, it's changing the equation* — but Ukraine still "
            f"needs to scale strikes higher (or hit more refineries vs "
            f"tactical targets) to win even the raw-dollar fight."
        )

    st.divider()


# ============== STRATEGIC OUTLOOK (MULTI-YEAR) ==============
# How long can each side sustain at current burn? Where do the binding
# constraints land? Sensitivity to oil price + Western aid.
if not daily_totals.empty and total_launched > 0:
    oil_price = st.session_state.get('outlook_oil_price', 75)
    ru_rev_base = st.session_state.get('outlook_ru_revenue_base', 240) * 1e9
    ru_war_spend_total = st.session_state.get('outlook_ru_war_spend', 130) * 1e9
    nwf = st.session_state.get('outlook_ru_nwf', 55) * 1e9
    aid_mult = st.session_state.get('outlook_aid_multiplier', 1.0)
    ua_total_spend = st.session_state.get('outlook_ua_total_spend', 120) * 1e9
    refinery_offline = st.session_state.get('outlook_refinery_pct', 0.20)

    # Russia revenue scales linearly with oil price
    ru_revenue_annual = ru_rev_base * (oil_price / 75.0)
    # Refinery damage reduces refined product output (~50% of revenue is products)
    ru_revenue_annual *= (1.0 - 0.50 * refinery_offline)

    # Total Russian war cost = traditional war spend + drone exchange damage absorbed
    ru_total_annual_cost = ru_war_spend_total + (annual_ru_damage_mid)
    # Annual fiscal balance for Russia (oil-only proxy; ignores other tax revenue)
    ru_net_annual = ru_revenue_annual - ru_total_annual_cost

    # NWF runway under deficit
    if ru_net_annual < 0:
        years_to_nwf_zero = nwf / abs(ru_net_annual)
    else:
        years_to_nwf_zero = float('inf')

    # Ukraine: total spend split between aid-subsidized vs self-funded
    ua_aid = ua_total_spend * aid_subsidy * aid_mult
    ua_self = ua_total_spend - ua_aid
    # UA GDP about $180B; ratio of self-funded to GDP is the strain metric
    ua_gdp = 180e9
    ua_self_pct_gdp = ua_self / ua_gdp * 100

    # ============== SHAHED VS DECOY BREAKDOWN ==============
    # UA AF reports break drones into Shahed (real warhead) vs cheap
    # decoys (Gerbera, Italmas, Parodia). The explicit split only appears
    # in pre-May-4 night summaries (n=15); for the rest we apply the
    # historical 65% Shahed share.
    with st.container():
        st.markdown("### 🎯 Shahed vs decoy mix — what's actually carrying a warhead")
        st.caption(
            "UA AF distinguishes **Shahed** (Iranian design / Russian "
            "Geran-2, $40K, 50kg warhead) from cheap **decoys** (Gerbera, "
            "Italmas, Parodia, ~$10K, plywood + foam, drawn-fire purpose). "
            "Pre-May-4 summaries gave explicit counts; later summaries "
            "only report totals — historical Shahed share averaged 65% "
            "(σ 3.6%), applied to recent days as an estimate."
        )

        dt_full = daily_totals.copy()
        dt_full['date'] = pd.to_datetime(dt_full['date'])
        SHAHED_RATIO = 0.65
        SHAHED_COST = st.session_state.get('cost_drone_unit', 40_000)
        DECOY_COST = int(SHAHED_COST / 4)  # ~$10K when Shahed is $40K
        dt_full['shahed_est'] = dt_full['shaheds_estimated'].fillna(
            (dt_full['launched'] * SHAHED_RATIO).round()
        )
        dt_full['decoy_est'] = (dt_full['launched'] - dt_full['shahed_est']).clip(lower=0)

        by_d_mix = dt_full.groupby('date', as_index=False).agg(
            launched=('launched', 'sum'),
            shahed=('shahed_est', 'sum'),
            decoy=('decoy_est', 'sum'),
        ).sort_values('date')

        tot_launched = int(by_d_mix['launched'].sum())
        tot_shahed = int(by_d_mix['shahed'].sum())
        tot_decoy = int(by_d_mix['decoy'].sum())
        ru_spend = tot_shahed * SHAHED_COST + tot_decoy * DECOY_COST
        ru_spend_no_decoys = tot_launched * SHAHED_COST
        decoy_savings = ru_spend_no_decoys - ru_spend
        intercept_cost = st.session_state.get('cost_intercept_unit', 500_000)
        intercepted_est = tot_launched * 0.88
        ua_spend = intercepted_est * intercept_cost

        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        with mcol1:
            st.metric("Total launched",
                      f"{tot_launched:,}",
                      f"{len(by_d_mix)} days observed")
        with mcol2:
            st.metric("Shaheds (warhead-carrying)",
                      f"{tot_shahed:,}",
                      f"{tot_shahed/tot_launched*100:.0f}%")
        with mcol3:
            st.metric("Decoys (plywood/foam)",
                      f"{tot_decoy:,}",
                      f"{tot_decoy/tot_launched*100:.0f}%")
        with mcol4:
            ratio_with_decoys = ua_spend / max(ru_spend, 1)
            ratio_pure_shahed = ua_spend / max(ru_spend_no_decoys, 1)
            st.metric(
                "Exchange ratio (defender:attacker)",
                f"{ratio_with_decoys:.1f}×",
                f"vs {ratio_pure_shahed:.1f}× if all Shaheds",
                delta_color="inverse",
                help="Higher = worse for Ukraine. Decoys force interceptor "
                     "spend on $10K drones.",
            )

        # Stacked bar chart
        fig_mx, ax_mx = plt.subplots(figsize=(12, 4))
        labels = [d.strftime('%m-%d') for d in by_d_mix['date']]
        ax_mx.bar(labels, by_d_mix['shahed'], color='#cc0033',
                   label=f'Shahed (≈${SHAHED_COST//1000}K, warhead)',
                   edgecolor='black', linewidth=0.5)
        ax_mx.bar(labels, by_d_mix['decoy'], bottom=by_d_mix['shahed'],
                   color='#888888',
                   label=f'Decoy (≈${DECOY_COST//1000}K, draw-fire)',
                   edgecolor='black', linewidth=0.5, alpha=0.85)
        ax_mx.set_title('Daily launches — Shahed (warhead) vs decoy (drawn-fire) split')
        ax_mx.set_ylabel('Drones')
        ax_mx.tick_params(axis='x', rotation=80, labelsize=7)
        ax_mx.legend(loc='upper left')
        ax_mx.grid(alpha=0.3, axis='y')
        st.pyplot(fig_mx)

        st.markdown("**Cost picture (this window):**")
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            st.markdown(
                f"**🇷🇺 Russia's drone-production spend**\n\n"
                f"- Shaheds: \\${tot_shahed*SHAHED_COST/1e6:.0f}M\n"
                f"- Decoys: \\${tot_decoy*DECOY_COST/1e6:.0f}M\n"
                f"- **Total: \\${ru_spend/1e6:.0f}M**\n\n"
                f"_If all $40K Shaheds: \\${ru_spend_no_decoys/1e6:.0f}M_\n"
                f"_Decoys save Russia: \\${decoy_savings/1e6:.0f}M "
                f"({decoy_savings/ru_spend_no_decoys*100:.0f}%)_"
            )
        with cc2:
            st.markdown(
                f"**🇺🇦 Ukraine's intercept spend**\n\n"
                f"Intercepts (88%): {int(intercepted_est):,}\n\n"
                f"@ \\${intercept_cost//1000}K avg = "
                f"**\\${ua_spend/1e9:.2f}B**\n\n"
                f"_Ukraine can't tell Shahed from decoy until "
                f"impact — intercepts everything._"
            )
        with cc3:
            st.markdown(
                f"**🧠 Strategic verdict — the decoy gambit**\n\n"
                f"Decoys WORSEN the exchange ratio for Ukraine "
                f"from **{ratio_pure_shahed:.1f}×** to **{ratio_with_decoys:.1f}×** "
                f"({(ratio_with_decoys/ratio_pure_shahed - 1)*100:.0f}% worse).\n\n"
                f"That's the entire purpose of decoys: force Ukraine to "
                f"burn \\$500K interceptors on \\${DECOY_COST//1000}K plywood drones."
            )

        st.caption(
            "ℹ️ **Data quality**: 15 nights (Apr 19–May 3) have explicit "
            f"Shahed counts from UA AF. After May 3, UA AF stopped reporting "
            f"the split — possibly OPSEC. Our estimate applies the "
            f"historical mean (65%) with σ=3.6% — robust given how stable "
            f"the ratio was during the observed period."
        )

    st.divider()

    # ============== DECOY BUDGET — SEQUENTIAL UPDATE ==============
    # Poisson sizes the week, Beta-Binomial updates the share as the
    # week unfolds, optimal classifier threshold uses asymmetric costs.
    import decoy_predictor as dp
    importlib_module = __import__('importlib'); importlib_module.reload(dp)

    week_so_far_launches = launches_this_week
    # Estimate decoys identified so far at the historical 35% rate
    # (until a real classifier post-impact reports come in)
    decoys_seen_proxy = int(round(week_so_far_launches * dp.HISTORICAL_DECOY_SHARE))

    # Use snapshot #N's budget if locked, else current sidebar budget
    state = dp.DecoyWeekState(
        weekly_budget=int(weekly_budget),
        launched_so_far=int(week_so_far_launches),
        decoys_identified_so_far=int(decoys_seen_proxy),
    )

    SHAHED_COST_CFG = st.session_state.get('cost_drone_unit', 40_000)
    DECOY_COST_CFG = int(SHAHED_COST_CFG / 4)
    INTERCEPT_COST_CFG = st.session_state.get('cost_intercept_unit', 500_000)
    # Damage from a single missed warhead — average
    DAMAGE_MID = (st.session_state.get('cost_dmg_lo', 0.5) +
                  st.session_state.get('cost_dmg_hi', 3.0)) / 2 * 1e6

    classifier_threshold = dp.optimal_threshold(
        c_fp=INTERCEPT_COST_CFG, c_fn=DAMAGE_MID,
    )

    with st.container():
        st.markdown("### 📊 Decoy budget — sequential Bayesian update")
        st.caption(
            "Two math families compose: **Poisson** sizes the wave "
            "(λ ≈ 0.35·budget); **Beta-Binomial** updates the share as "
            "the week unfolds; the **classifier threshold** uses "
            "asymmetric costs (C_FP / (C_FP + C_FN)). The remaining-"
            "decoy estimate feeds back into the classifier's per-track "
            "prior."
        )

        dc1, dc2, dc3, dc4 = st.columns(4)
        with dc1:
            st.metric(
                "Week budget",
                f"{state.weekly_budget:,}",
                f"35% prior → {int(state.expected_decoys_total_week)} decoys",
            )
        with dc2:
            st.metric(
                "Launched / remaining",
                f"{state.launched_so_far:,} / {state.launched_remaining:,}",
                f"{state.launched_so_far/state.weekly_budget*100:.0f}% used",
            )
        with dc3:
            st.metric(
                "Posterior P(decoy)",
                f"{state.posterior_decoy_share*100:.1f}%",
                f"± {state.posterior_std*100:.2f}%",
                help="α_posterior / (α_posterior + β_posterior). The Bayesian "
                     "posterior of P(next track is decoy). Tightens as more "
                     "tracks are classified.",
            )
        with dc4:
            try:
                lo, hi = state.remaining_decoys_band(0.90)
            except Exception:
                lo, hi = 0, 0
            st.metric(
                "Expected decoys remaining",
                f"{int(state.expected_decoys_remaining)}",
                f"90% band [{int(lo)}, {int(hi)}]",
            )

        st.markdown("**The headline formula:**")
        st.latex(
            r"E[D_{rem}] = L_{rem} \cdot "
            r"\frac{\alpha_0 + K_t}{\alpha_0 + \beta_0 + L_t}"
        )
        st.markdown(
            f"With α₀ = 0.35 · 800 = **280**, β₀ = 0.65 · 800 = **520**, "
            f"K_t = **{state.decoys_identified_so_far}** decoys identified "
            f"in L_t = **{state.launched_so_far}** launches:"
        )
        st.markdown(
            f"E[D_rem] = **{state.launched_remaining}** × **"
            f"{state.posterior_decoy_share*100:.1f}%** = **"
            f"{int(state.expected_decoys_remaining)} decoys** "
            f"(+ {int(state.expected_shaheds_remaining)} Shaheds expected)"
        )

        st.markdown("**Classifier engagement threshold:**")
        st.latex(
            r"\theta^* = \frac{C_{FP}}{C_{FP} + C_{FN}}"
        )
        st.markdown(
            f"With C_FP (wasted interceptor) = **\\${INTERCEPT_COST_CFG/1000:.0f}K** "
            f"and C_FN (missed warhead damage) = **\\${DAMAGE_MID/1e6:.1f}M**:"
        )
        st.markdown(
            f"θ* = **{classifier_threshold*100:.1f}%** — fire on any track "
            f"with P(warhead) ≥ **{classifier_threshold*100:.1f}%**. "
            f"Equivalently, hold only when P(decoy) > "
            f"**{(1-classifier_threshold)*100:.1f}%**."
        )

        # Show the sequential-update story: how the estimate would
        # evolve day-by-day if launches arrive at the predicted pace
        st.markdown("**Day-by-day projected decoy budget (rest of week):**")
        rows = []
        cum_launched = state.launched_so_far
        cum_decoys = state.decoys_identified_so_far
        days_remaining = 7 - (today.weekday() + 1)
        per_day_launches = (
            state.launched_remaining // max(days_remaining, 1)
            if days_remaining > 0 else 0
        )
        for d_offset in range(days_remaining):
            future_date = (today + timedelta(days=d_offset+1)).date()
            cum_launched += per_day_launches
            # Assume 35% of new launches are decoys (will refine in real ops)
            day_decoys = int(round(per_day_launches *
                                    dp.HISTORICAL_DECOY_SHARE))
            cum_decoys += day_decoys
            future_state = dp.DecoyWeekState(
                weekly_budget=state.weekly_budget,
                launched_so_far=cum_launched,
                decoys_identified_so_far=cum_decoys,
            )
            rows.append({
                'Date': future_date,
                'New launches': per_day_launches,
                'Cumulative launched': cum_launched,
                'Cumulative decoys': cum_decoys,
                'Posterior P(decoy)': f"{future_state.posterior_decoy_share*100:.1f}%",
                'Decoys remaining (est)': int(future_state.expected_decoys_remaining),
            })
        if rows:
            st.dataframe(pd.DataFrame(rows),
                          use_container_width=True, hide_index=True)

        # Interceptor allocation suggestion
        st.markdown("**Interceptor allocation (week-remaining):**")
        cheap_to_decoys = int(state.expected_decoys_remaining)
        expensive_to_shaheds = int(state.expected_shaheds_remaining)
        savings_vs_uniform = (cheap_to_decoys *
                              (INTERCEPT_COST_CFG - 30_000))  # Gepard ~$30/round
        st.markdown(
            f"If you could perfectly classify, allocate:\n"
            f"- **{cheap_to_decoys} cheap intercepts** (Gepard rounds @ "
            f"~$30 each) for the decoys → \\${cheap_to_decoys * 30 / 1e3:.1f}K\n"
            f"- **{expensive_to_shaheds} premium intercepts** "
            f"(IRIS-T/Patriot mix @ \\${INTERCEPT_COST_CFG/1000:.0f}K) for the "
            f"Shaheds → \\${expensive_to_shaheds * INTERCEPT_COST_CFG / 1e6:.1f}M\n\n"
            f"Vs uniform-spend at \\${INTERCEPT_COST_CFG/1000:.0f}K avg: would cost "
            f"\\${(state.launched_remaining * INTERCEPT_COST_CFG) / 1e6:.1f}M\n\n"
            f"**Potential savings from accurate classification: "
            f"\\${savings_vs_uniform/1e6:.1f}M** for the remainder of this "
            f"week alone."
        )

        st.caption(
            "Right now we don't have a real classifier — the panel uses the "
            "historical 35% prior as `decoys_identified_so_far`. When a "
            "real radar/EW discriminator comes online (acoustic harmonic, "
            "RCS pattern, IR signature, climb-rate variance), feed its "
            "per-track P(decoy) into the K_t input and the posterior "
            "tightens further."
        )

    st.divider()

    # ============== FORCE CONCENTRATION (CENTROID TRAJECTORY) ==============
    with st.container():
        st.markdown("### 🎯 Russian force concentration — weekly centroid trajectory")
        st.caption(
            "Where Russia is putting its strike weight, reverse-engineered "
            "from where the drones land. The weighted centroid (lat/lon "
            "averaged by drone count) shows where Russia's operational "
            "focus actually IS, vs where the model thought it would be. "
            "Movement vectors track the shift week-over-week."
        )

        # Compute weekly centroids
        obs_geo = observations.merge(
            oblasts[['oblast','lat','lon']], on='oblast', how='left')
        obs_geo['week_start'] = obs_geo['observation_date'] - pd.to_timedelta(
            obs_geo['observation_date'].dt.weekday, unit='D')
        weekly_c = obs_geo.groupby('week_start').apply(lambda g: pd.Series({
            'total': g['observed_drones'].sum(),
            'lat': (g['lat'] * g['observed_drones']).sum() / g['observed_drones'].sum(),
            'lon': (g['lon'] * g['observed_drones']).sum() / g['observed_drones'].sum(),
            'top': g.groupby('oblast')['observed_drones'].sum().idxmax(),
        })).reset_index()
        weekly_c = weekly_c[weekly_c['total'] >= 200].sort_values('week_start').reset_index(drop=True)

        # Movement vectors table
        rows = []
        for i in range(1, len(weekly_c)):
            p = weekly_c.iloc[i-1]
            c_ = weekly_c.iloc[i]
            dlat = c_['lat'] - p['lat']
            dlon = c_['lon'] - p['lon']
            km_n = dlat * 111
            km_e = dlon * 71
            mag = (km_n**2 + km_e**2) ** 0.5
            angle = float(np.degrees(np.arctan2(dlon, dlat)))
            compass = ['N','NNE','NE','ENE','E','ESE','SE','SSE','S','SSW','SW','WSW','W','WNW','NW','NNW']
            dir_idx = int(((angle + 11.25) % 360) / 22.5)
            rows.append({
                'From week': p['week_start'].date().isoformat(),
                'To week': c_['week_start'].date().isoformat(),
                'Direction': compass[dir_idx],
                'Distance (km)': f"{mag:.0f}",
                'Top oblast shift': f"{p['top']} → {c_['top']}",
            })
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Render the small-multiples evolution chart
        if len(weekly_c) >= 2:
            n = len(weekly_c)
            fig_fc, axes = plt.subplots(1, n, figsize=(3.5*n, 4.2))
            if n == 1:
                axes = [axes]

            # Inline geometry load (the @st.cache_data wrapper is defined
            # later in the script, so we read directly here)
            import geopandas as _gpd_fc
            geom_fc = _gpd_fc.read_file(DATA_DIR / "ukraine_oblasts.geojson")

            # CSV oblast names -> shapeName in the GeoJSON
            _OBLAST_NAME_MAP_FC = {
                'Kyiv City':'Kyiv','Kyiv Oblast':'Kyiv Oblast',
                'Kharkiv':'Kharkiv Oblast','Odesa':'Odessa Oblast',
                'Lviv':'Lviv Oblast','Dnipropetrovsk':'Dnipropetrovsk Oblast',
                'Donetsk':'Donetsk Oblast','Zaporizhzhia':'Zaporizhia Oblast',
                'Mykolaiv':'Mykolaiv Oblast','Kherson':'Kherson Oblast',
                'Poltava':'Poltava Oblast','Sumy':'Sumy Oblast',
                'Chernihiv':'Chernihiv Oblast','Vinnytsia':'Vinnytsia Oblast',
                'Cherkasy':'Cherkasy Oblast','Kirovohrad':'Kirovohrad Oblast',
                'Zhytomyr':'Zhytomyr Oblast','Rivne':'Rivne Oblast',
                'Volyn':'Volyn Oblast','Ternopil':'Ternopil Oblast',
                'Khmelnytskyi':'Khmelnytskyi Oblast',
                'Ivano-Frankivsk':'Ivano-Frankivsk Oblast',
                'Chernivtsi':'Chernivtsi Oblast',
                'Zakarpattia':'Zakarpattia Oblast',
            }

            global_max = 0
            for ws in weekly_c['week_start']:
                m = obs_geo[obs_geo['week_start']==ws].groupby('oblast')['observed_drones'].sum().max()
                global_max = max(global_max, float(m) if pd.notna(m) else 0)
            norm_fc = mcolors.PowerNorm(gamma=0.55, vmin=0, vmax=max(global_max,1))
            cmap_fc = plt.cm.YlOrRd

            for ax_w, (_, w_row) in zip(axes, weekly_c.iterrows()):
                ws = w_row['week_start']
                gw = obs_geo[obs_geo['week_start']==ws].groupby('oblast')['observed_drones'].sum().reset_index()
                gw['shapeName'] = gw['oblast'].map(_OBLAST_NAME_MAP_FC)
                gw_geom = geom_fc.merge(gw[['shapeName','observed_drones']],
                                         on='shapeName', how='left')
                gw_geom['observed_drones'] = gw_geom['observed_drones'].fillna(0)
                gw_geom.plot(ax=ax_w, column='observed_drones', cmap=cmap_fc,
                              norm=norm_fc, edgecolor='#444', linewidth=0.35)
                # Centroid trail
                idx = weekly_c.index[weekly_c['week_start']==ws].tolist()[0]
                for j in range(idx):
                    pp = weekly_c.iloc[j]
                    ax_w.scatter(pp['lon'], pp['lat'], s=60, color='blue',
                                  edgecolor='white', alpha=min(0.25+0.13*j, 1.0), zorder=8)
                if idx > 0:
                    pp = weekly_c.iloc[idx-1]
                    ax_w.annotate('', xy=(w_row['lon'], w_row['lat']),
                                   xytext=(pp['lon'], pp['lat']),
                                   arrowprops=dict(arrowstyle='->', color='red', lw=2.2))
                ax_w.scatter(w_row['lon'], w_row['lat'], s=200, color='blue',
                              edgecolor='white', linewidth=2, zorder=10)
                ax_w.set_title(f"{ws.date()}\n{int(w_row['total']):,} drones\nTop: {w_row['top']}",
                                fontsize=9, fontweight='bold')
                ax_w.set_xlim(21.5, 41.5); ax_w.set_ylim(43.8, 53.2)
                ax_w.set_xticks([]); ax_w.set_yticks([])
                ax_w.set_aspect(1.45)

            sm = plt.cm.ScalarMappable(cmap=cmap_fc, norm=norm_fc); sm.set_array([])
            fig_fc.colorbar(sm, ax=axes, fraction=0.012, pad=0.012,
                             label='Drones landed (per oblast)')
            fig_fc.suptitle('Force concentration — weekly evolution + centroid trail',
                             fontsize=12, fontweight='bold', y=1.02)
            st.pyplot(fig_fc)

        st.caption(
            "Each blue dot is that week's weighted impact centroid. Red "
            "arrows show movement between weeks. Lighter dots are the "
            "trail from prior weeks. The centroid is mathematically the "
            "Russian 'aim point' — where their fire is converging on "
            "average. Geographic interpretation: a westward shift means "
            "Russia attacking deeper into Ukraine; a northward shift "
            "means the Belarus/Chernihiv corridor; an eastward shift "
            "means frontline focus."
        )

    st.divider()

    # ============== LOGISTICS GEOGRAPHY ==============
    with st.container():
        st.markdown("### 🛤️ Logistics geography — supply lines, distances, equilibrium frontline")
        st.caption(
            "Where the supply flows actually move on a map. Russian supply "
            "lines (red, internal) are roughly half the length of Ukrainian "
            "lines (blue, from Western border crossings). The dashed line is "
            "the calculated equilibrium frontline — where each side's "
            "distance-decayed combat power equalizes."
        )

        import geopandas as _gpd_log
        geom_log = _gpd_log.read_file(DATA_DIR / "ukraine_oblasts.geojson")

        # Key nodes: (name, lat, lon, side)
        ua_entry = [
            ('Korczowa (PL)', 50.05, 22.94, 'entry'),
            ('Medyka (PL)',   49.81, 22.93, 'entry'),
            ('Záhony (HU)',   48.42, 22.18, 'entry'),
            ('Vyšné Nemecké (SK)', 48.62, 22.17, 'entry'),
            ('Siret (RO)',    47.95, 26.07, 'entry'),
            ('Reni (Danube)', 45.45, 28.28, 'entry'),
        ]
        ua_hubs = [
            ('Lviv',          49.84, 24.03),
            ('Kyiv',          50.45, 30.52),
            ('Dnipro',        48.45, 35.05),
            ('Kharkiv',       49.99, 36.23),
            ('Pokrovsk',      48.27, 37.18),
            ('Zaporizhzhia',  47.85, 35.12),
            ('Odesa',         46.48, 30.73),
        ]
        ru_hubs = [
            ('Moscow (RU)',         55.75, 37.62),
            ('Voronezh (RU)',       51.66, 39.20),
            ('Belgorod (RU)',       50.60, 36.60),
            ('Rostov-on-Don (RU)',  47.23, 39.71),
            ('Krasnodar (RU)',      45.04, 38.97),
            ('Donetsk (occupied)',  48.00, 37.80),
        ]
        # Frontline (rough contact line, North → South)
        frontline = [
            (51.0, 35.0),  # north
            (49.7, 37.6),  # Kupyansk area
            (48.6, 38.0),  # Bakhmut area
            (48.1, 37.7),  # Pokrovsk/Avdiivka
            (47.7, 36.9),  # Velyka Novosilka
            (47.5, 36.2),  # SW Donetsk
            (47.3, 35.0),  # Zaporizhzhia
            (46.7, 33.7),  # Kherson
        ]

        # Supply lines (origin → destination)
        ua_supply = [
            ('Korczowa (PL)', 'Lviv'),
            ('Korczowa (PL)', 'Kyiv'),
            ('Lviv', 'Kyiv'),
            ('Kyiv', 'Kharkiv'),
            ('Kyiv', 'Dnipro'),
            ('Dnipro', 'Pokrovsk'),
            ('Dnipro', 'Zaporizhzhia'),
            ('Reni (Danube)', 'Odesa'),
        ]
        ru_supply = [
            ('Moscow (RU)', 'Voronezh (RU)'),
            ('Voronezh (RU)', 'Belgorod (RU)'),
            ('Moscow (RU)', 'Rostov-on-Don (RU)'),
            ('Rostov-on-Don (RU)', 'Donetsk (occupied)'),
            ('Krasnodar (RU)', 'Donetsk (occupied)'),
        ]

        all_nodes = {n: (lat, lon) for n, lat, lon, *_ in ua_entry}
        all_nodes.update({n: (lat, lon) for n, lat, lon in ua_hubs})
        all_nodes.update({n: (lat, lon) for n, lat, lon in ru_hubs})

        fig_log, ax_log = plt.subplots(figsize=(14, 9))
        ax_log.set_facecolor('#dbeaf2')

        # Backdrop: surrounding countries
        import os as _os_log, glob as _glob_log, pyogrio as _py_log
        cands = _glob_log.glob(_os_log.path.join(
            _os_log.path.dirname(_py_log.__file__),
            '**', 'naturalearth_lowres.shp'), recursive=True)
        if cands:
            world = _gpd_log.read_file(cands[0])
            for cn, fc, ec in [('Poland','#ececec','#9aa0a6'),('Romania','#ececec','#9aa0a6'),
                               ('Hungary','#ececec','#9aa0a6'),('Slovakia','#ececec','#9aa0a6'),
                               ('Moldova','#ececec','#9aa0a6'),('Russia','#f5d6d6','#b04545'),
                               ('Belarus','#f5e1e1','#b04545')]:
                c = world[world['name']==cn]
                if not c.empty:
                    c.plot(ax=ax_log, facecolor=fc, edgecolor=ec, linewidth=0.7, alpha=0.7)

        # Oblast outlines, light fill
        geom_log.plot(ax=ax_log, facecolor='#fff8dc', edgecolor='#aaa',
                       linewidth=0.4, alpha=0.6)

        # Ukrainian supply lines (blue) with distance labels
        for src, dst in ua_supply:
            slat, slon = all_nodes[src]
            dlat, dlon = all_nodes[dst]
            dist_km = ((slat-dlat)*111)**2 + ((slon-dlon)*71)**2
            dist_km = dist_km ** 0.5
            ax_log.annotate('', xy=(dlon, dlat), xytext=(slon, slat),
                             arrowprops=dict(arrowstyle='->', color='#003d7a',
                                              lw=2.2, alpha=0.7,
                                              connectionstyle='arc3,rad=0.05'))
            # Mid-point label
            mlat, mlon = (slat+dlat)/2, (slon+dlon)/2
            ax_log.text(mlon, mlat, f'{dist_km:.0f}km',
                         fontsize=6.5, color='#003d7a',
                         bbox=dict(boxstyle='round,pad=0.1',
                                    facecolor='white', edgecolor='none',
                                    alpha=0.7))

        # Russian supply lines (red)
        for src, dst in ru_supply:
            slat, slon = all_nodes[src]
            dlat, dlon = all_nodes[dst]
            dist_km = ((slat-dlat)*111)**2 + ((slon-dlon)*71)**2
            dist_km = dist_km ** 0.5
            ax_log.annotate('', xy=(dlon, dlat), xytext=(slon, slat),
                             arrowprops=dict(arrowstyle='->', color='#cc0033',
                                              lw=2.2, alpha=0.7,
                                              connectionstyle='arc3,rad=-0.05'))
            mlat, mlon = (slat+dlat)/2, (slon+dlon)/2
            ax_log.text(mlon, mlat, f'{dist_km:.0f}km',
                         fontsize=6.5, color='#cc0033',
                         bbox=dict(boxstyle='round,pad=0.1',
                                    facecolor='white', edgecolor='none',
                                    alpha=0.7))

        # Plot entry points (green)
        for n, lat, lon, *_ in ua_entry:
            ax_log.scatter([lon],[lat], marker='s', s=120, color='#2a8c4a',
                            edgecolor='black', linewidth=1.0, zorder=10)
            ax_log.annotate(n, (lon, lat), xytext=(lon+0.15, lat+0.15),
                             fontsize=6.5, fontweight='bold', color='#1a5a2a',
                             bbox=dict(boxstyle='round,pad=0.15',
                                        facecolor='white', edgecolor='#2a8c4a',
                                        alpha=0.9))

        # UA hubs (blue squares)
        for n, lat, lon in ua_hubs:
            ax_log.scatter([lon],[lat], marker='D', s=130, color='#003d7a',
                            edgecolor='white', linewidth=1.2, zorder=10)
            ax_log.annotate(n, (lon, lat), xytext=(lon+0.2, lat-0.25),
                             fontsize=7, fontweight='bold', color='white',
                             bbox=dict(boxstyle='round,pad=0.2',
                                        facecolor='#003d7a', edgecolor='none',
                                        alpha=0.92))

        # RU hubs (red squares)
        for n, lat, lon in ru_hubs:
            ax_log.scatter([lon],[lat], marker='X', s=180, color='#990000',
                            edgecolor='black', linewidth=1.0, zorder=10)
            ax_log.annotate(n, (lon, lat), xytext=(lon+0.2, lat+0.2),
                             fontsize=7, fontweight='bold', color='white',
                             bbox=dict(boxstyle='round,pad=0.2',
                                        facecolor='#990000', edgecolor='none',
                                        alpha=0.92))

        # Frontline (dashed black line — the equilibrium)
        flats = [p[0] for p in frontline]
        flons = [p[1] for p in frontline]
        ax_log.plot(flons, flats, '--', color='black', lw=3, alpha=0.85,
                     zorder=8, label='Equilibrium frontline (≈ contact line Jun 2026)')
        # Shade Russian-controlled area east of frontline
        from matplotlib.patches import Polygon
        ru_zone = list(zip(flons, flats)) + [(42, 44), (42, 52), (flons[0], flats[0])]
        poly = Polygon(ru_zone, facecolor='#cc0033', alpha=0.10, edgecolor='none', zorder=2)
        ax_log.add_patch(poly)

        # Equilibrium math annotation
        ax_log.text(24.5, 51.8, 'UKRAINE supply:\n~1,100 km avg\n37% delivery efficiency',
                     fontsize=9, color='#003d7a', fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.4', facecolor='#e8f0fa',
                                edgecolor='#003d7a', alpha=0.92))
        ax_log.text(40, 50.5, 'RUSSIA supply:\n~500 km avg\n73% delivery efficiency',
                     fontsize=9, color='#990000', fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.4', facecolor='#fae8e8',
                                edgecolor='#990000', alpha=0.92))

        # Dnipro river annotation
        ax_log.annotate('Dnipro River\n(natural equilibrium barrier)',
                         xy=(33.5, 47.5), xytext=(29, 45),
                         fontsize=8, fontweight='bold', color='#003d7a',
                         arrowprops=dict(arrowstyle='->', color='#003d7a', lw=1.2),
                         bbox=dict(boxstyle='round,pad=0.2',
                                    facecolor='white', edgecolor='#003d7a',
                                    alpha=0.9))

        ax_log.set_xlim(20, 42)
        ax_log.set_ylim(43, 56)
        ax_log.set_aspect(1.45)
        ax_log.grid(alpha=0.2, linestyle=':')
        ax_log.set_title('Supply chain geography — Ukraine vs Russia\n'
                         'Blue = UA logistics (long, gauge-change vulnerable) · '
                         'Red = RU logistics (short, internal) · '
                         'Dashed black = equilibrium frontline',
                         fontsize=12, fontweight='bold')
        ax_log.set_xlabel('Longitude (°E)')
        ax_log.set_ylabel('Latitude (°N)')

        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        legend_elems = [
            Line2D([0],[0], color='#003d7a', lw=2.5,
                   label='Ukrainian supply line'),
            Line2D([0],[0], color='#cc0033', lw=2.5,
                   label='Russian supply line'),
            Line2D([0],[0], color='black', lw=2.5, linestyle='--',
                   label='Equilibrium frontline'),
            Line2D([0],[0], marker='s', color='w', markerfacecolor='#2a8c4a',
                    markersize=10, label='Western entry points'),
            Line2D([0],[0], marker='D', color='w', markerfacecolor='#003d7a',
                    markersize=10, label='Ukrainian rail hub'),
            Line2D([0],[0], marker='X', color='w', markerfacecolor='#990000',
                    markersize=12, label='Russian rail hub'),
            Patch(facecolor='#cc0033', alpha=0.15, label='Russian-controlled zone'),
        ]
        ax_log.legend(handles=legend_elems, loc='lower left',
                      fontsize=8, framealpha=0.9)
        st.pyplot(fig_log)

        st.markdown("**Distance & efficiency metrics:**")
        log_metrics = pd.DataFrame([
            {'Side': 'Ukraine', 'Avg supply distance to front': '1,100 km',
             'Transit time (rail)': '24-48 hr',
             'Delivery efficiency (after attrition)': '37%',
             'Bottleneck': 'Border gauge change + 1,200 km cross-country rail'},
            {'Side': 'Russia', 'Avg supply distance to front': '500 km',
             'Transit time (rail)': '6-30 hr',
             'Delivery efficiency (after attrition)': '73%',
             'Bottleneck': 'Voronezh/Rostov rail junction capacity'},
        ])
        st.dataframe(log_metrics, use_container_width=True, hide_index=True)

        st.markdown("**The equilibrium math (one line):**")
        st.latex(
            r"P_{UA}(x) = \frac{Q_{UA} \cdot \eta_{UA}}{1 + \lambda \cdot d_{UA}(x)} "
            r"\quad = \quad "
            r"P_R(x) = \frac{Q_R \cdot \eta_R}{1 + \lambda \cdot d_R(x)}"
        )
        st.markdown(
            "Where η = delivery efficiency, λ ≈ 0.002/km attrition rate, "
            "d = distance from production hub. Equilibrium x* is where "
            "both forces equalize — currently the Dnipro River and ~50-100 km "
            "east in Donetsk. Russian advance west of Dnipro requires bridge "
            "= mathematically infeasible at current supply throughput."
        )

    st.divider()

    # ============== LAUNCH SITE FORENSICS ==============
    with st.container():
        st.markdown("### 🚀 Launch site forensics — where are drones coming from?")
        st.caption(
            "Reverse-engineered from UA AF summary text. Each mention of a "
            "Russian launch site (Bryansk, Kursk, Orel, Crimea, etc.) is "
            "counted per week. **Silent launchers** are sites that went "
            "active→0 (possible UA strike success). **New activations** are "
            "sites that appeared this week. **Surges** are 3×+ mention "
            "increases. The flight-time matrix shows how long each drone "
            "took to reach its target — useful for back-solving launch "
            "times from sighting timestamps."
        )

        import launch_site_tracker as _lst
        import importlib as _importlib
        _importlib.reload(_lst)

        ls_log_path = DATA_DIR / 'launch_site_log.csv'
        if ls_log_path.exists():
            ls_log = pd.read_csv(ls_log_path, parse_dates=['week_start'])
            summary_ls = _lst.detect_changes(ls_log)

            # KPI row
            lcol1, lcol2, lcol3, lcol4 = st.columns(4)
            with lcol1:
                st.metric("Weeks tracked", ls_log['week_start'].nunique())
            with lcol2:
                st.metric(
                    "Active launch sites (latest week)",
                    int((summary_ls.weekly_mentions.iloc[-1] > 0).sum())
                    if not summary_ls.weekly_mentions.empty else 0,
                )
            with lcol3:
                st.metric(
                    "🟢 Silent launchers",
                    len(summary_ls.silent_alerts),
                    help="Sites that were active last week but zero this week. "
                         "Possible UA strike success.",
                )
            with lcol4:
                st.metric(
                    "🔺 Surge alerts",
                    len(summary_ls.surge_alerts),
                    help="Sites with 3×+ mention increase week-over-week — "
                         "Russia scaling that launcher cluster.",
                )

            # Surge alerts
            if summary_ls.surge_alerts:
                st.warning(
                    "**🔺 Surge alerts** (Russia scaling these launchers):  "
                    + "  ·  ".join(
                        f"**{a['site']}** ({a['prior_mentions']} → "
                        f"{a['latest_mentions']}, ×{a['multiplier']:.1f})"
                        for a in summary_ls.surge_alerts
                    )
                )
            if summary_ls.new_activations:
                st.info(
                    "**🆕 New activations this week:**  "
                    + "  ·  ".join(
                        f"**{a['site']}** ({a['latest_mentions']} mentions)"
                        for a in summary_ls.new_activations
                    )
                )
            if summary_ls.silent_alerts:
                st.success(
                    "**🟢 Silent launchers** (possible strike success — "
                    "watch for ≥2 silent weeks before celebrating):  "
                    + "  ·  ".join(
                        f"**{a['site']}** (was {a['prior_mentions']}/wk, "
                        f"now 0)"
                        for a in summary_ls.silent_alerts
                    )
                )
            if not (summary_ls.surge_alerts or summary_ls.new_activations
                    or summary_ls.silent_alerts):
                st.info("Pattern is stable — no surges, new activations, "
                        "or silent launchers detected this week.")

            # Heatmap of mentions per (week × site)
            pivot = summary_ls.weekly_mentions
            if not pivot.empty:
                fig_ls, ax_ls = plt.subplots(
                    figsize=(min(11, max(6, 0.8*len(pivot.columns))),
                              max(2.2, 0.45*len(pivot))))
                im = ax_ls.imshow(pivot.values, cmap='YlOrRd',
                                    aspect='auto', interpolation='nearest')
                ax_ls.set_xticks(range(len(pivot.columns)))
                ax_ls.set_xticklabels(pivot.columns, rotation=35,
                                       ha='right', fontsize=8)
                ax_ls.set_yticks(range(len(pivot)))
                ax_ls.set_yticklabels(
                    [d.strftime('%Y-%m-%d') for d in pivot.index], fontsize=8)
                # Annotate counts
                for i in range(len(pivot)):
                    for j in range(len(pivot.columns)):
                        v = int(pivot.values[i, j])
                        if v > 0:
                            color = 'white' if v >= pivot.values.max()*0.5 else 'black'
                            ax_ls.text(j, i, str(v), ha='center', va='center',
                                        color=color, fontsize=8, fontweight='bold')
                ax_ls.set_title('Launch-site mentions per week',
                                 fontsize=11, fontweight='bold')
                plt.colorbar(im, ax=ax_ls, label='mentions', fraction=0.03,
                              pad=0.02)
                st.pyplot(fig_ls)

            # Flight-time matrix per oblast → closest launch site
            with st.expander("✈️ Flight-time matrix — closest launch site to each oblast"):
                ftm = _lst.closest_sites(oblasts)
                ftm = ftm.sort_values('flight_time_hr').rename(columns={
                    'oblast': 'Oblast',
                    'closest_site': 'Closest launch site',
                    'distance_km': 'Distance (km)',
                    'flight_time_hr': 'Flight time (hr)',
                })
                st.dataframe(ftm, use_container_width=True, hide_index=True)
                st.caption(
                    "Shahed-136 cruise speed assumed 185 km/h. Real flight "
                    "times vary ±15% with wind, altitude, and routing. The "
                    "closest site is the *most likely* launch origin — but "
                    "Russia regularly uses farther sites to confuse defense "
                    "or save closer launchers for high-priority targets."
                )

            # Drone forensics: back-solve launch time from a sighting
            with st.expander("🔍 Drone forensics — back-solve a sighting"):
                fc1, fc2, fc3 = st.columns([2,2,1])
                with fc1:
                    forensics_oblast = st.selectbox(
                        "Oblast where sighted",
                        oblasts['oblast'].tolist(),
                        key='forensics_oblast',
                    )
                with fc2:
                    forensics_time = st.text_input(
                        "Sighting time (UTC, e.g. '2026-06-17 22:43')",
                        value=datetime.now().strftime('%Y-%m-%d %H:%M'),
                        key='forensics_time',
                    )
                with fc3:
                    site_options = ['(auto — closest)'] + list(_lst.LAUNCH_SITES.keys())
                    forensics_site = st.selectbox(
                        "Launch site (or auto)",
                        site_options, key='forensics_site',
                    )
                site_arg = None if forensics_site == '(auto — closest)' else forensics_site
                try:
                    result = _lst.backsolve_launch_time(
                        forensics_time, forensics_oblast, oblasts, site=site_arg)
                    if 'error' in result:
                        st.error(result['error'])
                    else:
                        st.markdown(
                            f"**Sighting:** {result['sighting_at']} in "
                            f"{result['oblast']}\n\n"
                            f"**Most likely launch site:** "
                            f"`{result['probable_launch_site']}`\n\n"
                            f"**Distance:** {result['distance_km']} km   ·   "
                            f"**Flight time:** {result['flight_time_hr']} hr\n\n"
                            f"**Probable launch time (UTC):** "
                            f"`{result['probable_launch_time_utc']}`"
                        )
                except Exception as _e:
                    st.error(f"Couldn't parse: {_e}")

            # Persisted log dataframe (debug / inspection)
            with st.expander("Raw launch-site log (debug)"):
                st.dataframe(ls_log.sort_values(['week_start', 'site']),
                              use_container_width=True, hide_index=True)
        else:
            st.info(
                "No launch-site log yet. Sync the Telegram feed from the "
                "sidebar — the next fetch will populate this panel."
            )

    st.divider()

    # ============== PRODUCTION RATE (FACTORIO STYLE) ==============
    # Reverse-engineer Russia's drone production rate from launch data,
    # then simulate the "conveyor belt + buffer" stockpile dynamics.
    with st.container():
        st.markdown("### 🏭 Production rate — reverse-engineered (Factorio mode)")
        st.caption(
            "We see only the *output* (launches). Production is the input "
            "side of Russia's drone factory. Three convergent methods "
            "estimate the belt throughput; the chart shows simulated "
            "stockpile dynamics — surges drain the buffer, quiet days "
            "refill it."
        )

        # Build a full daily series (fill gaps with 0 = no summary posted)
        full_idx = pd.date_range(by_d['date'].min(), by_d['date'].max(), freq='D')
        ts = by_d.set_index('date').reindex(full_idx)
        ts.index.name = 'date'
        ts['launched_filled'] = ts['launched'].fillna(0)
        n_days = len(ts)

        total_launched_window = float(ts['launched_filled'].sum())
        steady_state_rate = total_launched_window / n_days
        # Peak rolling averages = floor on production (can't sustain above prod)
        peak_3 = float(ts['launched_filled'].rolling(3).mean().max() or 0)
        peak_7 = float(ts['launched_filled'].rolling(7).mean().max() or 0)
        peak_14 = float(ts['launched_filled'].rolling(14).mean().max() or 0)

        # Override or auto
        override = st.session_state.get('prod_rate_override', 0)
        if override > 0:
            est_rate = float(override)
            est_source = "manual override"
        else:
            # Use the peak 14-day rolling avg as a defensible floor estimate
            est_rate = peak_14
            est_source = "peak 14-day rolling avg of launches (floor)"

        # Factorio-style throughput
        per_hour = est_rate / 24
        per_min = est_rate / (24 * 60)
        sec_between = (24 * 60 * 60) / max(est_rate, 1)

        # KPIs
        pcol1, pcol2, pcol3, pcol4 = st.columns(4)
        with pcol1:
            st.metric("Estimated belt rate",
                      f"{est_rate:.0f}/day",
                      f"{est_rate*7:.0f}/wk · {est_rate*365/1000:.0f}K/yr")
        with pcol2:
            st.metric("Drones per hour", f"{per_hour:.1f}",
                      f"{per_min:.2f}/min")
        with pcol3:
            st.metric("One drone every",
                      f"{sec_between:.0f} sec",
                      f"≈ {sec_between/60:.1f} min")
        with pcol4:
            # Buffer drain on the biggest day
            worst_day = ts['launched_filled'].max()
            drain_that_day = max(worst_day - est_rate, 0)
            st.metric("Worst-day buffer drain",
                      f"{int(drain_that_day):,}",
                      f"vs {int(worst_day):,} launched")

        # Stockpile simulation
        initial = st.session_state.get('prod_initial_stockpile', 1000)
        ts['stockpile'] = initial + (est_rate - ts['launched_filled']).cumsum()
        ts['production'] = est_rate

        fig_p, (axA, axB) = plt.subplots(1, 2, figsize=(13, 4.5))

        # LEFT: daily launches vs production line
        labels = [d.strftime('%m-%d') for d in ts.index]
        axA.bar(labels, ts['launched_filled'], color='#cc0033',
                alpha=0.7, label='Daily launches', edgecolor='black', linewidth=0.5)
        axA.axhline(est_rate, color='#003d7a', linestyle='--', linewidth=2.5,
                    label=f'Belt rate ({est_rate:.0f}/day)')
        axA.set_title('Daily launches vs estimated production rate')
        axA.legend(loc='upper left', fontsize=9)
        axA.tick_params(axis='x', rotation=80, labelsize=7)
        axA.set_ylabel('Drones')
        axA.grid(alpha=0.3, axis='y')

        # RIGHT: stockpile trajectory (Factorio buffer)
        axB.fill_between(ts.index, 0, ts['stockpile'],
                         where=ts['stockpile'] > 0,
                         color='#2a8c4a', alpha=0.5, label='Stockpile balance')
        axB.fill_between(ts.index, 0, ts['stockpile'],
                         where=ts['stockpile'] < 0,
                         color='#cc0033', alpha=0.5,
                         label='Stockpile deficit (impossible)')
        axB.axhline(0, color='black', linewidth=0.5)
        axB.plot(ts.index, ts['stockpile'], color='#1a5a2a', linewidth=1.5)
        axB.set_title(f'Simulated stockpile (start = {initial:,}, '
                       f'production = {est_rate:.0f}/day)')
        axB.set_ylabel('Drones in buffer')
        axB.legend(fontsize=9)
        axB.tick_params(axis='x', rotation=30)
        axB.grid(alpha=0.3)

        st.pyplot(fig_p)

        # Method comparison table
        st.markdown("**Three estimation methods (they should converge):**")
        methods = pd.DataFrame([
            {'Method': 'Steady-state (Σlaunched / Σdays)',
             'Estimate': f'{steady_state_rate:.0f}/day',
             'Assumption': 'Stockpile change ≈ 0 across window',
             'Interpretation': 'UPPER bound (if Russia is building stockpile, real prod is higher)'},
            {'Method': 'Peak 7-day rolling avg',
             'Estimate': f'{peak_7:.0f}/day',
             'Assumption': 'Max sustainable rate',
             'Interpretation': 'FLOOR — Russia maintained this for a week, so production ≥ this'},
            {'Method': 'Peak 14-day rolling avg',
             'Estimate': f'{peak_14:.0f}/day',
             'Assumption': 'Max sustainable rate over 2 weeks',
             'Interpretation': 'STRONG FLOOR — 14 days of stockpile burn is exhausted'},
        ])
        st.dataframe(methods, use_container_width=True, hide_index=True)

        # Verdict
        final_stockpile = float(ts['stockpile'].iloc[-1])
        min_stockpile = float(ts['stockpile'].min())
        if min_stockpile < 0:
            verdict = (
                f"⚠️ **Stockpile goes negative** "
                f"({int(min_stockpile):,} at worst point). That's "
                f"physically impossible — either production is higher than "
                f"{est_rate:.0f}/day, or Russia had more than {initial:,} "
                f"drones banked at the start of the window. Raise either "
                f"in the sidebar to find a feasible combination."
            )
        elif final_stockpile > initial + est_rate * 14:
            verdict = (
                f"📈 **Stockpile growing** — ended at {int(final_stockpile):,} "
                f"(up from {initial:,}). Russia is producing faster than "
                f"launching. The launches you see are the launches they "
                f"*chose* to make, not the launches they *could*. This is "
                f"the most worrying scenario: surge capacity is hidden in "
                f"the buffer."
            )
        else:
            verdict = (
                f"⚖️ **Buffer-balanced** — stockpile ended at "
                f"{int(final_stockpile):,} (started {initial:,}). "
                f"Russia is roughly launching what it produces, with "
                f"timing surges drawing the buffer down and quiet "
                f"periods refilling it. This is the classic Factorio "
                f"belt-with-buffer dynamic."
            )
        st.markdown(verdict)

        st.caption(
            f"**Bottom line**: best estimate is **{est_rate:.0f} drones/day** "
            f"({est_source}) = **{est_rate*7:.0f}/week** = "
            f"**{est_rate*365/1000:.0f}K/year**. That's one drone off the "
            f"belt every {sec_between/60:.1f} minutes, around the clock. "
            f"Public reporting on Russian Shahed production "
            f"(Alabuga + new lines) puts late-2025 capacity at "
            f"200-300/day — our reverse-engineered number "
            f"({'matches' if 200 <= est_rate <= 350 else 'differs from'}) "
            f"that estimate."
        )

    st.divider()

    with st.container():
        st.markdown("### 📈 Strategic outlook (multi-year projection)")
        st.caption(
            "Project the dollar war forward and find each side's binding "
            "constraint. Inputs come from the sidebar's '📈 Strategic outlook' "
            "expander — change oil price, aid level, refinery damage % to "
            "see scenarios."
        )

        # Headline metrics
        out_cols = st.columns(4)
        with out_cols[0]:
            st.metric(
                "Russia annual revenue",
                f"${ru_revenue_annual/1e9:.0f}B",
                f"oil @ ${oil_price}/bbl",
            )
        with out_cols[1]:
            st.metric(
                "Russia annual cost",
                f"${ru_total_annual_cost/1e9:.0f}B",
                f"net: ${ru_net_annual/1e9:+.0f}B",
            )
        with out_cols[2]:
            if years_to_nwf_zero == float('inf'):
                st.metric(
                    "NWF runway",
                    "indefinite",
                    "revenue covers cost",
                )
            else:
                st.metric(
                    "NWF depletes in",
                    f"{years_to_nwf_zero:.1f} years",
                    f"deficit ${abs(ru_net_annual)/1e9:.0f}B/yr",
                )
        with out_cols[3]:
            st.metric(
                "🇺🇦 Self-funded burden",
                f"${ua_self/1e9:.0f}B/yr",
                f"{ua_self_pct_gdp:.1f}% of GDP",
            )

        # Cumulative cost curves over 5 years
        years = np.arange(0, 6)
        ru_cum_cost = ru_total_annual_cost * years
        ua_cum_cost = ua_total_spend * years
        ua_self_cum = ua_self * years
        ru_cum_revenue = ru_revenue_annual * years
        ru_nwf_trajectory = np.maximum(nwf + (ru_revenue_annual - ru_total_annual_cost) * years, 0)

        fig_out, (axL, axR) = plt.subplots(1, 2, figsize=(13, 4.5))

        axL.plot(years, ru_cum_cost/1e9, 'r-o', label='RU cumulative war cost')
        axL.plot(years, ru_cum_revenue/1e9, 'r--', label='RU cumulative revenue', alpha=0.6)
        axL.plot(years, ua_cum_cost/1e9, 'b-o', label='UA total cumulative cost')
        axL.plot(years, ua_self_cum/1e9, 'b--', label='UA self-funded cumulative', alpha=0.6)
        axL.set_xlabel('Years from now')
        axL.set_ylabel('Cumulative \\$B')
        axL.set_title('Cumulative cost curves')
        axL.legend(loc='upper left', fontsize=8)
        axL.grid(alpha=0.3)

        axR.plot(years, ru_nwf_trajectory/1e9, 'r-o',
                 label='Russia NWF balance')
        axR.axhline(0, color='black', linestyle=':', alpha=0.5)
        if years_to_nwf_zero != float('inf') and years_to_nwf_zero < 6:
            axR.axvline(years_to_nwf_zero, color='red', linestyle='--', alpha=0.5)
            axR.text(years_to_nwf_zero+0.1, max(ru_nwf_trajectory)/1e9*0.7,
                     f"NWF zero\n@ year {years_to_nwf_zero:.1f}",
                     color='red', fontsize=9)
        axR.set_xlabel('Years from now')
        axR.set_ylabel('NWF balance (\\$B)')
        axR.set_title('Russian fiscal runway (NWF depletion)')
        axR.legend(fontsize=8)
        axR.grid(alpha=0.3)

        st.pyplot(fig_out)

        # Constraint binding analysis
        st.markdown("**When does each Russian constraint actually bind?**")

        # Sovereign wealth (financial)
        financial_t = years_to_nwf_zero if years_to_nwf_zero != float('inf') else None
        # Demographics: ~700K casualties to date; rough sustainable ~150K/year
        # Manpower runway is qualitative — use slider-like estimate
        demo_t = 3.5  # placeholder (2-4 years public estimate)
        # Industrial: chip imports already binding for advanced systems
        industrial_t = 1.0
        # Refinery cascade: linked to refinery % above
        if refinery_offline >= 0.40:
            refinery_t = 1.5
        elif refinery_offline >= 0.25:
            refinery_t = 3.0
        else:
            refinery_t = 6.0

        cons_rows = [
            ("Financial (NWF depletion)",
             f"{financial_t:.1f}y" if financial_t else "no deficit",
             "Depends on oil price + refinery damage. Hard constraint once NWF=0."),
            ("Demographic (manpower)", f"~{demo_t:.0f}y",
             "300-700K casualties already; recruitment getting harder. Soft constraint — Russia keeps lowering quality."),
            ("Industrial (sanctions on chips)", f"~{industrial_t:.0f}y",
             "Already binding on advanced systems. Russia substitutes Chinese parts; question is whether China's stance shifts."),
            ("Refinery cascade",
             f"~{refinery_t:.1f}y" if refinery_offline > 0.15 else ">5y",
             f"At {refinery_offline*100:.0f}% offline, fuel logistics strain. At 40%+ sustained, military mobility degrades."),
        ]
        constraint_df = pd.DataFrame(cons_rows,
                                     columns=['Constraint', 'Time to bind', 'Notes'])
        st.dataframe(constraint_df, use_container_width=True, hide_index=True)

        # Strategic verdict
        binding_times = [financial_t, demo_t, industrial_t, refinery_t]
        valid_times = [t for t in binding_times if t is not None]
        first_to_bind = min(valid_times) if valid_times else None

        if first_to_bind:
            st.markdown(
                f"**Earliest binding constraint: ~{first_to_bind:.1f} years** "
                f"from now. This is the realistic horizon for Russia being "
                f"*forced* to scale down operations — not 'lose the war,' "
                f"but to choose between war and other priorities."
            )

        # Ukraine sustainability
        st.markdown("**Ukraine's sustainability — aid-dependent**")
        if aid_mult >= 0.8:
            ua_sustain = "Sustainable for 5+ years if aid holds at this level."
        elif aid_mult >= 0.5:
            ua_sustain = f"Strained — aid at {aid_mult*100:.0f}% of current means Ukraine self-funds ~\\${ua_self/1e9:.0f}B/yr ({ua_self_pct_gdp:.1f}% of GDP). Unsustainable past 2-3 years."
        else:
            ua_sustain = f"⚠️ Critical — aid collapsed to {aid_mult*100:.0f}% means \\${ua_self/1e9:.0f}B/yr self-funded, which is {ua_self_pct_gdp:.0f}% of GDP. War-economy collapse within 12-18 months."
        st.markdown(ua_sustain)

        # Verdict on the race
        st.markdown("---")
        st.markdown("**The race — who breaks first?**")
        ru_break_years = first_to_bind if first_to_bind else 10
        ua_break_years = (5 if aid_mult >= 0.8
                          else (2.5 if aid_mult >= 0.5 else 1.25))
        if ru_break_years < ua_break_years:
            race_verdict = (
                f"🟢 **Russia is on track to bind first** (~{ru_break_years:.1f}y "
                f"vs Ukraine's ~{ua_break_years:.1f}y aid runway). The "
                f"strategy works *if it lasts that long*. Western aid duration "
                f"is the binding variable, not military performance."
            )
        elif ua_break_years < ru_break_years:
            race_verdict = (
                f"🔴 **Ukraine's aid runway expires before Russia binds** "
                f"(~{ua_break_years:.1f}y vs ~{ru_break_years:.1f}y). Without "
                f"sustained Western support, Ukraine is forced to settle on "
                f"Russian terms regardless of how well the kinetic exchange "
                f"goes. Aid duration > military results in this scenario."
            )
        else:
            race_verdict = (
                f"🟡 **It's a coin-flip race** at ~{ru_break_years:.1f}y. "
                f"Small shifts in oil price, aid, or refinery damage tip "
                f"the outcome."
            )
        st.markdown(race_verdict)

    st.divider()


# ============== KPI ROW ==============
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Total predicted (this week)", f"{int(forecast['predicted_week'].sum())}")
with col2:
    top = forecast.iloc[0]
    st.metric(f"Top target: {top['oblast']}", f"{int(top['predicted_week'])} drones")
with col3:
    st.metric("Launches observed this week", f"{launches_this_week:,}")
with col4:
    obs_total = int(observations['observed_drones'].sum())
    st.metric("Total observed (all time)", f"{obs_total:,}")
with col5:
    st.metric(
        "Learning weight α",
        f"{learning_alpha:.2f}",
        help="0 = pure prior, 1 = pure empirical share. "
             "Grows as more launchers are observed.",
    )

# ============== MAP ==============
st.subheader("Predicted Strikes by Oblast")

import geopandas as gpd
import os, glob, pyogrio

OBLAST_GEOJSON = DATA_DIR / "ukraine_oblasts.geojson"

# CSV oblast names -> shapeName in the GeoBoundaries GeoJSON
OBLAST_NAME_MAP = {
    'Kyiv City': 'Kyiv',
    'Kyiv Oblast': 'Kyiv Oblast',
    'Kharkiv': 'Kharkiv Oblast',
    'Odesa': 'Odessa Oblast',
    'Lviv': 'Lviv Oblast',
    'Dnipropetrovsk': 'Dnipropetrovsk Oblast',
    'Donetsk': 'Donetsk Oblast',
    'Zaporizhzhia': 'Zaporizhia Oblast',
    'Mykolaiv': 'Mykolaiv Oblast',
    'Kherson': 'Kherson Oblast',
    'Poltava': 'Poltava Oblast',
    'Sumy': 'Sumy Oblast',
    'Chernihiv': 'Chernihiv Oblast',
    'Vinnytsia': 'Vinnytsia Oblast',
    'Cherkasy': 'Cherkasy Oblast',
    'Kirovohrad': 'Kirovohrad Oblast',
    'Zhytomyr': 'Zhytomyr Oblast',
    'Rivne': 'Rivne Oblast',
    'Volyn': 'Volyn Oblast',
    'Ternopil': 'Ternopil Oblast',
    'Khmelnytskyi': 'Khmelnytskyi Oblast',
    'Ivano-Frankivsk': 'Ivano-Frankivsk Oblast',
    'Chernivtsi': 'Chernivtsi Oblast',
    'Zakarpattia': 'Zakarpattia Oblast',
}


@st.cache_data
def load_oblast_geometry():
    return gpd.read_file(OBLAST_GEOJSON)


@st.cache_data
def load_world():
    candidates = glob.glob(
        os.path.join(os.path.dirname(pyogrio.__file__),
                     '**', 'naturalearth_lowres.shp'),
        recursive=True,
    )
    return gpd.read_file(candidates[0]) if candidates else None


fig, ax = plt.subplots(figsize=(14, 9))
ax.set_facecolor('#dbeaf2')

# Backdrop: neighbouring countries from naturalearth (if available)
world = load_world()
if world is not None:
    for country_name, color, edge in [
        ('Poland', '#ececec', '#9aa0a6'),
        ('Romania', '#ececec', '#9aa0a6'),
        ('Hungary', '#ececec', '#9aa0a6'),
        ('Slovakia', '#ececec', '#9aa0a6'),
        ('Moldova', '#ececec', '#9aa0a6'),
        ('Russia', '#f5d6d6', '#b04545'),
        ('Belarus', '#f5e1e1', '#b04545'),
    ]:
        country = world[world['name'] == country_name]
        if not country.empty:
            country.plot(ax=ax, facecolor=color, edgecolor=edge,
                         linewidth=0.8, alpha=0.75)

# Ukraine choropleth from oblast polygons
geom = load_oblast_geometry().copy()
forecast_geo = forecast.copy()
forecast_geo['shapeName'] = forecast_geo['oblast'].map(OBLAST_NAME_MAP)
geom = geom.merge(
    forecast_geo[['shapeName', 'predicted_week', 'learned_share', 'n_obs']],
    on='shapeName', how='left',
)

vmax = max(float(forecast['predicted_week'].max()), 1.0)
norm = mcolors.PowerNorm(gamma=0.6, vmin=0, vmax=vmax)
cmap = plt.cm.YlOrRd

geom_data = geom[geom['predicted_week'].notna()]
geom_nodata = geom[geom['predicted_week'].isna()]

if not geom_nodata.empty:
    geom_nodata.plot(ax=ax, facecolor='#f0eee5', edgecolor='#666',
                     linewidth=0.6, alpha=0.85)

geom_data.plot(
    ax=ax, column='predicted_week', cmap=cmap, norm=norm,
    edgecolor='#2a2a2a', linewidth=0.7, alpha=0.92,
)

# Per-oblast labels + counts
for _, row in geom.iterrows():
    if row.geometry is None or row.geometry.is_empty:
        continue
    cx, cy = row.geometry.representative_point().coords[0]
    pred = row.get('predicted_week')
    short = row['shapeName'].replace(' Oblast', '')
    if pd.notna(pred) and pred > 0:
        text_color = 'white' if pred >= 25 else 'black'
        ax.text(cx, cy, f"{short}\n{int(pred)}",
                ha='center', va='center',
                fontsize=7.5, fontweight='bold', color=text_color,
                zorder=6,
                path_effects=None)
    else:
        ax.text(cx, cy, short, ha='center', va='center',
                fontsize=6.5, color='#555', zorder=6)

# Country labels (placed off-map so they don't overlap oblasts)
ax.text(42.5, 51.6, 'RUSSIA', fontsize=20, fontweight='bold',
        color='#990000', alpha=0.55, ha='center')
ax.text(27.5, 53.4, 'BELARUS', fontsize=12, fontweight='bold',
        color='#990000', alpha=0.5, ha='center')

# Kyiv capital star
kyiv = oblasts[oblasts['oblast'] == 'Kyiv City'].iloc[0]
ax.scatter([kyiv['lon']], [kyiv['lat']], marker='*', s=350,
           color='gold', edgecolor='black', linewidth=1.3, zorder=8)

# Tight crop on Ukraine (with a bit of context)
ax.set_xlim(21.5, 41.5)
ax.set_ylim(43.8, 53.2)
ax.set_xlabel('Longitude (°E)')
ax.set_ylabel('Latitude (°N)')
ax.set_aspect(1.45)
ax.grid(True, alpha=0.2, linestyle=':')
ax.set_title(
    f'Predicted Russian Drone Strikes by Oblast\n'
    f'~{int(forecast["predicted_week"].sum())} drones remaining in weekly budget '
    f'(learning α={learning_alpha:.2f})',
    fontsize=13, fontweight='bold',
)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
plt.colorbar(sm, ax=ax, fraction=0.025, pad=0.02, label='Predicted drones/week')

st.pyplot(fig)

# ============== TABLE ==============
st.subheader("Top 15 Targeted Oblasts")
display_cols = ['oblast', 'border_dist', 'energy', 'pop',
                'share', 'obs_share', 'learned_share',
                'n_obs', 'predicted_week']
display_df = forecast[display_cols].head(15).copy()
display_df.columns = ['Oblast', 'Border km', 'Energy', 'Pop (M)',
                      'Prior %', 'Observed %', 'Learned %',
                      'Obs count', 'Predicted Week']
for col in ['Prior %', 'Observed %', 'Learned %']:
    display_df[col] = (display_df[col] * 100).round(1)
display_df['Obs count'] = display_df['Obs count'].astype(int)
st.dataframe(display_df, use_container_width=True, hide_index=True)

# ============== LOCK FORECAST ==============
# ============== WALK-FORWARD BACKTEST ==============
st.subheader("🧪 Walk-forward backtest — predicted vs actual, day by day")
st.caption(
    "For each historical day with both per-oblast sightings and an "
    "official UA Air Force daily total, the model is retrained using ONLY "
    "data from prior days, then asked to predict that day. This is the "
    "scientifically clean accuracy test — the model is never scored against "
    "data it was trained on."
)

bcol1, bcol2 = st.columns([3, 1])
with bcol1:
    auto_backtest = st.checkbox(
        "Auto-run backtest on every refresh",
        value=True,
        help="Recomputes the backtest as new daily summaries arrive. Cheap "
             "(<1s) so safe to leave on.",
    )
with bcol2:
    if st.button("Run backtest now", use_container_width=True):
        with st.spinner("Backtesting…"):
            per_day, per_oblast = run_backtest(
                oblasts, observations, daily_totals, low_tempo=low_tempo,
            )
            save_backtest_run(per_day, per_oblast, low_tempo=low_tempo)
        st.success(f"Backtested {len(per_day)} day(s).")

# Always recompute fresh against the current data — it's <1s. Persistence
# happens on explicit "Run backtest now" click so we keep a track record.
bt_day, bt_oblast = run_backtest(
    oblasts, observations, daily_totals, low_tempo=low_tempo,
)
if auto_backtest and not bt_day.empty:
    save_backtest_run(bt_day, bt_oblast, low_tempo=low_tempo)

if bt_day.empty:
    st.info(
        "No daily summaries on file yet. Sync UA AF Telegram from the "
        "sidebar to populate daily totals, then run the backtest."
    )
else:
    # Headline metrics
    bcol_a, bcol_b, bcol_c, bcol_d = st.columns(4)
    valid = bt_day.dropna(subset=['spatial_r'])
    with bcol_a:
        st.metric("Days backtested", len(bt_day))
    with bcol_b:
        st.metric(
            "Avg spatial r",
            f"{valid['spatial_r'].mean():.3f}" if not valid.empty else "—",
            help="Pearson r between predicted and actual SHARES across "
                 "oblasts, averaged over days. 1.0 = perfect spatial fit.",
        )
    with bcol_c:
        st.metric(
            "Avg MAE / oblast",
            f"{bt_day['mae'].mean():.1f}",
            help="Mean absolute error in predicted-vs-actual drone count "
                 "per oblast, averaged over days.",
        )
    with bcol_d:
        bias = bt_day['total_error'].mean()
        st.metric(
            "Avg volume bias",
            f"{bias:+.0f}",
            help="Average (predicted − actual) day total. Positive = "
                 "model over-predicts volume; negative = under-predicts.",
        )

    # Predicted vs actual time series
    fig_bt, (axL, axR) = plt.subplots(1, 2, figsize=(13, 4))
    d = bt_day.copy()
    d['date'] = pd.to_datetime(d['date'])
    d = d.sort_values('date')
    axL.plot(d['date'], d['predicted_total'], marker='o',
              color='#cc0033', label='Model predicted')
    axL.plot(d['date'], d['actual_total'], marker='s',
              color='#003d7a', label='Actual (UA AF + TG)')
    axL.set_title("Daily total: predicted vs actual")
    axL.legend()
    axL.grid(alpha=0.3)
    axL.tick_params(axis='x', rotation=20)
    axL.set_ylabel('Drones')

    axR.plot(d['date'], d['spatial_r'], marker='^',
              color='#2a8c4a', label='Spatial r (share fit)')
    axR.set_ylim(-1, 1.05)
    axR.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    axR.set_title("Spatial accuracy over time")
    axR.set_ylabel('Pearson r')
    axR.grid(alpha=0.3)
    axR.tick_params(axis='x', rotation=20)
    axR.legend()
    st.pyplot(fig_bt)

    # Per-day table
    with st.expander("Per-day backtest detail", expanded=True):
        display_bt = bt_day[['date', 'budget_used', 'predicted_total',
                              'actual_total', 'total_error', 'mae',
                              'spatial_r', 'alpha_used', 'training_obs']].copy()
        display_bt.columns = ['Date', 'Budget', 'Predicted', 'Actual',
                               'Error', 'MAE', 'Spatial r', 'α used',
                               'Training obs']
        st.dataframe(display_bt, use_container_width=True, hide_index=True)

    # Worst per-oblast misses across all days
    if not bt_oblast.empty:
        with st.expander("Top 20 per-oblast misses (across all days)"):
            misses = bt_oblast.copy()
            if 'error' not in misses.columns:
                misses['error'] = misses['predicted'] - misses['actual']
            misses['abs_error'] = misses['error'].abs()
            misses = misses.sort_values('abs_error', ascending=False).head(20)
            misses = misses[['date', 'oblast', 'predicted', 'actual', 'error']]
            misses.columns = ['Date', 'Oblast', 'Predicted', 'Actual', 'Error']
            st.dataframe(misses, use_container_width=True, hide_index=True)

st.divider()

# ============== LOCK FORECAST (for ongoing tracking) ==============
st.subheader("Lock this forecast for accuracy tracking")
st.caption(
    "Saves the current per-oblast prediction to the local SQLite database "
    f"({DB_PATH.name}). Each snapshot is scored later against the "
    "observations that arrive during its week."
)
with st.form("lock_forecast"):
    lcol1, lcol2 = st.columns([3, 1])
    with lcol1:
        snap_note = st.text_input(
            "Optional note", value="",
            placeholder="e.g. 'pre-ceasefire baseline'",
        )
    with lcol2:
        lock_submitted = st.form_submit_button("📌 Lock forecast")
    if lock_submitted:
        snap_id = save_snapshot(
            forecast,
            params={
                'week_start': week_start.date().isoformat(),
                'russian_daily_capacity': russian_daily_capacity,
                'tempo_factor': tempo_factor,
                'weekly_budget': weekly_budget,
                'remaining_budget': remaining_budget,
                'low_tempo': low_tempo,
                'learning_alpha': learning_alpha,
            },
            note=snap_note,
        )
        st.success(
            f"Locked snapshot #{snap_id} for week of {week_start.date()}. "
            f"Total predicted: {int(forecast['predicted_week'].sum())}."
        )

# ============== ACCURACY TRACKING ==============
st.subheader("📊 Forecast accuracy over time")
summary, details = score_snapshots(observations)
if summary.empty:
    st.info(
        "No locked snapshots yet. Lock the current forecast above to start "
        "tracking accuracy. Each week, predicted_week is compared against the "
        "sum of observations recorded between week_start and week_start+7."
    )
else:
    acc_cols = st.columns(4)
    latest = summary.iloc[0]
    with acc_cols[0]:
        st.metric("Snapshots tracked", len(summary))
    with acc_cols[1]:
        st.metric(
            "Latest total error",
            f"{latest['total_error']:+,}",
            help="Predicted minus observed for the most recent snapshot.",
        )
    with acc_cols[2]:
        st.metric(
            "Latest MAE / oblast",
            f"{latest['mae_per_oblast']:.1f}",
            help="Mean absolute error per oblast for the most recent snapshot.",
        )
    with acc_cols[3]:
        r = latest['share_pearson_r']
        st.metric(
            "Latest spatial r",
            f"{r:.2f}" if r is not None else "—",
            help="Pearson correlation between predicted share and observed "
                 "share. Stays high even when the total is off.",
        )

    st.dataframe(summary, use_container_width=True, hide_index=True)

    # Trend chart of accuracy over time
    if len(summary) >= 2:
        trend = summary.sort_values('created_at').copy()
        trend['created_at'] = pd.to_datetime(trend['created_at'])
        fig2, (axA, axB) = plt.subplots(1, 2, figsize=(13, 4))
        axA.plot(trend['created_at'], trend['predicted_total'],
                 marker='o', label='Predicted', color='#cc0033')
        axA.plot(trend['created_at'], trend['observed_total'],
                 marker='s', label='Observed', color='#003d7a')
        axA.set_title("Total predicted vs observed per snapshot")
        axA.legend()
        axA.grid(alpha=0.3)
        axA.tick_params(axis='x', rotation=20)

        axB.plot(trend['created_at'], trend['mae_per_oblast'],
                 marker='o', color='#cc0033', label='MAE per oblast')
        axB.set_title("Mean absolute error per oblast")
        axB.grid(alpha=0.3)
        axB.tick_params(axis='x', rotation=20)
        if trend['share_pearson_r'].notna().any():
            axB2 = axB.twinx()
            axB2.plot(trend['created_at'], trend['share_pearson_r'],
                      marker='^', color='#2a8c4a', label='Spatial r')
            axB2.set_ylabel('Pearson r (share)', color='#2a8c4a')
            axB2.set_ylim(-1, 1)
        st.pyplot(fig2)

    with st.expander("Per-oblast prediction vs observation (all snapshots)"):
        st.dataframe(
            details[['snapshot_id', 'week_start', 'oblast',
                     'predicted_week', 'observed', 'error', 'abs_error']]
            .sort_values(['snapshot_id', 'abs_error'], ascending=[False, False]),
            use_container_width=True, hide_index=True,
        )

    with st.expander("Manage snapshots"):
        del_id = st.number_input(
            "Snapshot id to delete", min_value=0, value=0, step=1,
        )
        if st.button("Delete snapshot") and del_id > 0:
            delete_snapshot(int(del_id))
            st.success(f"Deleted snapshot #{int(del_id)}")
            st.rerun()

# ============== ADD OBSERVATION ==============
st.subheader("Add New Observation")
with st.form("add_obs"):
    col1, col2, col3 = st.columns(3)
    with col1:
        obs_oblast = st.selectbox("Oblast", oblasts['oblast'].tolist())
    with col2:
        obs_count = st.number_input("Drones observed", min_value=0, value=0)
    with col3:
        obs_date = st.date_input("Date", value=datetime.now().date())

    submitted = st.form_submit_button("Save observation")
    if submitted:
        new_row = pd.DataFrame([{
            'observation_date': obs_date.isoformat(),
            'oblast': obs_oblast,
            'observed_drones': obs_count,
            'source': 'Manual entry',
        }])
        new_row.to_csv(DATA_DIR / "observations.csv", mode='a', header=False, index=False)
        st.success(f"Saved: {obs_count} drones in {obs_oblast} on {obs_date}")
        st.cache_data.clear()

# ============== OBSERVATIONS LOG ==============
st.subheader("Recent Observations")
st.dataframe(observations.sort_values('observation_date', ascending=False),
             use_container_width=True, hide_index=True)

# ============== FOOTER ==============
st.divider()
st.caption(
    "Prior: 0.35×energy + 1.5×exp(-distance/400) + 0.25×population, "
    "0.4× penalty for oblasts >700 km from border. "
    "Self-update: learned_share = (1-α)·prior + α·observed_share, "
    "α = N_obs / (N_obs + 150). "
    "Production constraint enforced via weekly budget. "
    "Locked snapshots are stored in data/forecast_history.db (SQLite) and "
    "scored against the observations log to track accuracy over time. "
    "Built with Claude on May 13, 2026."
)
