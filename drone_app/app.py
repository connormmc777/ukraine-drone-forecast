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


# NB: the mtime_key argument MUST NOT start with an underscore. Streamlit's
# @st.cache_data treats leading-underscore args as "unhashable" and skips
# them for cache-key computation — which means the cache never invalidates
# on file changes and every render serves whatever was cached at process
# start. The ttl is a belt-and-suspenders: even if a caller forgets to pass
# a fresh mtime, the cache expires within 60s.
@st.cache_data(ttl=60)
def load_oblasts(mtime_key: float):
    return pd.read_csv(DATA_DIR / "oblast_features.csv")


@st.cache_data(ttl=60)
def load_observations(mtime_key: float):
    return pd.read_csv(DATA_DIR / "observations.csv")


@st.cache_data(ttl=60)
def load_daily_totals(mtime_key: float):
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
_panel['days_since_hit'] = (today - _panel['last_hit']).dt.days

# ---- Calibration state: is Actual in "drones" units or raw "sightings"? ----
# When the morning summary has landed for the latest observation date, the
# observations are already scaled to the summary total → real drone counts.
# Otherwise the counts are raw sighting posts, and each drone appears in
# multiple posts as it crosses oblasts. We reverse-engineer the multiplier
# from historical nights where BOTH metrics exist.
def _sighting_to_drone_ratio(_obs_df, _daily_df):
    if _daily_df.empty:
        return None
    _pn = _obs_df.groupby(_obs_df['observation_date'].dt.date)['observed_drones'].sum()
    _d = _daily_df.copy()
    _d['date'] = pd.to_datetime(_d['date']).dt.date
    _dn = _d.groupby('date')['launched'].sum()
    _m = pd.DataFrame({'sightings': _pn, 'launched': _dn}).dropna()
    _m = _m[_m['launched'] > 0]
    if len(_m) < 5:
        return None
    _r = _m['launched'] / _m['sightings']
    return {'median': float(_r.median()),
            'p25': float(_r.quantile(0.25)),
            'p75': float(_r.quantile(0.75)),
            'n': int(len(_m))}

_ratio_stats = _sighting_to_drone_ratio(observations, daily_totals)
_summary_landed = False
if not daily_totals.empty:
    _daily_dates = set(pd.to_datetime(daily_totals['date']).dt.date.tolist())
    _summary_landed = _latest_date_overall.date() in _daily_dates

# Estimated actual (drone units) — same as raw if calibrated, else × median
if _summary_landed or _ratio_stats is None:
    _panel['est_actual'] = _panel['actual_today'].astype(float)
    _panel['est_low'] = _panel['est_actual']
    _panel['est_high'] = _panel['est_actual']
else:
    _panel['est_actual'] = _panel['actual_today'] * _ratio_stats['median']
    _panel['est_low'] = _panel['actual_today'] * _ratio_stats['p25']
    _panel['est_high'] = _panel['actual_today'] * _ratio_stats['p75']

# Volume delta (only interpretable when calibrated)
_panel['delta_vol'] = _panel['est_actual'] - _panel['today_budget']

# Spatial-share delta (unit-free — meaningful even with raw sightings, since
# both actual and predicted are shares of tonight's total)
_tonight_total = float(_panel['actual_today'].sum())
if _tonight_total > 0:
    _panel['actual_share'] = _panel['actual_today'] / _tonight_total
else:
    _panel['actual_share'] = 0.0
_panel['delta_share_pp'] = (_panel['actual_share'] - _panel['share']) * 100.0

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

    # Calibration state banner — one line explaining what units 'Actual' is in
    if _summary_landed:
        st.success(
            "🟢 **Calibrated** — Actual counts are rescaled to match "
            f"tonight's UA Air Force morning summary total. "
            "'Actual' and 'Budget' are in the same units (drones)."
        )
    elif _ratio_stats:
        st.warning(
            f"🟡 **Uncalibrated — morning summary pending.** Actual counts "
            f"are raw sighting posts (each drone can appear in several). "
            f"Historical median: **1 sighting ≈ {_ratio_stats['median']:.2f} "
            f"drones** (IQR {_ratio_stats['p25']:.2f}–{_ratio_stats['p75']:.2f}, "
            f"n = {_ratio_stats['n']} nights). The 'Est drones' column applies "
            f"the median so it's comparable to 'Budget'. When the morning "
            f"summary lands (~06:00 Kyiv) this rescales automatically."
        )
    else:
        st.warning(
            "🟡 **Uncalibrated & no historical ratio yet** — Actual is raw "
            "sighting posts. Not directly comparable to Budget."
        )

    _display = _panel.copy()
    _display['last_hit'] = _display['last_hit'].dt.date
    _cols_out = [
        'oblast', 'last_hit', 'days_since_hit',
        'actual_today', 'est_actual', 'today_budget',
        'delta_vol', 'delta_share_pp',
    ]
    _display = _display[_cols_out]
    _actual_label = 'Actual (drones)' if _summary_landed \
                    else 'Actual (sightings)'
    _display.columns = [
        'Region', 'Last hit', 'Days since',
        _actual_label, 'Est drones', 'Budget today',
        'Δ volume', 'Δ share (pp)',
    ]
    _display['Est drones'] = _display['Est drones'].round(1)
    _display['Budget today'] = _display['Budget today'].round(1)
    _display['Δ volume'] = _display['Δ volume'].round(1)
    _display['Δ share (pp)'] = _display['Δ share (pp)'].round(2)
    st.dataframe(_display, use_container_width=True, hide_index=True)
    st.caption(
        "**Δ volume** = Est drones − Budget (in drone units, meaningful once "
        "calibrated). **Δ share (pp)** = actual share of tonight's total "
        "− predicted share, in percentage points — this is UNIT-FREE and "
        "meaningful even before the morning summary lands. Positive = region "
        "is over-represented tonight vs the structural prior."
    )

with _col_bar:
    if len(_panel):
        _top = _panel.iloc[0]
        _fig_l, _ax_l = plt.subplots(figsize=(4, 3))
        # Use est_actual so bars are always in drone units (apples-to-apples
        # with Budget). When calibrated est_actual == actual_today.
        _act_val = float(_top['est_actual'])
        _bud_val = float(_top['today_budget'])
        _vals = [_act_val, _bud_val]
        _bars = _ax_l.bar(['Est actual', 'Budget'], _vals,
                          color=['#cc0033', '#003d7a'])
        # IQR error bar on the estimated actual (only visible when uncalibrated)
        if not _summary_landed and _ratio_stats:
            _lo = float(_top['est_low'])
            _hi = float(_top['est_high'])
            _ax_l.errorbar([0], [_act_val],
                            yerr=[[_act_val - _lo], [_hi - _act_val]],
                            fmt='none', color='black', capsize=6, linewidth=1.2)
        _cal_tag = 'calibrated' if _summary_landed else 'ratio-estimated'
        _ax_l.set_title(f"{_top['oblast']} — today ({_cal_tag})", fontsize=10)
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

# ============== TEARDOWN ==============
# Streamlit reruns this script on every autorefresh tick (~30s). Matplotlib
# figures created with plt.subplots() stay alive in the pyplot state machine
# until explicitly closed — st.pyplot() serializes the figure to PNG and
# returns, but does NOT close it. 15 open figures * 30s render loop =
# ~50MB/min of leaked figure state, which is why the app container was
# OOMKilled every few hours at 4Gi. plt.close('all') here releases every
# figure at the end of each rerun.
plt.close('all')
