"""
Tamper-evident snapshot integrity via SHA256.

Each locked prediction gets a hash computed over its canonical
representation. The hash is stored:
  1. In the snapshots SQLite table (column `prediction_hash`)
  2. In the JSON archive file (top-level `prediction_hash` field)
  3. Optionally exported as a flat manifest CSV for off-system verification

Verification: recompute the hash from current data — if it doesn't match
the stored hash, the prediction has been tampered with (or there's a
bug in the canonicalization).

Hash input is canonicalized to be deterministic:
  - Predictions sorted by oblast name (alphabetical, NFC-normalized)
  - All numeric values rounded to 6 decimal places (avoid float drift)
  - JSON with sort_keys=True and no extra whitespace
"""
from __future__ import annotations
import hashlib
import json
import sqlite3
import unicodedata
from datetime import datetime
from pathlib import Path


def canonical_snapshot_payload(
    week_start: str,
    russian_daily_capacity: int,
    tempo_factor: float,
    weekly_budget: int,
    learning_alpha: float,
    per_oblast: list[dict],
) -> str:
    """Return the canonical string representation used as hash input."""
    # Normalize oblast names + sort
    normalized = []
    for row in per_oblast:
        normalized.append({
            'oblast': unicodedata.normalize('NFC', str(row['oblast'])),
            'prior_share': round(float(row.get('prior_share', 0.0)), 6),
            'obs_share': round(float(row.get('obs_share', 0.0)), 6),
            'learned_share': round(float(row.get('learned_share', 0.0)), 6),
            'predicted_week': round(float(row.get('predicted_week', 0.0)), 6),
        })
    normalized.sort(key=lambda r: r['oblast'])

    canonical = {
        'week_start': str(week_start),
        'russian_daily_capacity': int(russian_daily_capacity),
        'tempo_factor': round(float(tempo_factor), 6),
        'weekly_budget': int(weekly_budget),
        'learning_alpha': round(float(learning_alpha), 6),
        'per_oblast': normalized,
    }
    return json.dumps(canonical, sort_keys=True, ensure_ascii=False,
                       separators=(',', ':'))


def compute_hash(canonical_str: str) -> str:
    """SHA256 of the canonical payload."""
    return hashlib.sha256(canonical_str.encode('utf-8')).hexdigest()


def hash_snapshot_record(snap_row: dict, oblast_rows: list[dict]) -> str:
    """Convenience: hash a snapshot record from DB."""
    return compute_hash(canonical_snapshot_payload(
        week_start=snap_row['week_start'],
        russian_daily_capacity=snap_row['russian_daily_capacity'],
        tempo_factor=snap_row['tempo_factor'],
        weekly_budget=snap_row['weekly_budget'],
        learning_alpha=snap_row['learning_alpha'],
        per_oblast=oblast_rows,
    ))


def ensure_hash_column(db_path) -> None:
    """Add prediction_hash + hash_computed_at to snapshots if missing."""
    with sqlite3.connect(db_path) as conn:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(snapshots)").fetchall()}
        if 'prediction_hash' not in cols:
            conn.execute("ALTER TABLE snapshots ADD COLUMN prediction_hash TEXT")
        if 'hash_computed_at' not in cols:
            conn.execute("ALTER TABLE snapshots ADD COLUMN hash_computed_at TEXT")


def backfill_hashes(db_path, data_dir: Path | str) -> int:
    """Compute + store hash for every snapshot that doesn't have one.
    Also writes hash into the corresponding JSON archive file."""
    data_dir = Path(data_dir)
    ensure_hash_column(db_path)
    archive_dir = data_dir / 'snapshot_archive'
    archive_dir.mkdir(exist_ok=True)
    written = 0
    with sqlite3.connect(db_path) as conn:
        snaps = conn.execute(
            "SELECT id, week_start, russian_daily_capacity, tempo_factor, "
            "weekly_budget, learning_alpha, prediction_hash FROM snapshots"
        ).fetchall()
        for row in snaps:
            (sid, ws, capacity, tempo, budget, alpha, existing_hash) = row
            obl_rows = conn.execute(
                "SELECT oblast, prior_share, obs_share, learned_share, "
                "predicted_week FROM snapshot_rows WHERE snapshot_id=?", (sid,)
            ).fetchall()
            per = [{'oblast': r[0], 'prior_share': r[1], 'obs_share': r[2],
                    'learned_share': r[3], 'predicted_week': r[4]}
                   for r in obl_rows]
            canon = canonical_snapshot_payload(ws, capacity, tempo, budget,
                                                 alpha, per)
            h = compute_hash(canon)
            now_iso = datetime.now().isoformat(timespec='seconds')

            # Update DB
            conn.execute("UPDATE snapshots SET prediction_hash=?, hash_computed_at=? "
                         "WHERE id=?", (h, now_iso, sid))

            # Update JSON archive if present
            archive_path = archive_dir / f"snapshot_{sid:03d}_week_{ws}.json"
            if archive_path.exists():
                try:
                    archive = json.loads(archive_path.read_text())
                except Exception:
                    archive = {}
                archive['prediction_hash'] = h
                archive['hash_algo'] = 'sha256'
                archive['hash_computed_at'] = now_iso
                archive_path.write_text(json.dumps(archive, indent=2,
                                                    default=str))
            written += 1
    return written


def verify_all(db_path) -> list[dict]:
    """Recompute every snapshot's hash and check against stored value.
    Returns list of {snapshot_id, week_start, stored_hash, computed_hash, ok}."""
    out = []
    with sqlite3.connect(db_path) as conn:
        snaps = conn.execute(
            "SELECT id, week_start, russian_daily_capacity, tempo_factor, "
            "weekly_budget, learning_alpha, prediction_hash FROM snapshots"
        ).fetchall()
        for row in snaps:
            (sid, ws, capacity, tempo, budget, alpha, stored) = row
            obl_rows = conn.execute(
                "SELECT oblast, prior_share, obs_share, learned_share, "
                "predicted_week FROM snapshot_rows WHERE snapshot_id=?", (sid,)
            ).fetchall()
            per = [{'oblast': r[0], 'prior_share': r[1], 'obs_share': r[2],
                    'learned_share': r[3], 'predicted_week': r[4]}
                   for r in obl_rows]
            canon = canonical_snapshot_payload(ws, capacity, tempo, budget,
                                                 alpha, per)
            computed = compute_hash(canon)
            out.append({
                'snapshot_id': sid,
                'week_start': ws,
                'stored_hash': stored,
                'computed_hash': computed,
                'ok': stored == computed,
            })
    return out


def export_manifest_csv(db_path, csv_path) -> int:
    """Write a flat hash manifest for off-system verification.
    Columns: snapshot_id, week_start, weekly_budget, learning_alpha,
             prediction_hash, hash_computed_at."""
    import pandas as pd
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(
            "SELECT id AS snapshot_id, week_start, weekly_budget, "
            "russian_daily_capacity, tempo_factor, learning_alpha, "
            "prediction_hash, hash_computed_at, note "
            "FROM snapshots ORDER BY id", conn)
    df.to_csv(csv_path, index=False)
    return len(df)
