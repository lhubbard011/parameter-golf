#!/usr/bin/env python3
"""
Experiment results database for autoresearch + parameter-golf.

Usage:
    # Add a result
    python db.py add --project pgolf --desc "MoR 4x int8" --val_bpb 1.4010 --params_m 7.6 --artifact_mb 5.3 --steps 2326 --commit abc1234

    # List results
    python db.py list                        # all
    python db.py list --project pgolf        # filter by project
    python db.py list --status keep          # filter by status

    # Import from TSV
    python db.py import-tsv autoresearch results.tsv
    python db.py import-tsv pgolf pgolf_results.tsv

    # Best results
    python db.py best                        # overall best
    python db.py best --project pgolf        # best per project

    # Stats
    python db.py stats

    # JSON export (for dashboard)
    python db.py json
"""

import argparse
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent / "experiments.db"


def get_db():
    db = sqlite3.connect(str(DB_PATH))
    db.row_factory = sqlite3.Row
    db.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL DEFAULT (datetime('now')),
            project TEXT NOT NULL,
            description TEXT NOT NULL,
            val_bpb REAL,
            val_loss REAL,
            params_m REAL,
            artifact_mb REAL,
            memory_gb REAL,
            steps INTEGER,
            commit_hash TEXT,
            status TEXT DEFAULT 'run',
            env_vars TEXT,
            notes TEXT
        )
    """)
    db.execute("""
        CREATE INDEX IF NOT EXISTS idx_project ON experiments(project)
    """)
    db.execute("""
        CREATE INDEX IF NOT EXISTS idx_status ON experiments(status)
    """)
    db.commit()
    return db


def cmd_add(args):
    db = get_db()
    db.execute("""
        INSERT INTO experiments (project, description, val_bpb, val_loss, params_m, artifact_mb,
                                 memory_gb, steps, commit_hash, status, env_vars, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        args.project, args.desc, args.val_bpb, args.val_loss, args.params_m,
        args.artifact_mb, args.memory_gb, args.steps, args.commit, args.status,
        args.env, args.notes,
    ))
    db.commit()
    print(f"Added: {args.desc} (val_bpb={args.val_bpb})")


def cmd_list(args):
    db = get_db()
    query = "SELECT * FROM experiments WHERE 1=1"
    params = []
    if args.project:
        query += " AND project = ?"
        params.append(args.project)
    if args.status:
        query += " AND status = ?"
        params.append(args.status)
    query += " ORDER BY timestamp DESC"
    if args.limit:
        query += f" LIMIT {args.limit}"

    rows = db.execute(query, params).fetchall()
    if not rows:
        print("No results found.")
        return

    # Header
    print(f"{'ID':>4} {'Project':<14} {'val_bpb':>9} {'Params':>8} {'Artifact':>9} {'Steps':>6} {'Status':<8} Description")
    print("-" * 100)
    for r in rows:
        bpb = f"{r['val_bpb']:.6f}" if r['val_bpb'] else "—"
        params = f"{r['params_m']:.1f}M" if r['params_m'] else "—"
        artifact = f"{r['artifact_mb']:.1f}MB" if r['artifact_mb'] else "—"
        steps = str(r['steps']) if r['steps'] else "—"
        status = r['status'] or "—"
        print(f"{r['id']:>4} {r['project']:<14} {bpb:>9} {params:>8} {artifact:>9} {steps:>6} {status:<8} {r['description']}")


def cmd_best(args):
    db = get_db()
    query = """
        SELECT project, MIN(val_bpb) as best_bpb, description, params_m, artifact_mb, steps, commit_hash
        FROM experiments
        WHERE val_bpb > 0
    """
    params = []
    if args.project:
        query += " AND project = ?"
        params.append(args.project)
    query += " GROUP BY project ORDER BY best_bpb"

    rows = db.execute(query, params).fetchall()
    for r in rows:
        print(f"[{r['project']}] Best: {r['best_bpb']:.6f} — {r['description']}")
        if r['params_m']:
            print(f"  params={r['params_m']:.1f}M artifact={r['artifact_mb'] or '?'}MB steps={r['steps'] or '?'}")


def cmd_stats(args):
    db = get_db()
    rows = db.execute("""
        SELECT project,
               COUNT(*) as total,
               SUM(CASE WHEN status='keep' THEN 1 ELSE 0 END) as kept,
               SUM(CASE WHEN status='discard' THEN 1 ELSE 0 END) as discarded,
               SUM(CASE WHEN status='crash' THEN 1 ELSE 0 END) as crashed,
               MIN(CASE WHEN val_bpb > 0 THEN val_bpb END) as best,
               MAX(CASE WHEN val_bpb > 0 THEN val_bpb END) as worst
        FROM experiments GROUP BY project
    """).fetchall()

    for r in rows:
        print(f"\n[{r['project']}]")
        print(f"  Total: {r['total']} | Kept: {r['kept']} | Discarded: {r['discarded']} | Crashed: {r['crashed']}")
        print(f"  Best: {r['best']:.6f} | Worst: {r['worst']:.6f}")


def cmd_import_tsv(args):
    db = get_db()
    path = Path(args.file)
    project = args.project

    with open(path) as f:
        header = f.readline().strip().split("\t")
        count = 0
        for line in f:
            fields = line.strip().split("\t")
            if len(fields) < 2:
                continue

            if project == "autoresearch":
                # Format: commit, val_bpb, memory_gb, status, description
                commit = fields[0] if len(fields) > 0 else None
                val_bpb = float(fields[1]) if len(fields) > 1 and fields[1] != "0.000000" else None
                memory_gb = float(fields[2]) if len(fields) > 2 else None
                status = fields[3] if len(fields) > 3 else None
                desc = fields[4] if len(fields) > 4 else ""
                db.execute("""
                    INSERT INTO experiments (project, description, val_bpb, memory_gb, commit_hash, status)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (project, desc, val_bpb, memory_gb, commit, status))

            elif project == "pgolf":
                # Format: timestamp, val_bpb, val_loss, artifact_mb, steps, description
                ts = fields[0] if len(fields) > 0 else None
                val_bpb = float(fields[1]) if len(fields) > 1 else None
                val_loss = float(fields[2]) if len(fields) > 2 else None
                artifact_mb = float(fields[3]) if len(fields) > 3 and fields[3] != "0.00" else None
                steps = int(fields[4]) if len(fields) > 4 else None
                desc = fields[5] if len(fields) > 5 else ""
                db.execute("""
                    INSERT INTO experiments (project, description, val_bpb, val_loss, artifact_mb,
                                             steps, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (project, desc, val_bpb, val_loss, artifact_mb, steps, ts))

            count += 1

    db.commit()
    print(f"Imported {count} experiments from {path} into project '{project}'")


def cmd_json(args):
    db = get_db()
    rows = db.execute("SELECT * FROM experiments ORDER BY id").fetchall()
    data = [dict(r) for r in rows]
    print(json.dumps(data, indent=2, default=str))


def main():
    parser = argparse.ArgumentParser(description="Experiment results database")
    sub = parser.add_subparsers(dest="cmd")

    p_add = sub.add_parser("add")
    p_add.add_argument("--project", required=True)
    p_add.add_argument("--desc", required=True)
    p_add.add_argument("--val_bpb", type=float)
    p_add.add_argument("--val_loss", type=float)
    p_add.add_argument("--params_m", type=float)
    p_add.add_argument("--artifact_mb", type=float)
    p_add.add_argument("--memory_gb", type=float)
    p_add.add_argument("--steps", type=int)
    p_add.add_argument("--commit")
    p_add.add_argument("--status", default="run")
    p_add.add_argument("--env")
    p_add.add_argument("--notes")

    p_list = sub.add_parser("list")
    p_list.add_argument("--project")
    p_list.add_argument("--status")
    p_list.add_argument("--limit", type=int)

    p_best = sub.add_parser("best")
    p_best.add_argument("--project")

    sub.add_parser("stats")

    p_import = sub.add_parser("import-tsv")
    p_import.add_argument("project")
    p_import.add_argument("file")

    sub.add_parser("json")

    args = parser.parse_args()
    if not args.cmd:
        parser.print_help()
        return

    {"add": cmd_add, "list": cmd_list, "best": cmd_best, "stats": cmd_stats,
     "import-tsv": cmd_import_tsv, "json": cmd_json}[args.cmd](args)


if __name__ == "__main__":
    main()
