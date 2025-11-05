#!/usr/bin/env python3
"""
health_check.py

Checks the integrity of results produced by the analysis pipeline and optionally
verifies that the Flask endpoint /results_data/<rel> can serve each asset.

Usage:
  python health_check.py                 # auto-discover results root, file checks only
  python health_check.py --dry-run       # same as above (no http requests)
  python health_check.py --root /path/to/results --server-url http://127.0.0.1:8080
"""

import argparse
import json
import os
from pathlib import Path
import glob
import sys

try:
    import requests
except Exception:
    requests = None

COMMON_CANDIDATES = [
    'results',
    'Scratch_Assay_Analysis/results',
    'Scratch_Assay_Analysis/Scratch_Assay_Analysis/results',
    os.path.join(str(Path.home()), 'helloworld', 'woundhealinganalysis', 'Scratch_Assay_Analysis', 'results'),
    os.path.join(str(Path.home()), 'woundhealinganalysis', 'Scratch_Assay_Analysis', 'results'),
]


def find_results_root(provided_root=None):
    if provided_root:
        p = Path(provided_root).expanduser().resolve()
        if p.is_dir():
            return str(p)
        else:
            return None
    cwd = Path.cwd()
    # check common candidates relative to cwd
    for cand in COMMON_CANDIDATES:
        p = Path(cand)
        if not p.is_absolute():
            p = cwd.joinpath(cand)
        if p.is_dir():
            # ensure there are *_summary.json files
            if any(p.rglob('*_summary.json')):
                return str(p.resolve())
    # walk up to 3 levels looking for results/
    cur = cwd
    for _ in range(4):
        cand = cur / 'results'
        if cand.is_dir() and any(cand.rglob('*_summary.json')):
            return str(cand.resolve())
        cur = cur.parent
    return None


def load_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        return None


def relpath_under_root(fs_path, root):
    try:
        return os.path.relpath(fs_path, root).replace(os.sep, '/')
    except Exception:
        return fs_path


def check_asset_file(fs_path):
    if not fs_path:
        return False, "missing"
    return (os.path.exists(fs_path), "exists" if os.path.exists(fs_path) else "missing")


def http_check(url, timeout=4.0):
    if requests is None:
        return False, "requests_not_installed"
    try:
        r = requests.head(url, timeout=timeout, allow_redirects=True)
        if r.status_code == 200:
            return True, f"HTTP 200"
        else:
            return False, f"HTTP {r.status_code}"
    except Exception as e:
        return False, f"err: {e}"


def main():
    parser = argparse.ArgumentParser(description="Health check results & optional HTTP serving")
    parser.add_argument('--root', type=str, default=None, help='Path to results root (optional)')
    parser.add_argument('--server-url', type=str, default=None, help='Base server URL to test /results_data/ endpoints, e.g. http://127.0.0.1:8080')
    parser.add_argument('--dry-run', action='store_true', help='Do not perform HTTP checks (faster).')
    parser.add_argument('--limit', type=int, default=None, help='Limit how many experiments to check (for speed).')
    args = parser.parse_args()

    results_root = find_results_root(args.root)
    if not results_root:
        print("Could not find results root. Try passing --root /full/path/to/results")
        sys.exit(2)
    print("Results root:", results_root)

    # find all summary JSONs
    summary_paths = sorted([str(p) for p in Path(results_root).rglob('*_summary.json')])
    if not summary_paths:
        print("No *_summary.json files found under results root.")
        sys.exit(1)

    if args.limit:
        summary_paths = summary_paths[: args.limit]

    overall_ok = True
    print(f"Found {len(summary_paths)} summary files. Checking each...\n")

    for summary in summary_paths:
        print("=" * 60)
        print("Summary:", summary)
        data = load_json(summary)
        if data is None:
            print("  -> ERROR: could not load JSON or invalid JSON.")
            overall_ok = False
            continue

        # Gather candidates: prefer explicit csv_path / plot_path / plot_json / video_path / gallery_thumbs
        # If keys are relative, attempt to resolve them under results_root
        keys = {
            'csv_path': data.get('csv_path') or data.get('timeseries_csv') or data.get('trajectories_csv') or None,
            'plot_path': data.get('plot_path') or data.get('plot_png') or None,
            'plot_json': data.get('plot_json') or data.get('interactive_plot') or None,
            'video_path': data.get('video_path') or data.get('video') or None,
            'gallery': data.get('gallery') or data.get('gallery_thumbs') or None,
            'trajectories_plot': data.get('trajectories_plot') or data.get('trajectories_plot') or None
        }

        # If values look like relative paths (not absolute), join with results_root
        resolved = {}
        for k, v in keys.items():
            if not v:
                resolved[k] = None
                continue
            if isinstance(v, list):
                resolved_list = []
                for item in v:
                    if not item:
                        continue
                    if os.path.isabs(item):
                        resolved_list.append(item)
                    else:
                        ab = os.path.join(results_root, item)
                        resolved_list.append(ab)
                resolved[k] = resolved_list
            else:
                if os.path.isabs(v):
                    resolved[k] = v
                else:
                    resolved[k] = os.path.join(results_root, v)

        print("  Experiment:", data.get('experiment') or os.path.splitext(os.path.basename(summary))[0])
        # check each path
        checks = [
            ('CSV', resolved.get('csv_path')),
            ('Static Plot', resolved.get('plot_path')),
            ('Interactive JSON', resolved.get('plot_json')),
            ('Video', resolved.get('video_path')),
            ('Trajectories Plot', resolved.get('trajectories_plot'))
        ]
        for label, p in checks:
            ok, msg = check_asset_file(p)
            status = "OK" if ok else "MISSING"
            print(f"   {label:18s}: {p if p else 'None':80.80s} -> {status} ({msg})")
            if not ok:
                overall_ok = False

            # If server-url given and file exists, try HTTP head request to /results_data/<rel>
            if args.server_url and not args.dry_run:
                if p and os.path.exists(p):
                    rel = relpath_under_root(p, results_root)
                    url = args.server_url.rstrip('/') + '/results_data/' + rel
                    ok_http, http_msg = http_check(url)
                    print(f"       -> HTTP check: {url} -> {http_msg}")
                    if not ok_http:
                        overall_ok = False

        # gallery list
        gal = resolved.get('gallery') or []
        if gal:
            print("   Gallery (showing up to 6):")
            for g in gal[:6]:
                ok, msg = check_asset_file(g)
                status = "OK" if ok else "MISSING"
                print(f"      - {g:80.80s} -> {status} ({msg})")
                if args.server_url and not args.dry_run and ok:
                    rel = relpath_under_root(g, results_root)
                    url = args.server_url.rstrip('/') + '/results_data/' + rel
                    ok_http, http_msg = http_check(url)
                    print(f"           HTTP: {http_msg}")
                    if not ok_http:
                        overall_ok = False
        else:
            print("   Gallery: none listed in JSON.")

        # quick sanity check: the static plot file size (if exists)
        p = resolved.get('plot_path')
        if p and os.path.exists(p):
            try:
                size = os.path.getsize(p)
                print(f"   Static plot size: {size/1024:.1f} KB")
            except Exception:
                pass

    print("\n" + "=" * 60)
    if overall_ok:
        print("HEALTH CHECK: OK. All referenced assets exist and HTTP endpoints returned 200 (if tested).")
        sys.exit(0)
    else:
        print("HEALTH CHECK: WARN/ERROR found. Some assets are missing or HTTP checks failed.")
        print(" - If HTTP checks failed, verify Flask's RESULTS_FOLDER and that /results_data/<rel> is reachable.")
        print(" - Run fix_results_structure.py to normalize folder layout if needed.")
        sys.exit(3)


if __name__ == '__main__':
    main()
