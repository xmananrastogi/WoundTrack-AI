#!/usr/bin/env python3
"""
fix_results_structure.py

Safe tool to normalize the results folder layout for WoundTrack AI.

Usage:
    python fix_results_structure.py --dry-run
    python fix_results_structure.py --apply
    python fix_results_structure.py --root /full/path/to/results --apply

Default behavior is a dry-run. When --apply is provided the script will move files.
"""

import argparse
import json
import os
import shutil
import glob
from pathlib import Path
from typing import List, Tuple

def find_results_root(candidate_paths: List[str]) -> str:
    for p in candidate_paths:
        if p and os.path.isdir(p):
            # Ensure there are summary files under it
            summaries = list(Path(p).rglob('*_summary.json'))
            if summaries:
                return os.path.abspath(p)
    return ""

def safe_move(src: str, dst: str, dry_run: bool) -> Tuple[str, str]:
    """
    Move src -> dst safely (create dst dir). If dst exists, append suffix.
    Returns (src, final_dst) (final_dst is path that would be used or was used).
    """
    dst_dir = os.path.dirname(dst)
    os.makedirs(dst_dir, exist_ok=True)
    base, ext = os.path.splitext(os.path.basename(dst))
    candidate = dst
    i = 1
    while os.path.exists(candidate):
        candidate = os.path.join(dst_dir, f"{base}__dup{i}{ext}")
        i += 1
    if dry_run:
        return (src, candidate)
    else:
        shutil.move(src, candidate)
        return (src, candidate)

def ensure_subdirs(base_experiment_dir: str):
    for sub in ('csv', 'plots', 'gallery', 'video', 'tracking'):
        os.makedirs(os.path.join(base_experiment_dir, sub), exist_ok=True)

def discover_assets(summary_path: str) -> dict:
    """
    Given a summary JSON path, find candidate asset files around it.
    Returns dict with paths (maybe None) for keys: timeseries, summary, plot_static,
    plot_interactive, gallery_list, video, tracking_files (list)
    """
    base = os.path.dirname(summary_path)
    name = os.path.splitext(os.path.basename(summary_path))[0]
    # candidate names (common patterns)
    candidates = {}
    candidates['summary'] = summary_path
    # timeseries - common names
    timeseries_patterns = [
        f"{name.replace('_summary','')}_timeseries.csv",
        f"{name}_timeseries.csv",
        f"{name.replace('_summary','')}.csv",
        f"{name}.csv",
    ]
    candidates['timeseries'] = None
    for p in timeseries_patterns:
        for try_path in (os.path.join(base, 'csv', p), os.path.join(base, p), os.path.join(base, '..', 'csv', p), os.path.join(base, '..', p)):
            if os.path.exists(try_path):
                candidates['timeseries'] = os.path.abspath(try_path)
                break
        if candidates['timeseries']:
            break

    # static plot
    plot_patterns = [
        f"{name.replace('_summary','')}_analysis.png",
        f"{name}_analysis.png",
        f"{name}.png",
        f"{name.replace('_summary','')}.png"
    ]
    candidates['plot_static'] = None
    for p in plot_patterns:
        for try_path in (os.path.join(base, 'plots', p), os.path.join(base, p), os.path.join(base, '..', 'plots', p)):
            if os.path.exists(try_path):
                candidates['plot_static'] = os.path.abspath(try_path)
                break
        if candidates['plot_static']:
            break

    # interactive
    interactive_patterns = [
        f"{name.replace('_summary','')}_analysis_interactive.json",
        f"{name}_analysis_interactive.json",
        f"{name}_interactive.json"
    ]
    candidates['plot_interactive'] = None
    for p in interactive_patterns:
        for try_path in (os.path.join(base, 'plots', p), os.path.join(base, p), os.path.join(base, '..', 'plots', p)):
            if os.path.exists(try_path):
                candidates['plot_interactive'] = os.path.abspath(try_path)
                break
        if candidates['plot_interactive']:
            break

    # gallery (all images under gallery folder)
    gallery_candidates = []
    for gdir in (os.path.join(base, 'gallery'), os.path.join(base, '..', 'gallery'), os.path.join(base, 'plots', 'gallery')):
        if os.path.isdir(gdir):
            for ext in ('*.png', '*.jpg', '*.jpeg', '*.svg'):
                gallery_candidates += sorted(glob.glob(os.path.join(gdir, ext)))
    # also check for nested 'gallery/gallery' weirdness
    nested = []
    for p in Path(base).rglob('gallery'):
        try:
            gp = str(p)
            for ext in ('*.png', '*.jpg', '*.jpeg', '*.svg'):
                nested += sorted(glob.glob(os.path.join(gp, ext)))
        except Exception:
            pass
    gallery_candidates += nested
    candidates['gallery'] = sorted(list(dict.fromkeys(gallery_candidates)))

    # video
    video_patterns = [
        f"{name.replace('_summary','')}_analysis_video.mp4",
        f"{name}_analysis_video.mp4",
        f"{name}.mp4"
    ]
    candidates['video'] = None
    for p in video_patterns:
        for try_path in (os.path.join(base, 'video', p), os.path.join(base, p), os.path.join(base, '..', 'video', p)):
            if os.path.exists(try_path):
                candidates['video'] = os.path.abspath(try_path)
                break
        if candidates['video']:
            break

    # tracking files (trajectories.csv / velocities.csv / trajectories_plot.png)
    tracking_list = []
    for tname in ('trajectories.csv', 'velocities.csv', 'trajectories_plot.png', 'trajectories_plot.jpg', 'trajectories.png'):
        for try_path in (os.path.join(base, 'tracking', tname), os.path.join(base, tname), os.path.join(base, '..', 'tracking', tname)):
            if os.path.exists(try_path):
                tracking_list.append(os.path.abspath(try_path))
    candidates['tracking'] = sorted(list(dict.fromkeys(tracking_list)))

    return candidates

def update_summary_paths(summary_json_path: str, new_paths: dict, results_root: str):
    """
    Load JSON, replace full file paths for keys we moved with relative /results_data style
    (store relative paths inside summary so front-end can path_to_url_for_result).
    """
    try:
        with open(summary_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        return False

    # helper to convert an absolute fs path under results_root to stored relative path
    def rel_for(fs_path):
        if not fs_path:
            return None
        try:
            abs_fs = os.path.abspath(fs_path)
            base = os.path.abspath(results_root)
            rel = os.path.relpath(abs_fs, base)
            return rel.replace(os.sep, '/')
        except Exception:
            return fs_path

    # common keys to update
    mapping = {
        'timeseries': ['timeseries_csv', 'timeseries', 'csv_path', 'trajectories_csv'],
        'plot_static': ['plot_path', 'plot', 'plot_png'],
        'plot_interactive': ['plot_json', 'interactive_plot'],
        'video': ['video_path', 'video'],
        'gallery': ['gallery', 'gallery_thumbs'],
        'tracking': ['trajectories_csv', 'trajectories_plot', 'velocities_csv']
    }

    # For each moved item, set sensible keys in JSON
    if new_paths.get('timeseries'):
        data['csv_path'] = rel_for(new_paths['timeseries'])
    if new_paths.get('summary'):
        data['summary_path'] = rel_for(new_paths['summary'])  # optional
    if new_paths.get('plot_static'):
        data['plot_path'] = rel_for(new_paths['plot_static'])
        # also set plot_b64 to None so front-end will try to load via /results_data/
        data['plot_b64'] = None
    if new_paths.get('plot_interactive'):
        data['plot_json'] = rel_for(new_paths['plot_interactive'])
    if new_paths.get('video'):
        data['video_path'] = rel_for(new_paths['video'])
    if new_paths.get('gallery'):
        data['gallery'] = [rel_for(p) for p in new_paths['gallery']]
        data['gallery_thumbs'] = [rel_for(p) for p in new_paths['gallery'][:3]]
    if new_paths.get('tracking'):
        # pick relevant tracking paths
        for t in new_paths['tracking']:
            if t.lower().endswith('.csv') and 'traject' in t.lower():
                data['trajectories_csv'] = rel_for(t)
            if t.lower().endswith('.csv') and 'velocity' in t.lower():
                data['velocities_csv'] = rel_for(t)
            if any(x in t.lower() for x in ('plot', 'trajectories_plot')):
                data['trajectories_plot'] = rel_for(t)

    # write back
    try:
        with open(summary_json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception:
        return False

def normalize_experiment_path(results_root: str, summary_file: str, dry_run: bool):
    """
    Determine canonical condition & experiment from summary path, and plan moves for files found near summary.
    Performs moves if dry_run is False.
    """
    # heuristics like the web app: if path is results/<condition>/<exp>/... or results/<condition>/<exp>_summary.json
    base = os.path.abspath(results_root)
    rel = os.path.relpath(summary_file, base).replace(os.sep, '/')
    parts = rel.split('/')
    # Choose condition and experiment name heuristically
    condition = 'Unknown'
    experiment = os.path.splitext(os.path.basename(summary_file))[0].replace('_summary', '')
    if len(parts) >= 2:
        # parts[0] usually condition
        condition = parts[0]
        # if summary inside nested folder like condition/exp/... find exp
        if len(parts) >= 3:
            experiment = parts[1]
        else:
            # summary directly under condition/...
            # if filename begins with condition_ then extract remainder (condition_experiment_summary.json)
            base_name = os.path.splitext(os.path.basename(summary_file))[0]
            if base_name.startswith(condition + '_'):
                experiment = base_name[len(condition)+1:].replace('_summary','')
            else:
                experiment = base_name.replace('_summary','')
    # Build canonical directories
    experiment_dir = os.path.join(base, condition, experiment)
    ensure_subdirs(experiment_dir)

    # discover assets near original summary location
    assets = discover_assets(summary_file)

    planned = {'summary': None, 'timeseries': None, 'plot_static': None, 'plot_interactive': None, 'gallery': [], 'video': None, 'tracking': []}
    # move summary JSON into csv/ as *_summary.json (keep name)
    dst_summary = os.path.join(experiment_dir, 'csv', os.path.basename(summary_file))
    planned['summary'] = dst_summary
    moved = {}

    # Move/plan each asset into the canonical subfolders
    if assets.get('timeseries'):
        dst_ts = os.path.join(experiment_dir, 'csv', os.path.basename(assets['timeseries']))
        moved['timeseries'] = safe_move(assets['timeseries'], dst_ts, dry_run)
        planned['timeseries'] = moved['timeseries'][1]
    if assets.get('plot_static'):
        dst_plot = os.path.join(experiment_dir, 'plots', os.path.basename(assets['plot_static']))
        moved['plot_static'] = safe_move(assets['plot_static'], dst_plot, dry_run)
        planned['plot_static'] = moved['plot_static'][1]
    if assets.get('plot_interactive'):
        dst_plotj = os.path.join(experiment_dir, 'plots', os.path.basename(assets['plot_interactive']))
        moved['plot_interactive'] = safe_move(assets['plot_interactive'], dst_plotj, dry_run)
        planned['plot_interactive'] = moved['plot_interactive'][1]
    # gallery (copy or move all found)
    planned_gallery = []
    for g in assets.get('gallery', [])[:]:  # copy list
        dst_g = os.path.join(experiment_dir, 'gallery', os.path.basename(g))
        moved_g = safe_move(g, dst_g, dry_run)
        planned_gallery.append(moved_g[1])
        moved.setdefault('gallery', []).append(moved_g)
    planned['gallery'] = planned_gallery
    # video
    if assets.get('video'):
        dst_vid = os.path.join(experiment_dir, 'video', os.path.basename(assets['video']))
        mv = safe_move(assets['video'], dst_vid, dry_run)
        moved['video'] = mv
        planned['video'] = mv[1]
    # tracking
    planned_tracking = []
    for t in assets.get('tracking', []):
        dst_t = os.path.join(experiment_dir, 'tracking', os.path.basename(t))
        mv = safe_move(t, dst_t, dry_run)
        planned_tracking.append(mv[1])
        moved.setdefault('tracking', []).append(mv)
    planned['tracking'] = planned_tracking

    # finally move the summary JSON itself
    mv_summary = safe_move(summary_file, os.path.join(experiment_dir, 'csv', os.path.basename(summary_file)), dry_run)
    moved['summary'] = mv_summary
    planned['summary'] = mv_summary[1]

    # If apply (not dry-run), update the JSON inside the moved summary to reference relative result paths
    if not dry_run:
        summary_new_path = moved['summary'][1]
        # Build new_paths absolute map for update
        new_paths = {}
        if 'timeseries' in moved:
            new_paths['timeseries'] = moved['timeseries'][1] if moved['timeseries'] else None
        if 'plot_static' in moved:
            new_paths['plot_static'] = moved['plot_static'][1] if moved['plot_static'] else None
        if 'plot_interactive' in moved:
            new_paths['plot_interactive'] = moved['plot_interactive'][1] if moved['plot_interactive'] else None
        if 'video' in moved:
            new_paths['video'] = moved['video'][1] if moved['video'] else None
        if 'gallery' in moved:
            new_paths['gallery'] = [m[1] for m in moved['gallery']]
        if 'tracking' in moved:
            new_paths['tracking'] = [m[1] for m in moved['tracking']]
        update_summary_paths(summary_new_path, new_paths, results_root=results_root)

    return {
        'experiment_dir': experiment_dir,
        'condition': condition,
        'experiment': experiment,
        'planned': planned
    }

def main():
    parser = argparse.ArgumentParser(description="Normalize results folder structure for WoundTrack AI")
    parser.add_argument('--root', type=str, default=None, help='Path to results root (optional).')
    parser.add_argument('--apply', action='store_true', help='Actually perform moves. Default: dry-run only.')
    args = parser.parse_args()

    # Candidate default paths (common in this project)
    cwd = os.getcwd()
    candidates = [
        args.root,
        os.path.join(cwd, 'results'),
        os.path.join(cwd, 'Scratch_Assay_Analysis', 'results'),
        os.path.join(cwd, 'Scratch_Assay_Analysis', 'Scratch_Assay_Analysis', 'results'),
        os.path.join(Path.home(), 'helloworld', 'woundhealinganalysis', 'Scratch_Assay_Analysis', 'results'),
        os.path.join(Path.home(), 'woundhealinganalysis', 'Scratch_Assay_Analysis', 'results'),
    ]
    results_root = find_results_root(candidates)
    if not results_root:
        print("Results root not found. Try passing --root /full/path/to/results")
        return

    print("Auto-discovered results root:", results_root)
    print("="*30)
    print("Dry-run:" , not args.apply)
    print()

    summary_files = sorted(list(Path(results_root).rglob('*_summary.json')))
    if not summary_files:
        print("No *_summary.json found under results root. Nothing to do.")
        return

    planned_all = []
    for sfile in summary_files:
        info = normalize_experiment_path(results_root, str(sfile), dry_run=not args.apply)
        planned_all.append(info)

    # Print summary
    print("\n=== Planned transformations ===\n")
    for info in planned_all:
        print(f"Condition: {info['condition']}, Experiment: {info['experiment']}")
        print(f"  -> canonical dir: {info['experiment_dir']}")
        p = info['planned']
        for k, v in p.items():
            if isinstance(v, list):
                if v:
                    print(f"    {k}:")
                    for item in v:
                        print(f"      - {item}")
            else:
                if v:
                    print(f"    {k}: {v}")
        print()

    if not args.apply:
        print("Dry-run complete. Re-run with --apply to perform moves.")
    else:
        print("Apply complete. Files moved and summary JSONs updated where possible.")

if __name__ == "__main__":
    main()
