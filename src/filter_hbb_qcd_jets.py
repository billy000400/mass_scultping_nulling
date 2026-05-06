#!/usr/bin/env python3
"""
Filter Hbb and QCD jets by sd_mass range from ROOT files.

Splits jets into two outputs:
- 75_175: sd_mass in [75, 175] GeV
- outside_75_175: sd_mass outside that range

Reads HToBB and ZJetsToNuNu (QCD) files; preserves original filenames in each subdir.

Usage:
    python filter_hbb_qcd_jets.py --input-dir /path/to/input --output-dir /path/to/output
    python filter_hbb_qcd_jets.py  # Uses default paths

Output layout:
    <output-dir>/75_175/<HToBB_*.root, ZJetsToNuNu_*.root>
    <output-dir>/outside_75_175/<HToBB_*.root, ZJetsToNuNu_*.root>
"""

import argparse
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import uproot
import awkward as ak


def _arrays_to_branches(arrays) -> dict:
    """Convert awkward arrays to dict of branches for uproot."""
    branches = {}
    for field in ak.fields(arrays):
        arr = arrays[field]
        if arr.ndim > 1:
            branches[field] = ak.to_list(arr)
        else:
            branches[field] = ak.to_numpy(arr)
    return branches


def _write_tree(path: str, branches: dict) -> None:
    """Create parent dirs and write ROOT file with tree."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with uproot.recreate(path) as out_file:
        out_file["tree"] = branches


def filter_and_save_file(
    input_path: str,
    output_dir: str,
    filename: str,
    mass_min: float = 75.0,
    mass_max: float = 175.0,
) -> dict:
    """
    Filter a single ROOT file by sd_mass range and save to two outputs:
    - 75_175: jets with sd_mass in [mass_min, mass_max]
    - outside_75_175: jets with sd_mass outside that range

    Returns:
        dict with statistics about the filtering
    """
    output_path_75_175 = os.path.join(output_dir, "75_175", filename)
    output_path_outside = os.path.join(output_dir, "outside_75_175", filename)
    try:
        with uproot.open(input_path) as f:
            tree = f["tree"]
            arrays = tree.arrays(library="ak")
            sd_mass = ak.to_numpy(arrays["jet_sdmass"])

        mask_in = (sd_mass >= mass_min) & (sd_mass <= mass_max)
        mask_out = ~mask_in
        n_total = len(sd_mass)
        n_75_175 = int(mask_in.sum())
        n_outside = int(mask_out.sum())

        if n_75_175 > 0:
            filtered_in = arrays[mask_in]
            _write_tree(output_path_75_175, _arrays_to_branches(filtered_in))
        if n_outside > 0:
            filtered_out = arrays[mask_out]
            _write_tree(output_path_outside, _arrays_to_branches(filtered_out))

        return {
            "input": input_path,
            "output_75_175": output_path_75_175,
            "output_outside": output_path_outside,
            "n_total": n_total,
            "n_75_175": n_75_175,
            "n_outside": n_outside,
            "status": "success",
        }

    except Exception as e:
        return {
            "input": input_path,
            "output_75_175": output_path_75_175,
            "output_outside": output_path_outside,
            "n_total": 0,
            "n_75_175": 0,
            "n_outside": 0,
            "status": f"error: {str(e)}",
        }


def main():
    parser = argparse.ArgumentParser(
        description="Filter Hbb and QCD jets by sd_mass range"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="/scope-vol/JetClass/Pythia/train_100M",
        help="Input directory containing ROOT files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/scope-vol/JetClass/Pythia/HbbvsQCD",
        help="Base output directory; 75_175/ and outside_75_175/ subdirs are created",
    )
    parser.add_argument(
        "--mass-min",
        type=float,
        default=75.0,
        help="Minimum sd_mass (GeV)",
    )
    parser.add_argument(
        "--mass-max",
        type=float,
        default=175.0,
        help="Maximum sd_mass (GeV)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Find Hbb (HToBB) and QCD (ZJetsToNuNu) files
    hbb_files = sorted(input_dir.glob("HToBB_*.root"))
    qcd_files = sorted(input_dir.glob("ZJetsToNuNu_*.root"))

    print(f"Found {len(hbb_files)} HToBB files and {len(qcd_files)} ZJetsToNuNu (QCD) files")
    print(f"Mass range: [{args.mass_min}, {args.mass_max}] GeV")
    print(f"Output directory: {output_dir}")
    print()

    # Create output subdirectories
    (output_dir / "75_175").mkdir(parents=True, exist_ok=True)
    (output_dir / "outside_75_175").mkdir(parents=True, exist_ok=True)

    # Prepare tasks: (input_path, output_dir, filename)
    tasks = []
    for f in hbb_files:
        tasks.append((str(f), str(output_dir), f.name))
    for f in qcd_files:
        tasks.append((str(f), str(output_dir), f.name))

    # Process files in parallel
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                filter_and_save_file,
                inp,
                out_dir,
                name,
                args.mass_min,
                args.mass_max,
            ): (inp, out_dir, name)
            for inp, out_dir, name in tasks
        }

        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            results.append(result)

            status = result["status"]
            if status == "success":
                print(
                    f"[{i+1}/{len(tasks)}] {Path(result['input']).name}: "
                    f"75_175={result['n_75_175']:,}, outside={result['n_outside']:,} / {result['n_total']:,} jets"
                )
            else:
                print(f"[{i+1}/{len(tasks)}] {Path(result['input']).name}: {status}")

    # Summary
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)

    hbb_total = sum(r["n_total"] for r in results if "HToBB" in r["input"])
    hbb_75_175 = sum(r["n_75_175"] for r in results if "HToBB" in r["input"])
    hbb_outside = sum(r["n_outside"] for r in results if "HToBB" in r["input"])
    qcd_total = sum(r["n_total"] for r in results if "ZJetsToNuNu" in r["input"])
    qcd_75_175 = sum(r["n_75_175"] for r in results if "ZJetsToNuNu" in r["input"])
    qcd_outside = sum(r["n_outside"] for r in results if "ZJetsToNuNu" in r["input"])

    print("75_175 (sd_mass in [75, 175] GeV):")
    print(f"  HToBB (Hbb):      {hbb_75_175:,} / {hbb_total:,} jets ({100*hbb_75_175/hbb_total:.1f}%)")
    print(f"  ZJetsToNuNu (QCD): {qcd_75_175:,} / {qcd_total:,} jets ({100*qcd_75_175/qcd_total:.1f}%)")
    print("outside_75_175:")
    print(f"  HToBB (Hbb):      {hbb_outside:,} / {hbb_total:,} jets ({100*hbb_outside/hbb_total:.1f}%)")
    print(f"  ZJetsToNuNu (QCD): {qcd_outside:,} / {qcd_total:,} jets ({100*qcd_outside/qcd_total:.1f}%)")
    print()
    print(f"  Total 75_175:     {hbb_75_175 + qcd_75_175:,} jets -> {output_dir / '75_175'}")
    print(f"  Total outside:    {hbb_outside + qcd_outside:,} jets -> {output_dir / 'outside_75_175'}")


if __name__ == "__main__":
    main()
