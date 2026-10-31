#!/usr/bin/env python3


import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLUMN_CANDIDATES = {
    "x": ["locx", "x", "posx", "xloc", "xpos"],
    "y": ["locy", "y", "posy", "yloc", "ypos"],
    "z": ["locz", "z", "posz", "zloc", "zpos"],
    "cx": ["cosx", "cx", "dirx", "xcos"],
    "cy": ["cosy", "cy", "diry", "ycos"],
    "cz": ["cosz", "cz", "dirz", "zcos"],
    "element": ["element", "elem", "elementnumber", "elementnum"],
    "stage": ["stage", "stagenumber", "stagenum"],
    "ray": ["raynumber", "raynum", "ray", "number", "id"],
}
REQUIRED = ["x", "y", "z"]


def norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def load_csv(path: Path) -> pd.DataFrame:
    """Load CSV, auto-detecting separator and skipping junk header lines."""
    # Find the first line that looks like a header (contains letters and a separator)
    with open(path, "r", errors="replace") as f:
        lines = f.readlines()
    header_idx = 0
    for i, line in enumerate(lines[:30]):
        if re.search(r"[A-Za-z]", line) and re.search(r"[,;\t]", line):
            header_idx = i
            break
    df = pd.read_csv(path, sep=None, engine="python", skiprows=header_idx)
    df.columns = [str(c).strip().lstrip("#").strip() for c in df.columns]
    df = df.dropna(how="all")
    return df


def map_columns(df: pd.DataFrame, overrides: dict) -> dict:
    lookup = {norm(c): c for c in df.columns}
    mapping = {}
    for key, cands in COLUMN_CANDIDATES.items():
        if overrides.get(key):
            if overrides[key] not in df.columns:
                sys.exit(f"Column '{overrides[key]}' (for {key}) not found. "
                         f"Available: {list(df.columns)}")
            mapping[key] = overrides[key]
            continue
        for cand in cands:
            if cand in lookup:
                mapping[key] = lookup[cand]
                break
    missing = [k for k in REQUIRED if k not in mapping]
    if missing:
        sys.exit(f"Could not detect columns {missing}. Available: {list(df.columns)}\n"
                 f"Use --col-x / --col-y / --col-z to name them explicitly.")
    return mapping


def build_standard_frame(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    out = pd.DataFrame({k: pd.to_numeric(df[v], errors="coerce") for k, v in mapping.items()})
    out = out.dropna(subset=REQUIRED)
    for k in ("element", "stage"):
        if k in out:
            out[k] = out[k].abs().astype(int)  # SolTrace uses a sign for front/back hits
    return out


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #
def select_receiver(d: pd.DataFrame, stage, element) -> pd.DataFrame:
    """Rays hitting the receiver. Defaults to the last stage if none is given."""
    if "stage" not in d:
        print("[warn] no 'stage' column -> using ALL rays as receiver rays")
        return d
    if stage is None:
        stage = int(d["stage"].max())
    sel = d[d["stage"] == stage]
    if element is not None and "element" in d:
        sel = sel[sel["element"] == element]
    print(f"Receiver selection: stage={stage}, element={element if element is not None else 'all'} "
          f"-> {len(sel)} hits")
    return sel


def print_summary(d: pd.DataFrame, rec: pd.DataFrame, args) -> None:
    print("\n=== GENERAL SUMMARY ===")
    print(f"Rows (ray hits) read : {len(d)}")
    print(d[["x", "y", "z"]].describe().round(5).to_string())

    if {"stage", "element"} <= set(d.columns):
        print("\n=== HITS PER STAGE / ELEMENT ===")
        tally = d.groupby(["stage", "element"]).size().rename("hits")
        print(tally.to_string())

    print("\n=== RECEIVER STATISTICS ===")
    if rec.empty:
        print("No receiver hits found - check --receiver-stage / --receiver-element.")
        return
    print(f"Hits on receiver     : {len(rec)}")
    print(f"Mean position (x,y,z): {rec.x.mean():.5f}, {rec.y.mean():.5f}, {rec.z.mean():.5f}")
    print(f"Std  position (x,y,z): {rec.x.std():.5f}, {rec.y.std():.5f}, {rec.z.std():.5f}")

    if args.nsun:
        eff = len(rec) / args.nsun
        print(f"Optical efficiency (hits / launched sun rays = {args.nsun}): {eff:.4f}")
        if args.dni and args.mirror_area:
            # each ray carries the same power: P_total = DNI * aperture area
            p_ray = args.dni * args.mirror_area / args.nsun
            print(f"Power per ray   : {p_ray:.6f} W")
            print(f"Receiver power  : {p_ray * len(rec):.3f} W")
    else:
        print("(give --nsun to get an optical efficiency)")


def add_cylinder_angle(rec: pd.DataFrame, args) -> pd.DataFrame:
    """Angle around a tube whose axis is parallel to Y, centred at (x0, z0)."""
    rec = rec.copy()
    x0 = args.tube_x if args.tube_x is not None else rec.x.mean()
    z0 = args.tube_z if args.tube_z is not None else rec.z.mean()
    # 0 deg = top (+z), 180 deg = bottom (-z, facing the mirror)
    rec["theta_deg"] = np.degrees(np.arctan2(rec.x - x0, rec.z - z0)) % 360
    rec["r"] = np.hypot(rec.x - x0, rec.z - z0)
    return rec



def save(fig, outdir: Path, name: str, show: bool):
    fig.tight_layout()
    fig.savefig(outdir / name, dpi=150)
    if not show:
        plt.close(fig)


def make_plots(d: pd.DataFrame, rec: pd.DataFrame, args, outdir: Path):
    show = args.show
    # 1) Hits per stage/element
    if {"stage", "element"} <= set(d.columns):
        tally = d.groupby(["stage", "element"]).size()
        fig, ax = plt.subplots(figsize=(7, 4))
        labels = [f"S{s}-E{e}" for s, e in tally.index]
        ax.bar(labels, tally.values, color="tab:blue")
        ax.set_ylabel("Number of ray hits")
        ax.set_title("Ray hits per stage / element")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        save(fig, outdir, "01_hits_per_element.png", show)

    # 2) All hits, side view (x-z), coloured by stage
    fig, ax = plt.subplots(figsize=(7, 5))
    sample = d.sample(min(len(d), 50000), random_state=0)
    if "stage" in sample:
        sc = ax.scatter(sample.x, sample.z, c=sample.stage, s=1, cmap="viridis")
        fig.colorbar(sc, ax=ax, label="Stage")
    else:
        ax.scatter(sample.x, sample.z, s=1)
    ax.set_xlabel("x"); ax.set_ylabel("z"); ax.set_aspect("equal", "datalim")
    ax.set_title("Ray hit locations, cross-section (x-z)")
    save(fig, outdir, "02_all_hits_xz.png", show)

    if rec.empty:
        return

    # 3) Receiver hit map (x-y) + 2D flux map
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].scatter(rec.x, rec.y, s=1, alpha=0.4)
    axes[0].set_xlabel("x"); axes[0].set_ylabel("y"); axes[0].set_title("Receiver hits (x-y)")
    h = axes[1].hist2d(rec.x, rec.y, bins=args.bins, cmap="inferno")
    fig.colorbar(h[3], ax=axes[1], label="Hits per bin")
    axes[1].set_xlabel("x"); axes[1].set_ylabel("y"); axes[1].set_title("Receiver flux map (x-y)")
    save(fig, outdir, "03_receiver_map.png", show)

    # 4) Distributions of x, y, z on receiver
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, col in zip(axes, ["x", "y", "z"]):
        ax.hist(rec[col], bins=args.bins, color="tab:orange", edgecolor="k", linewidth=0.2)
        ax.axvline(rec[col].mean(), color="r", ls="--", label=f"mean={rec[col].mean():.4f}")
        ax.set_xlabel(col); ax.set_ylabel("Hits"); ax.legend()
    fig.suptitle("Receiver hit distributions")
    save(fig, outdir, "04_receiver_distributions.png", show)

    # 5) Circumferential flux profile (for a tube receiver)
    rc = add_cylinder_angle(rec, args)
    fig = plt.figure(figsize=(12, 4.5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.hist(rc.theta_deg, bins=args.bins, color="tab:red", edgecolor="k", linewidth=0.2)
    ax1.set_xlabel("Angle around tube (deg, 0 = top, 180 = bottom)")
    ax1.set_ylabel("Hits"); ax1.set_title("Circumferential flux profile")
    ax2 = fig.add_subplot(1, 2, 2, projection="polar")
    counts, edges = np.histogram(np.radians(rc.theta_deg), bins=72, range=(0, 2 * np.pi))
    ax2.bar(edges[:-1], counts, width=np.diff(edges), align="edge", color="tab:red", alpha=0.8)
    ax2.set_theta_zero_location("N"); ax2.set_theta_direction(-1)
    ax2.set_title("Polar flux profile", pad=15)
    save(fig, outdir, "05_circumferential_flux.png", show)

    # 6) Unrolled flux map (angle vs y) - nice for tube receivers
    fig, ax = plt.subplots(figsize=(8, 4.5))
    h = ax.hist2d(rc.theta_deg, rc.y, bins=args.bins, cmap="inferno")
    fig.colorbar(h[3], ax=ax, label="Hits per bin")
    ax.set_xlabel("Angle around tube (deg)"); ax.set_ylabel("y (along tube)")
    ax.set_title("Unrolled tube flux map")
    save(fig, outdir, "06_unrolled_flux_map.png", show)

    # 7) Direction cosines at the receiver (incidence angle)
    if {"cx", "cy", "cz"} <= set(rec.columns):
        rcos = rec.dropna(subset=["cx", "cy", "cz"])
        # angle between the ray and the inward radial direction of the tube
        x0 = args.tube_x if args.tube_x is not None else rec.x.mean()
        z0 = args.tube_z if args.tube_z is not None else rec.z.mean()
        nx = -(rcos.x - x0); nz = -(rcos.z - z0)
        norm_n = np.hypot(nx, nz).replace(0, np.nan)
        cos_inc = ((rcos.cx * nx + rcos.cz * nz) / norm_n).clip(-1, 1)
        inc_deg = np.degrees(np.arccos(cos_inc))
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].hist(inc_deg.dropna(), bins=args.bins, color="tab:green", edgecolor="k", linewidth=0.2)
        axes[0].set_xlabel("Incidence angle on tube surface (deg)"); axes[0].set_ylabel("Hits")
        axes[0].set_title("Incidence angle distribution")
        axes[1].hist(rcos.cz, bins=args.bins, color="tab:purple", edgecolor="k", linewidth=0.2)
        axes[1].set_xlabel("cos z (direction)"); axes[1].set_ylabel("Hits")
        axes[1].set_title("Direction cosine z at receiver")
        save(fig, outdir, "07_incidence_angles.png", show)

    # 8) Cumulative profile along tube (y) - uniformity check
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(rec.y, bins=args.bins, color="tab:cyan", edgecolor="k", linewidth=0.2)
    ax.set_xlabel("y (along tube)"); ax.set_ylabel("Hits")
    ax.set_title(f"Axial flux profile (non-uniformity: "
                 f"{(np.histogram(rec.y, bins=args.bins)[0].std() / max(np.histogram(rec.y, bins=args.bins)[0].mean(), 1e-12)) * 100:.1f} % CV)")
    save(fig, outdir, "08_axial_profile.png", show)


# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="Analyse and plot a SolTrace ray-data CSV.")
    p.add_argument("csv", type=Path, help="Path to the SolTrace CSV export")
    p.add_argument("--outdir", type=Path, default=Path("soltrace_results"))
    p.add_argument("--receiver-stage", type=int, default=None, help="Stage number of the receiver (default: last stage)")
    p.add_argument("--receiver-element", type=int, default=None, help="Element number of the receiver in that stage")
    p.add_argument("--nsun", type=int, default=None, help="Number of sun rays launched (for efficiency)")
    p.add_argument("--dni", type=float, default=None, help="Direct normal irradiance, W/m2")
    p.add_argument("--mirror-area", type=float, default=None, help="Aperture area, m2")
    p.add_argument("--tube-x", type=float, default=None, help="Tube centre x (default: mean of hits)")
    p.add_argument("--tube-z", type=float, default=None, help="Tube centre z (default: mean of hits)")
    p.add_argument("--bins", type=int, default=60)
    p.add_argument("--show", action="store_true", help="Display plots interactively")
    for k in COLUMN_CANDIDATES:
        p.add_argument(f"--col-{k}", dest=f"col_{k}", default=None, help=f"CSV column name for {k}")
    return p.parse_args()


def main():
    args = parse_args()
    if not args.csv.exists():
        sys.exit(f"File not found: {args.csv}")
    args.outdir.mkdir(parents=True, exist_ok=True)

    raw = load_csv(args.csv)
    print(f"Loaded {len(raw)} rows, columns: {list(raw.columns)}")
    overrides = {k: getattr(args, f"col_{k}") for k in COLUMN_CANDIDATES}
    mapping = map_columns(raw, overrides)
    print("Column mapping:", mapping)

    d = build_standard_frame(raw, mapping)
    rec = select_receiver(d, args.receiver_stage, args.receiver_element)

    print_summary(d, rec, args)
    make_plots(d, rec, args, args.outdir)

    if not rec.empty:
        add_cylinder_angle(rec, args).to_csv(args.outdir / "receiver_hits.csv", index=False)
    print(f"\nDone. Figures and receiver_hits.csv saved in: {args.outdir.resolve()}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()