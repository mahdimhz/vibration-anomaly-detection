from pathlib import Path

import pandas as pd
from tqdm import tqdm


def load_femto_bearing(bearing_path: Path) -> pd.DataFrame:
    bearing_files = sorted(bearing_path.glob("acc_*.csv"))
    femto_columns = ["hour", "min", "sec", "microsec", "acc_h", "acc_v"]
    bearing_frames = []

    for file_idx, bearing_file in enumerate(bearing_files):
        with bearing_file.open("r", encoding="utf-8", errors="ignore") as sample_file:
            first_line = sample_file.readline()
        delimiter = ";" if first_line.count(";") > first_line.count(",") else ","
        bearing_frame = pd.read_csv(
            bearing_file,
            sep=delimiter,
            header=None,
            names=femto_columns,
        )
        bearing_frame["file_idx"] = file_idx
        bearing_frames.append(bearing_frame)

    return pd.concat(bearing_frames, ignore_index=True)


def load_all_femto_bearings(femto_dir: Path) -> dict[str, pd.DataFrame]:
    bearing_paths = sorted({csv_path.parent for csv_path in femto_dir.rglob("acc_*.csv")})
    bearing_frames = {}

    for bearing_path in tqdm(bearing_paths, desc="FEMTO bearings"):
        bearing_key = bearing_path.relative_to(femto_dir).as_posix().replace("/", "__")
        bearing_frames[bearing_key] = load_femto_bearing(bearing_path)

    return bearing_frames


def load_ims_run(run_path: Path) -> pd.DataFrame:
    ims_columns = [
        "b1_ch1",
        "b1_ch2",
        "b2_ch1",
        "b2_ch2",
        "b3_ch1",
        "b3_ch2",
        "b4_ch1",
        "b4_ch2",
    ]
    archive_suffixes = {".7z", ".zip", ".rar"}
    run_files = sorted(
        path for path in run_path.iterdir()
        if path.is_file() and path.suffix.lower() not in archive_suffixes
    )
    run_frames = []

    for file_idx, run_file in enumerate(run_files):
        run_frame = pd.read_csv(run_file, sep="\t", header=None, names=ims_columns)
        run_frame["file_idx"] = file_idx
        run_frames.append(run_frame)

    return pd.concat(run_frames, ignore_index=True)
