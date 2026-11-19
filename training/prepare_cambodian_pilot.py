"""Prepare the confirmed Cambodian photos for a two-session local pilot."""
import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from training.preprocess import PILOT_SCOPE, candidates, group_keys


def pilot_rows(manifest):
    """Keep the 50 permitted self-collected Cambodian photos and split them by capture date.

    Photos from the first date become "train", photos from the second date "test".
    Splitting by date (instead of randomly) prevents near-identical shots taken
    minutes apart from ending up on both sides.
    """
    with manifest.open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    by_date = defaultdict(list)
    for row in rows:
        if row["source"] != "RoadSense_Cambodian":
            continue
        origin = json.loads(row["source_metadata"])
        if origin.get("source") != "self_collected" or row["permission_status"] != "confirmed":
            continue
        if row["label_status"] != "reviewed" or row.get("duplicate_group"):
            raise ValueError(f"Review or perceptual match needs attention: {row['image_id']}")
        # EXIF stores the camera time as "YYYY:MM:DD HH:MM:SS".
        captured_at = datetime.strptime(origin["exif_datetime"], "%Y:%m:%d %H:%M:%S")
        capture_date = captured_at.date().isoformat()
        if not Path(row["archive_member"]).name.startswith(captured_at.strftime("%Y%m%d_")):
            raise ValueError(f"Filename disagrees with capture date: {row['image_id']}")
        by_date[capture_date].append(dict(row,
            capture_group=f"cambodian-camera-date:{capture_date}", eligible="True",
            exclusion_reasons="", capture_group_basis="EXIF date; geographic independence unverified",
            pilot_scope=PILOT_SCOPE))
    if len(by_date) != 2 or sum(map(len, by_date.values())) != 50:
        raise ValueError("Expected the 50 authorized photos and two documented capture dates.")
    owners = {}
    output = []
    # Earlier date -> train, later date -> test.
    for split, date in zip(("train", "test"), sorted(by_date)):
        for row in sorted(by_date[date], key=lambda item: item["image_id"]):
            for key in group_keys(row):
                if owners.setdefault(key, split) != split:
                    raise ValueError(f"Capture or duplicate group overlaps dates: {key}")
            output.append(dict(row, split=split))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = pilot_rows(args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # Re-read the written file with the normal checks to make sure it is valid.
    candidates(args.output)
    print(Counter((row["split"], row["verified_quality_label"]) for row in rows))


if __name__ == "__main__":
    main()
