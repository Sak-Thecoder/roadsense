"""Prepare an auditable road-quality split and record explicit StreetSurfaceVis approval.

Two sub-commands:
  prepare  - filter the audited images, remove duplicates, group photos taken close
             together (in place or time), and propose train/validation/test splits.
             Nothing is eligible for training yet.
  approve  - after a human has reviewed the label mapping and usage rights, mark the
             proposal as approved so training scripts can use it.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from contextlib import ExitStack
from io import StringIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from scipy.spatial import cKDTree

from backend.app.preprocess import CLASSES
from roadsense.audit import group_duplicates
from training.prepare_manifest import make_splits
from training.preprocess import validate_partitions

STREET = "streetSurfaceVis"
CAMBODIA = "RoadSense_Cambodian"
# StreetSurfaceVis uses 5 quality levels; we merge them into our 3 grades.
MAPPING = {"excellent": "Good", "good": "Good", "intermediate": "Moderate",
           "bad": "Poor", "very_bad": "Poor"}
SURFACES = ("asphalt", "concrete", "paving_stones", "sett", "unpaved")
VERSION = "https://doi.org/10.5281/zenodo.11449977"
LICENSE = "https://creativecommons.org/licenses/by-sa/4.0/"
EARTH_RADIUS_M = 6_371_000
# Photos closer than 400 m, or taken by the same person less than 60 s apart, count as one scene.
GEOGRAPHIC_DISTANCE_M = 400
SESSION_GAP_MS = 60_000
PENDING = "street_mapping_scope_and_usage_review_pending"


def read_rows(path):
    """Read a CSV file into a list of dictionaries (one per row)."""
    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def write_rows(path, rows):
    """Write rows to a new CSV file (mode "x" refuses to overwrite an existing file)."""
    with path.open("x", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def verify_image_hashes(rows):
    """Re-check every image's SHA-256 so the files are exactly the ones that were audited."""
    # ExitStack keeps every opened ZIP archive open until the end, then closes them all.
    with ExitStack() as stack:
        archives = {}
        for row in rows:
            path = Path(row["image_path"])
            if row["archive_member"]:
                if path not in archives:
                    archives[path] = stack.enter_context(ZipFile(path))
                digest = hashlib.sha256(archives[path].read(row["archive_member"])).hexdigest()
            else:
                digest = sha256(path)
            if (digest != row["sha256"] or row["read_status"] != "readable"
                    or len(row["dhash"]) != 16):  # dhash = 64-bit perceptual hash stored as 16 hex characters
                raise ValueError(f"Re-audit image bytes or metadata for {row['image_id']}.")


def street_metadata(rows, csv_path):
    """Copy surface type, GPS position, capture time, original rating and licence info from the
    official StreetSurfaceVis CSV onto each row, after checking the two match one-to-one."""
    source_rows = read_rows(csv_path)
    by_id = {r["mapillary_image_id"]: r for r in source_rows}
    if len(by_id) != len(source_rows) or set(by_id) != {r["original_id"] for r in rows}:
        raise ValueError("StreetSurfaceVis image IDs and source CSV do not match one-to-one.")
    for row in rows:
        identifier = row["original_id"]
        metadata = by_id[identifier]
        if json.loads(row["source_metadata"]) != metadata or Path(row["image_path"]).stem != identifier:
            raise ValueError(f"StreetSurfaceVis source metadata changed for {identifier}.")
        contributor = metadata["user_name"]
        row.update(original_split={"True": "train", "False": "test"}[metadata["train"]],
                   country="Germany (author dataset area)", surface_type=metadata["surface_type"],
                   latitude=metadata["latitude"], longitude=metadata["longitude"],
                   captured_at=metadata["captured_at"], original_label=metadata["surface_quality"],
                   proposed_quality_label=MAPPING[metadata["surface_quality"]],
                   verified_quality_label="", license=f"CC BY-SA 4.0 ({LICENSE})",
                   attribution=contributor, source_url=f"https://www.mapillary.com/app/?focus=photo&pKey={identifier}",
                   contributor_url=f"https://www.mapillary.com/app/user/{contributor}",
                   dataset_reference=VERSION)
        if row["surface_type"] not in SURFACES:
            raise ValueError(f"Unknown StreetSurfaceVis surface for {identifier}.")


def coordinates(rows):
    """Convert latitude/longitude to 3-D points on a unit sphere, so straight-line (chord)
    distance can be used for fast nearest-neighbour search."""
    degrees = np.array([[float(r["latitude"]), float(r["longitude"])] for r in rows])
    if not np.isfinite(degrees).all() or (np.abs(degrees) > [90, 180]).any():
        raise ValueError("Invalid StreetSurfaceVis coordinates.")
    lat, lon = np.deg2rad(degrees).T
    return np.column_stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


def group_street_locations(rows):
    """Give photos the same capture_group if they are within 400 m of each other, or were taken
    by the same contributor less than 60 seconds apart. Uses union-find to merge groups."""
    parent = list(range(len(rows)))

    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    # A k-d tree finds all pairs of points within a distance quickly.
    distances = cKDTree(coordinates(rows))
    # Convert 400 m along the Earth's surface into a straight-line distance on the unit sphere.
    chord = 2 * np.sin(GEOGRAPHIC_DISTANCE_M / (2 * EARTH_RADIUS_M))
    for a, b in distances.query_pairs(chord):
        parent[root(a)] = root(b)

    contributors = defaultdict(list)
    for index, row in enumerate(rows):
        user_id = json.loads(row["source_metadata"])["user_id"]
        contributors[user_id].append((int(row["captured_at"]), index))
    for captures in contributors.values():
        captures.sort()  # by capture time
        # Compare each photo with the next one by the same contributor.
        for (first, a), (second, b) in zip(captures, captures[1:]):
            if second - first <= SESSION_GAP_MS:
                parent[root(a)] = root(b)

    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[root(index)].append(row)
    for group in groups.values():
        name = min(r["original_id"] for r in group)
        for row in group:
            row["capture_group"] = f"street-location-session:{name}"


def cross_hash_pairs(left, right):
    """Yield pairs of images (one from each list) that look nearly identical.

    XOR of two perceptual hashes followed by bit_count() = how many of the 64 bits differ
    (Hamming distance). 4 or fewer differing bits means the images look almost the same.
    """
    hashes = [(row["original_id"], int(row["dhash"], 16)) for row in right]
    for row in left:
        value = int(row["dhash"], 16)
        for identifier, other in hashes:
            if (value ^ other).bit_count() <= 4:
                yield row["original_id"], identifier


def deduplicate_street(rows, reasons):
    """Exclude byte-identical copies. If the copies were rated differently, exclude all of them
    (the label is unreliable); otherwise keep one, preferring the copy in the author's test split."""
    by_hash = defaultdict(list)
    for row in rows:
        by_hash[row["sha256"]].append(row)
    for copies in by_hash.values():
        if len(copies) < 2:
            continue
        copies.sort(key=lambda r: (r["original_split"] != "test", r["image_id"]))
        if len({r["original_label"] for r in copies}) > 1:
            for row in copies:
                peers = ",".join(r["original_id"] for r in copies if r is not row)
                reasons[row["image_id"]] = f"identical_bytes_conflicting_original_ratings:{peers}"
        else:
            for row in copies[1:]:
                reasons[row["image_id"]] = f"redundant_exact_duplicate_of:{copies[0]['original_id']}"


def prepare(args):
    """The "prepare" command: build a proposed split for human review."""
    rows = sorted(read_rows(args.unified), key=lambda r: r["image_id"])
    by_id = {r["image_id"]: r for r in rows}
    if len(by_id) != len(rows):
        raise ValueError("Duplicate IDs in the unified audit.")
    pilot = {r["image_id"]: r["split"] for r in read_rows(args.pilot)}
    if not set(pilot) <= by_id.keys():
        raise ValueError("Pilot IDs missing from the unified audit.")
    street = [r for r in rows if r["source"] == STREET]
    street_metadata(street, args.street_csv)
    # image_id -> why it is excluded. Everything not listed here stays in the proposal.
    reasons = {}
    local = []
    for row in rows:
        source = row["source"]
        identifier = row["image_id"]
        if source == STREET:
            if row["surface_type"] not in args.surfaces:
                reasons[identifier] = "outside_selected_surface_scope"
        elif source == "RDD_SPLIT":
            reasons[identifier] = "no_verified_overall_quality_label|local_export_rights_unverified"
        elif source == CAMBODIA:
            origin = json.loads(row["source_metadata"])["source"]
            if origin != "self_collected" or row["permission_status"] != "confirmed":
                reasons[identifier] = "online_training_permission_unconfirmed"
                if row["verified_quality_label"] not in CLASSES:
                    reasons[identifier] += "|outside_three_class_labels"
                for reason in row["exclusion_reasons"].split("|"):
                    if reason.startswith("exact_duplicate_of_"):
                        reasons[identifier] += "|" + reason
            elif row["label_status"] != "reviewed" or row["verified_quality_label"] not in CLASSES:
                reasons[identifier] = "no_verified_overall_quality_label"
            else:
                row.update(capture_group="cambodian:all-authorized-self-collected",
                           split_policy="train_only")
                local.append(row)
        else:
            raise ValueError(f"Unexpected source {source}.")

    # ---- Remove duplicates and check for look-alike images across the original train/test ----
    verify_image_hashes(street + local)
    street = [r for r in street if r["image_id"] not in reasons]
    deduplicate_street(street, reasons)
    street = [r for r in street if r["image_id"] not in reasons]
    author_train = [r for r in street if r["original_split"] == "train"]
    author_test = [r for r in street if r["original_split"] == "test"]
    reviewed = {(r["train_id"], r["test_id"]) for r in read_rows(args.pair_review)
                if r["review_decision"] == "different_scene"}
    unresolved = set(cross_hash_pairs(author_train, author_test)) - reviewed
    if unresolved:
        raise ValueError(f"Review StreetSurfaceVis cross-test visual matches: {sorted(unresolved)}")
    if (list(cross_hash_pairs(local, street))
            or {r["sha256"] for r in local} & {r["sha256"] for r in street}):
        raise ValueError("Review Cambodian/StreetSurfaceVis duplicate candidates before partitioning.")

    # ---- Group near-duplicates and nearby photos so each group lands in one split ----
    for row in street + local:
        row["duplicate_group"] = ""
    for group in (author_train, author_test, local):
        group_duplicates(group)
    group_street_locations(street)

    # The in-memory proposed labels are used only to balance the partitions.
    assignments = make_splits(
        [dict(r, verified_quality_label=r["proposed_quality_label"]) for r in street] + local,
        args.seed, cambodia_min_groups=10)
    proposed = []
    counts = Counter()
    for assignment in assignments:
        row = by_id[assignment["image_id"]]
        split = assignment["split"]
        counts[f"{split}/{row['source']}/{assignment['verified_quality_label']}"] += 1
        proposed.append(dict(row, split=split, eligible="False",
                             exclusion_reasons=PENDING if row["source"] == STREET else "proposed_split"))

    # Every row gets every column (missing ones empty) so the CSV is rectangular.
    proposed.sort(key=lambda r: (r["split"], r["image_id"]))
    fields = list(dict.fromkeys(key for row in proposed for key in row))
    proposed = [{key: row.get(key, "") for key in fields} for row in proposed]
    splits = {r["image_id"]: r["split"] for r in proposed}
    street_by_split = {split: [r for r in proposed if r["source"] == STREET and r["split"] == split]
                       for split in ("train", "validation", "test")}
    # Report the closest distance (in metres) between photos of different splits, as a leakage check.
    minima = {}
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        chord, _ = cKDTree(coordinates(street_by_split[a])).query(coordinates(street_by_split[b]))
        minima[f"{a}/{b}"] = round(2 * EARTH_RADIUS_M * float(np.arcsin(chord.min() / 2)), 2)

    # image_decisions.csv: an audit trail explaining what happened to every image.
    ledger = []
    for row in rows:
        identifier = row["image_id"]
        if identifier in pilot:
            historical_reason = "later_exif_capture_date_reserved_for_test" if pilot[identifier] == "test" else ""
        elif row["source"] == CAMBODIA:
            historical_reason = "online_source_failed_pilot_self_collected_and_permission_filter"
        else:
            historical_reason = "source_outside_cambodian_pilot"
        ledger.append(dict(image_id=identifier, source=row["source"], original_id=row["original_id"],
                           source_split=row["original_split"], pilot_split=pilot.get(identifier, ""),
                           pilot_exclusion_from_training=historical_reason,
                           prior_full_manifest_exclusions=row["exclusion_reasons"],
                           proposed_split=splits.get(identifier, ""),
                           current_exclusion=reasons.get(identifier, PENDING if row["source"] == STREET else "")))

    args.output.mkdir(parents=True, exist_ok=False)
    manifest = args.output / "proposed_splits.csv"
    write_rows(manifest, proposed)
    write_rows(args.output / "image_decisions.csv", ledger)
    summary = dict(source_counts=dict(Counter(r["source"] for r in rows)), proposed_counts=dict(counts),
                   excluded_from_proposal=len(reasons), seed=args.seed, surfaces=args.surfaces,
                   minimum_geographic_distance_m=minima, street_source_csv_sha256=sha256(args.street_csv),
                   proposed_manifest_sha256=sha256(manifest), street_status=PENDING)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def approve(args):
    """The "approve" command: turn a reviewed proposal into a training-ready manifest.

    The approval JSON must contain the SHA-256 of the exact proposal file plus the
    reviewer's name and evidence, so an approval can't be reused for a changed proposal.
    """
    approval_bytes = args.approval.read_bytes()
    evidence = json.loads(approval_bytes)
    proposal_bytes = args.proposal.read_bytes()
    if (evidence.get("proposed_manifest_sha256") != hashlib.sha256(proposal_bytes).hexdigest()
            or evidence.get("mapping_and_surface_scope_approved") is not True
            or any(not str(evidence.get(key) or "").strip()
                   for key in ("reviewed_by", "mapping_approval_evidence", "training_usage_evidence"))):
        raise ValueError("Record the human mapping/scope approval and training-usage evidence for this proposal hash.")
    approval_hash = hashlib.sha256(approval_bytes).hexdigest()
    rows = list(csv.DictReader(StringIO(proposal_bytes.decode("utf-8-sig"), newline="")))
    verify_image_hashes(rows)
    for row in rows:
        if row["source"] == STREET and row["exclusion_reasons"] == PENDING:
            row.update(verified_quality_label=row["proposed_quality_label"], label_status="approved_mapping",
                       reviewer=evidence["reviewed_by"], permission_status="approved",
                       mapping_approval_evidence=evidence["mapping_approval_evidence"],
                       training_usage_evidence=evidence["training_usage_evidence"])
        elif not (row["source"] == CAMBODIA and row["split_policy"] == "train_only"
                  and row["split"] == "train" and row["permission_status"] == "confirmed"):
            raise ValueError(f"Unexpected approval candidate {row['image_id']}.")
        row.update(eligible="True", exclusion_reasons="", approval_record_sha256=approval_hash)
    validate_partitions(rows)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    write_rows(args.output, [{key: row.get(key, "") for key in fields} for row in rows])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    stage = commands.add_parser("prepare", help="Build partitions for review")
    for option in ("unified", "street-csv", "pilot", "pair-review", "output"):
        stage.add_argument(f"--{option}", type=Path, required=True)
    stage.add_argument("--seed", type=int, default=42)
    stage.add_argument("--surfaces", choices=SURFACES, nargs="+", default=list(SURFACES))
    promote = commands.add_parser("approve", help="Apply documented human approval to a proposal")
    for option in ("proposal", "approval", "output"):
        promote.add_argument(f"--{option}", type=Path, required=True)
    args = parser.parse_args()
    try:
        (prepare if args.command == "prepare" else approve)(args)
    except (KeyError, OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
