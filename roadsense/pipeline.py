"""Sample-first image indexing, auditing and leakage-aware splitting."""
import argparse
import csv
import hashlib
import json
import random
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageOps

FIELDS = ['image_id', 'original_path', 'source', 'damage_types', 'group_id', 'country',
          'quality_label', 'license', 'attribution', 'source_url', 'sha256', 'dhash']
LABELS = {'Good', 'Moderate', 'Poor', 'Uncertain'}
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}


def read_csv(path):
    with open(path, newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields=FIELDS):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def damage_types(xml_path):
    if not xml_path.exists():
        return []
    root = ET.parse(xml_path).getroot()
    return sorted({(node.text or '').strip() for node in root.findall('.//object/name')
                   if (node.text or '').strip()})


def fingerprints(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert('L').resize((9, 8))
        pixels = list(im.getdata())
        value = 0
        for y in range(8):
            for x in range(8):
                value = (value << 1) | (pixels[y * 9 + x] > pixels[y * 9 + x + 1])
    return digest.hexdigest(), f'{value:016x}'


def index(args):
    root = Path(args.root).resolve()
    if not root.is_dir():
        raise ValueError('Source root must be an existing directory')
    mapping = {}
    if args.manifest:
        for row in read_csv(args.manifest):
            key = str(Path(row['path']).expanduser())
            if key in mapping:
                raise ValueError(f'Duplicate manifest path: {key}')
            mapping[key] = row
    paths = sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if not args.all:
        paths = paths[:args.limit]
    rows, errors = [], []
    for path in paths:
        rel = str(path.relative_to(root))
        meta = mapping.get(rel, mapping.get(str(path), {}))
        try:
            sha, dhash = fingerprints(path)
        except (OSError, ValueError) as exc:
            errors.append({'path': str(path), 'error': str(exc)})
            continue
        annotation = Path(args.annotations).resolve() / Path(rel).with_suffix('.xml') if args.annotations else path.with_suffix('.xml')
        if args.annotations and not annotation.exists():
            errors.append({'path': str(annotation), 'error': 'annotation missing'})
        try:
            damage = damage_types(annotation)
        except (ET.ParseError, OSError) as exc:
            errors.append({'path': str(annotation), 'error': f'annotation: {exc}'})
            damage = []
        label = meta.get('quality_label', '').strip()
        if label and label not in LABELS:
            raise ValueError(f'Invalid quality label for {rel}: {label}')
        rows.append(dict(image_id=hashlib.sha256((args.source + '\0' + rel).encode()).hexdigest()[:20],
                         original_path=str(path), source=args.source, damage_types='|'.join(damage),
                         group_id=meta.get('group_id', '').strip(), country=meta.get('country', '').strip(),
                         quality_label=label, license=meta.get('license', ''), attribution=meta.get('attribution', ''),
                         source_url=meta.get('source_url', ''), sha256=sha, dhash=dhash))
    if not args.dry_run:
        write_csv(args.output, rows)
        write_csv(args.errors, errors, ['path', 'error'])
    print(json.dumps({'discovered': len(paths), 'indexed': len(rows), 'errors': errors,
                      'sample_mode': not args.all, 'dry_run': args.dry_run}, indent=2))


def audit_rows(rows, threshold):
    exact, near = [], []
    # Pairwise comparison is intentionally transparent; use a smaller sample first.
    for i, a in enumerate(rows):
        for b in rows[i + 1:]:
            if a.get('sha256') and a['sha256'] == b.get('sha256'):
                exact.append([a['image_id'], b['image_id']])
            elif a.get('dhash') and b.get('dhash') and (int(a['dhash'], 16) ^ int(b['dhash'], 16)).bit_count() <= threshold:
                near.append([a['image_id'], b['image_id']])
    return {'count': len(rows), 'sources': dict(Counter(r['source'] for r in rows)),
            'labels': dict(Counter(r.get('quality_label') or 'Unlabeled' for r in rows)),
            'exact_duplicates': exact, 'near_duplicates': near, 'near_threshold': threshold}


def audit(args):
    report = audit_rows(read_csv(args.input), args.threshold)
    if not args.dry_run:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


def split_rows(rows, seed, holdout_fraction, threshold):
    if not 0 < holdout_fraction < 1:
        raise ValueError('Holdout fraction must be between 0 and 1')
    ids = [r['image_id'] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Image IDs must be unique; check combined indexes')
    for r in rows:
        if r.get('quality_label') and r['quality_label'] not in LABELS:
            raise ValueError(f"Invalid label: {r['quality_label']}")
    # Union source-scoped groups and duplicate pairs, including unlabeled rows.
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[b] = a
    seen = {}
    for i, r in enumerate(rows):
        if r.get('group_id'):
            key = (r['source'], r['group_id'])
            if key in seen:
                union(i, seen[key])
            else:
                seen[key] = i
    report = audit_rows(rows, threshold)
    positions = {r['image_id']: i for i, r in enumerate(rows)}
    for a, b in report['exact_duplicates'] + report['near_duplicates']:
        union(positions[a], positions[b])
    components = defaultdict(list)
    for i, row in enumerate(rows):
        components[find(i)].append(row)
    groups = list(components.values())
    rng = random.Random(seed)
    rng.shuffle(groups)
    # Holdout components must contain only explicitly Cambodian images.
    candidates = [g for g in groups if all(r.get('country', '').strip().lower() in ('cambodia', 'kh') for r in g)
                  and any(r.get('quality_label') in {'Good', 'Moderate', 'Poor'} for r in g)]
    if len(candidates) < 2:
        raise ValueError('Need at least two independent Cambodian groups with usable labels for a holdout')
    target = max(1, round(sum(len(g) for g in candidates) * holdout_fraction))
    chosen, count = [], 0
    for g in candidates[:-1]:
        if count >= target:
            break
        chosen.append(g)
        count += len(g)
    chosen_ids = {id(g) for g in chosen}
    remaining = [g for g in groups if id(g) not in chosen_ids]
    usable = lambda g: sum(r.get('quality_label') in {'Good', 'Moderate', 'Poor'} for r in g)
    totals = sum(usable(g) for g in remaining)
    targets = {'train': totals * .70, 'validation': totals * .15, 'test': totals * .15}
    counts = dict.fromkeys(targets, 0)
    assignment = {}
    # Largest groups first; deterministic random shuffle breaks equal-size ties.
    for g in sorted(remaining, key=usable, reverse=True):
        split = max(targets, key=lambda k: (targets[k] - counts[k]) / max(targets[k], 1))
        counts[split] += usable(g)
        for r in g:
            assignment[r['image_id']] = split
    for g in chosen:
        for r in g:
            assignment[r['image_id']] = 'cambodia_holdout'
    return [dict(r, split=assignment[r['image_id']], eligible=r.get('quality_label') in {'Good', 'Moderate', 'Poor'}) for r in rows]


def split(args):
    rows = read_csv(args.input)
    result = split_rows(rows, args.seed, args.holdout_fraction, args.threshold)
    if not args.dry_run:
        write_csv(args.output, result, FIELDS + ['split', 'eligible'])
    print(json.dumps({'rows': len(result), 'assigned': dict(Counter(r['split'] for r in result)),
                      'eligible': dict(Counter(r['split'] for r in result if r['eligible'])),
                      'distributions': {name: {'sources': dict(Counter(r['source'] for r in result if r['split'] == name)),
                                               'labels': dict(Counter(r.get('quality_label') or 'Unlabeled' for r in result if r['split'] == name))}
                                        for name in ('train', 'validation', 'test', 'cambodia_holdout')},
                      'missing_group': sum(not r.get('group_id') for r in rows), 'dry_run': args.dry_run}, indent=2))


def combine(args):
    rows = [row for path in args.inputs for row in read_csv(path)]
    if len({r['image_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate image IDs in inputs')
    if not args.dry_run:
        write_csv(args.output, rows)
    print(json.dumps({'rows': len(rows), 'dry_run': args.dry_run}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('index')
    p.add_argument('--root', required=True); p.add_argument('--source', required=True)
    p.add_argument('--annotations', help='XML root mirroring relative image paths; otherwise adjacent XML')
    p.add_argument('--manifest'); p.add_argument('--output', required=True); p.add_argument('--errors', required=True)
    p.add_argument('--limit', type=int, default=100); p.add_argument('--all', action='store_true')
    p.add_argument('--dry-run', action='store_true'); p.set_defaults(func=index)
    p = sub.add_parser('combine'); p.add_argument('inputs', nargs='+'); p.add_argument('--output', required=True)
    p.add_argument('--dry-run', action='store_true'); p.set_defaults(func=combine)
    p = sub.add_parser('audit'); p.add_argument('--input', required=True); p.add_argument('--output', required=True)
    p.add_argument('--threshold', type=int, default=4); p.add_argument('--dry-run', action='store_true'); p.set_defaults(func=audit)
    p = sub.add_parser('split'); p.add_argument('--input', required=True); p.add_argument('--output', required=True)
    p.add_argument('--seed', type=int, default=42); p.add_argument('--holdout-fraction', type=float, default=.2)
    p.add_argument('--threshold', type=int, default=4); p.add_argument('--dry-run', action='store_true'); p.set_defaults(func=split)
    args = parser.parse_args()
    if getattr(args, 'limit', 1) < 1 or not 0 <= getattr(args, 'threshold', 0) <= 64:
        parser.error('limit must be positive and threshold must be between 0 and 64')
    try:
        args.func(args)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
