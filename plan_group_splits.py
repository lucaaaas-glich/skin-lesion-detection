"""Prepare supplementary split CANDIDATES from recorded IDs, without training.

All known patient/lesion links form connected components before cohort filtering.
An assignment component is a constraint, NOT an imputed patient or lesion ID.
Missing IDs remain blank. No source images, labels, historical splits or weights
are changed. A regrouped reassessment of previously inspected data is not blind.
"""

import argparse
import csv
import hashlib
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

CLASSES = ('Benign', 'Malignant', 'Indeterminate')
SPLITS = ('train', 'val', 'reassessment')
FRACTIONS = (0.70, 0.15, 0.15)
COHORT_FIELDS = {'known_lesion': 'lesion_id', 'known_patient': 'patient_id',
                 'all_known_links': None}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_metadata(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        required = {'isic_id', 'diagnosis_1', 'lesion_id', 'patient_id'}
        if not required <= set(reader.fieldnames or []):
            raise ValueError('Metadata must contain image, diagnosis and both ID fields')
        rows = list(reader)
    meta = {}
    for row in rows:
        row = {key: (value or '').strip() for key, value in row.items()}
        identifier = row['isic_id']
        if not identifier or identifier in meta or row['diagnosis_1'] not in CLASSES:
            raise ValueError('Missing/duplicate image ID or unsupported diagnosis')
        for field in ('lesion_id', 'patient_id'):
            if row[field].lower() in {'nan', 'none', 'null', 'unknown', 'n/a'}:
                raise ValueError('Ambiguous missing-ID encoding; resolve the source first')
        meta[identifier] = row
    if not meta:
        raise ValueError('Empty metadata')
    return meta


def connected_components(meta):
    """Union actual equality links in separate patient and lesion namespaces."""
    parent = {identifier: identifier for identifier in meta}

    def find(identifier):
        while parent[identifier] != identifier:
            parent[identifier] = parent[parent[identifier]]
            identifier = parent[identifier]
        return identifier

    first = {}
    for identifier in sorted(meta):
        for field in ('patient_id', 'lesion_id'):
            value = meta[identifier].get(field, '')
            if value:
                key = (field, value)
                if key in first:
                    a, b = sorted((find(identifier), find(first[key])))
                    parent[b] = a
                else:
                    first[key] = identifier
    groups = defaultdict(list)
    for identifier in sorted(meta):
        groups[find(identifier)].append(identifier)
    return list(groups.values())


def historical_membership(directory, meta):
    membership = {}
    for split in ('train', 'val', 'test'):
        with (Path(directory) / f'{split}.csv').open(newline='', encoding='utf-8-sig') as stream:
            for row in csv.DictReader(stream):
                identifier = row['image_id']
                if identifier in membership or identifier not in meta:
                    raise ValueError('Historical split duplicate or unknown image')
                if row['diagnosis'] != meta[identifier]['diagnosis_1']:
                    raise ValueError('Historical diagnosis differs from metadata')
                membership[identifier] = split
    if set(membership) != set(meta):
        raise ValueError('Historical split and metadata IDs must match exactly')
    return membership


def make_plan(meta, cohort, seed=3710):
    if cohort not in COHORT_FIELDS:
        raise ValueError('Unknown cohort')
    field = COHORT_FIELDS[cohort]
    selected = {i for i, row in meta.items() if field is None or row.get(field, '')}
    if not selected:
        raise ValueError('No images in requested cohort')
    # Build on the full metadata first: an excluded row can link two included rows.
    groups = [sorted(set(group) & selected) for group in connected_components(meta)]
    groups = [group for group in groups if group]
    totals = Counter(meta[i]['diagnosis_1'] for i in selected)
    categories = [c for c in CLASSES if totals[c]]
    counts = {split: Counter() for split in SPLITS}
    sizes = Counter()
    assignment, components = {}, {}

    def place(group, split):
        component = 'assignment_' + hashlib.sha256('\n'.join(group).encode()).hexdigest()[:16]
        for identifier in group:
            assignment[identifier] = split
            components[identifier] = component
            counts[split][meta[identifier]['diagnosis_1']] += 1
        sizes[split] += len(group)

    pending = []
    for group in groups:
        if any(meta[i]['diagnosis_1'] == 'Indeterminate' for i in group):
            place(group, 'train')  # No held-out claim for this singleton class.
        else:
            pending.append(group)
    tie = lambda group: hashlib.sha256(f'{seed}:{group[0]}'.encode()).hexdigest()
    pending.sort(key=lambda group: (-len(group), tie(group)))
    # Greedy class/size balance, not optimisation against model predictions.
    # Groups are indivisible; exact image ratios are not promised.
    for group in pending:
        vector = Counter(meta[i]['diagnosis_1'] for i in group)

        def cost_delta(index):
            split, fraction = SPLITS[index], FRACTIONS[index]
            terms = [(sizes[split], len(group), len(selected) * fraction)]
            terms += [(counts[split][c], vector[c], totals[c] * fraction) for c in categories]
            return sum(((n + increment - target)**2 - (n - target)**2) / max(target, 1)
                       for n, increment, target in terms)

        place(group, SPLITS[min(range(3), key=lambda index: (cost_delta(index), index))])
    if set(assignment) != selected:
        raise AssertionError('Assignment coverage mismatch')
    overlap = {}
    for id_field in ('lesion_id', 'patient_id'):
        per_split = {split: {meta[i][id_field] for i, value in assignment.items()
                             if value == split and meta[i][id_field]} for split in SPLITS}
        overlap[id_field] = {f'{a}_{b}': len(per_split[a] & per_split[b])
                             for a, b in itertools.combinations(SPLITS, 2)}
        if any(overlap[id_field].values()):
            raise AssertionError('Known ID crosses candidate splits')
    if any(not counts[split][c] for split in SPLITS for c in ('Benign', 'Malignant')):
        raise ValueError('Cannot support both assessed classes in every split')
    summary = {
        'status': 'candidate_only_not_trained_or_selected', 'cohort': cohort, 'seed': seed,
        'target_image_fractions': dict(zip(SPLITS, FRACTIONS)),
        'algorithm': 'all-known-ID components; largest first; greedy class/size balance',
        'included_images': len(selected), 'excluded_images': len(meta) - len(selected),
        'assignment_components': len(groups),
        'class_counts': {c: totals[c] for c in CLASSES if totals[c]},
        'split_counts': dict(sizes), 'split_class_counts': {s: dict(counts[s]) for s in SPLITS},
        'missing_ids_in_selected_cohort': {f: sum(not meta[i][f] for i in selected)
                                          for f in ('lesion_id', 'patient_id')},
        'known_id_overlap_counts': overlap,
        'complete_lesion_id_coverage_and_separation': all(meta[i]['lesion_id'] for i in selected),
        'complete_patient_id_coverage_and_separation': all(meta[i]['patient_id'] for i in selected),
        'scope': 'Only within selected cohort, conditional on correctness of recorded IDs',
        'imputed_patient_or_lesion_ids': 0, 'fresh_blind_test': False,
        'course_acceptance_confirmed': False,
        'limitations': [
            'Known links do not resolve missing IDs or prove absence of undocumented links.',
            'Assignment component keys are not medical identifiers.',
            'Unknown-ID images must stay excluded from ALL phases of a restricted cohort experiment.',
            'Regrouped previously inspected data is supplementary reassessment, not fresh blind testing.',
            'Fresh models must be trained for this split; historical fitted weights are not valid here.',
            'Different cohorts change sample size and class/source mix; do not attribute changes to model quality.',
        ],
    }
    return assignment, components, summary


def write_candidates(metadata_path, historical_dir, output, seed=3710):
    metadata_path, historical_dir, output = map(Path, (metadata_path, historical_dir, output))
    if output.exists():
        raise ValueError('Preserve previous candidates: output must be new')
    for source in (metadata_path.parent, historical_dir):
        if output.resolve().is_relative_to(source.resolve()):
            raise ValueError('Output must be outside source directories')
    paths = [metadata_path] + [historical_dir / f'{s}.csv' for s in ('train', 'val', 'test')]
    before = {str(path): digest(path) for path in paths}
    meta = read_metadata(metadata_path)
    history = historical_membership(historical_dir, meta)
    plans = {cohort: make_plan(meta, cohort, seed) for cohort in COHORT_FIELDS}
    output.mkdir(parents=True, exist_ok=False)
    summaries = {}
    for cohort, (assignment, components, summary) in plans.items():
        target = output / cohort
        target.mkdir()
        fields = ['image_id', 'diagnosis', 'lesion_id', 'patient_id', 'assignment_component',
                  'candidate_split', 'historical_split']
        with (target / 'manifest.csv').open('x', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for i in sorted(assignment):
                writer.writerow(dict(image_id=i, diagnosis=meta[i]['diagnosis_1'],
                                     lesion_id=meta[i]['lesion_id'], patient_id=meta[i]['patient_id'],
                                     assignment_component=components[i], candidate_split=assignment[i],
                                     historical_split=history[i]))
        for split in SPLITS:
            (target / f'{split}_ids.txt').write_text(''.join(i+'\n' for i in sorted(assignment)
                                                         if assignment[i] == split), encoding='utf-8')
        (target / 'excluded_ids.txt').write_text(''.join(i+'\n' for i in sorted(set(meta)-set(assignment))),
                                                encoding='utf-8')
        summary['historical_membership_of_candidate_splits'] = {
            s: dict(Counter(history[i] for i in assignment if assignment[i] == s)) for s in SPLITS}
        summary['source_sha256'] = before
        (target / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
        summaries[cohort] = summary
    if {str(path): digest(path) for path in paths} != before:
        raise RuntimeError('Input checksum changed during preparation')
    (output / 'summary.json').write_text(json.dumps(summaries, indent=2)+'\n', encoding='utf-8')
    return summaries


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', required=True, type=Path)
    parser.add_argument('--historical-splits', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--seed', type=int, default=3710)
    args = parser.parse_args()
    result = write_candidates(args.metadata, args.historical_splits, args.output, args.seed)
    print(json.dumps({k: {f: v[f] for f in ('included_images', 'split_counts', 'split_class_counts',
                     'known_id_overlap_counts')} for k, v in result.items()}, indent=2))
