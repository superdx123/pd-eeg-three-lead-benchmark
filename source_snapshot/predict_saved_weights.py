"""Explicit reduced-checkpoint prediction adapter; never trains or fits anything.

Default only describes the interface. --check-architecture-only loads CPU weights
and checks parameter keys/shapes without reading EEG or evaluating the network.
--execute requires the original pinned CUDA software versions and performs test
inference only. This adapter checks saved SUBJECT means, not original per-window
1e-7 replay (those original native receipts remain in the research archive).
"""
from pathlib import Path
import argparse
import csv
from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
import sys

if os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in (None, ':4096:8'):
    raise RuntimeError('CUBLAS_WORKSPACE_CONFIG_MUST_BE_4096_8')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4194304), b''):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', choices=('S', 'W'), required=True)
    parser.add_argument('--seed', type=int, choices=(11, 22, 33, 44, 55), required=True)
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--repeat', type=int, default=1)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check-architecture-only', action='store_true')
    mode.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    assert 1 <= args.fold <= (31 if args.protocol == 'S' else 5)
    assert args.repeat in range(1, 6)
    if args.protocol == 'S':
        assert args.repeat == 1
    if not args.check_architecture_only and not args.execute:
        print(json.dumps({'status': 'NO_OPERATION_EXPLICIT_MODE_REQUIRED', 'training_called': False}))
        return
    execution = ROOT / 'experiments' / args.protocol
    task = Path(f'seed_{args.seed}')
    if args.protocol == 'W':
        task /= f'repeat_{args.repeat:02d}'
    task /= f'fold_{args.fold:02d}'
    path = ROOT / 'weights' / args.protocol / task / 'model.pt'
    manifest_path = ROOT / 'release_manifest_v3.json'
    if not manifest_path.exists():
        manifest_path = ROOT / 'release_manifest_v2.json'
    if not manifest_path.exists():
        manifest_path = ROOT / 'release_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    assert sha(path) == manifest['files'][path.relative_to(ROOT).as_posix()]['sha256']
    sys.path.insert(0, str(execution / 'frozen_ds'))
    import numpy as np
    import torch
    from dsmodel.models import build_model, get_config
    from dstraining.trainer import apply_preprocess
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    config = get_config('DS28_BR_MS_AUG_LS_WD3')
    assert checkpoint['model_config'] == asdict(config)
    assert checkpoint['threshold'] == 0.5 and checkpoint['n_parameters'] == 67190
    assert checkpoint['seed'] == args.seed and checkpoint['fold'] == args.fold
    assert checkpoint['protocol'] == args.protocol
    if args.protocol == 'W':
        assert checkpoint['repeat_index'] == args.repeat
    torch.set_num_threads(2)
    network = build_model(config, torch.device('cpu'))
    assert sum(p.numel() for p in network.parameters() if p.requires_grad) == 67190
    network.load_state_dict(checkpoint['model_state_dict'], strict=True)
    if args.check_architecture_only:
        print(json.dumps({'status': 'PASS_CPU_STATE_STRUCTURE_ONLY', 'protocol': args.protocol,
                          'seed': args.seed, 'fold': args.fold, 'n_parameters': 67190,
                          'signal_loaded': False, 'forward_inference_called': False,
                          'training_called': False, 'native_prediction_replay_claimed': False}))
        return
    output = ROOT.parent / 'replay_outputs' / args.protocol / task
    if output.exists():
        raise ValueError('OUTPUT_EXISTS_REFUSE_INFERENCE_OR_OVERWRITE')
    spec = importlib.util.spec_from_file_location('release_execution_common', execution / 'scripts/common.py')
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    common.verify_frozen()
    common.check_environment_versions()  # Hard gate, no version relaxation.
    assert torch.cuda.is_available(), 'PINNED_CUDA_RUNTIME_REQUIRED'
    reference_path = ROOT / 'metadata/predictions' / ('S_per_seed.csv' if args.protocol == 'S' else 'W_per_seed_by_fold.csv')
    assert sha(reference_path) == manifest['files'][reference_path.relative_to(ROOT).as_posix()]['sha256']
    if args.protocol == 'S':
        x, y, groups, outer, inner = common.inputs()
        held = outer.loc[outer.outer_fold.astype(int) == args.fold, 'outer_test_subject_id'].astype(str).tolist()
        assert len(held) == 1
        indices = np.flatnonzero(groups.astype(str) == held[0])
        reference = [r for r in read_csv(reference_path)
                     if int(r['seed']) == args.seed and int(r['outer_fold']) == args.fold]
    else:
        x, y, groups, wm = common.inputs()
        indices = common.role_indices(wm, args.repeat, args.fold)['test']
        reference = [r for r in read_csv(reference_path)
                     if int(r['seed']) == args.seed and int(r['repeat_index']) == args.repeat
                     and int(r['fold']) == args.fold]
    groups = groups.astype(str)
    # Same deterministic P-number mapping used only for public statistical files.
    mapping = {sid: f'P{n:03d}' for n, sid in enumerate(sorted(np.unique(groups)), 1)}
    features = apply_preprocess(x[indices], checkpoint['preprocess'], config.preprocess,
                                config.input_transform, config.n_bands)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    device = torch.device('cuda')
    network.to(device).eval()
    parts = []
    with torch.inference_mode():
        for start in range(0, len(features), 256):
            logits = network(torch.as_tensor(features[start:start + 256], device=device))
            assert logits.shape == (min(256, len(features) - start), 2)
            parts.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    probabilities = np.concatenate(parts).astype(np.float64)
    assert np.isfinite(probabilities).all()
    actual = []
    for subject in sorted(np.unique(groups[indices])):
        local = groups[indices] == subject
        label = int(np.unique(y[indices][local]).item())
        actual.append({'subject_id': mapping[subject], 'true_label': label,
                       'n_windows': int(local.sum()), 'prob_pd': float(probabilities[local].mean())})
    expected = {r['subject_id']: r for r in reference}
    assert set(expected) == {r['subject_id'] for r in actual}
    error = 0.0
    classification_mismatches = 0
    for row in actual:
        saved = expected[row['subject_id']]
        assert row['true_label'] == int(saved['true_label'])
        assert row['n_windows'] == int(saved['n_windows'])
        error = max(error, abs(row['prob_pd'] - float(saved['prob_pd'])))
        classification_mismatches += int(int(row['prob_pd'] >= 0.5) != int(float(saved['prob_pd']) >= 0.5))
    output.mkdir(parents=True, exist_ok=False)
    with (output / 'window_predictions.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=('window_index', 'subject_id', 'true_label', 'prob_pd'))
        writer.writeheader()
        for index, probability in zip(indices, probabilities):
            writer.writerow({'window_index': int(index), 'subject_id': mapping[groups[index]],
                             'true_label': int(y[index]), 'prob_pd': float(probability)})
    with (output / 'subject_predictions.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(actual[0]))
        writer.writeheader()
        writer.writerows(actual)
    passed = error <= 1e-7 and classification_mismatches == 0
    report = {'status': 'PASS_SUBJECT_MEAN_REPLAY' if passed else 'FAIL_SUBJECT_MEAN_REPLAY_KEEP_OUTPUTS',
              'max_subject_probability_error': error, 'frozen_tolerance': 1e-7,
              'classification_mismatches': classification_mismatches,
              'training_called': False, 'threshold_optimization_called': False,
              'original_per_window_native_replay_replaced': False,
              'scope': 'reduced_checkpoint_subject_mean_check_only',
              'model_sha256': sha(path)}
    with (output / 'receipt.json').open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'model_sha256'}))
    if not passed:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
