"""Independent, inference-only acceptance of one newly generated 778 DS28 S fold.

Only trusted local artifacts covered by this experiment's frozen manifests are
accepted. Checkpoints must load with weights_only=True; no pickle fallback,
training, fitting, threshold selection or tolerance relaxation is implemented.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import platform
import sys
from collections import Counter

if os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in (None, ':4096:8'):
    raise RuntimeError('CUBLAS_WORKSPACE_CONFIG must remain :4096:8')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
import torch

from common import (ROOT, sha, save, check_manifest, inputs, protocol, task_dir,
                    verify_frozen, check_environment_versions)

sys.path.insert(0, str(ROOT / 'frozen_ds'))
from dsmodel.models import get_config, build_model
from dstraining import trainer

MODEL = 'DS28_BR_MS_AUG_LS_WD3'
SEEDS = (11, 22, 33, 44, 55)
LR_VALUES = (.0003, .001)
METRIC_TIE = 1e-12
METRIC_CHECK = 1e-10
PREPROCESS_CHECK = 1e-12
REPLAY_TOLERANCE = 1e-7
REQUIRED_FILES = {
    'checkpoint.pt', 'windows.csv', 'subjects.csv', 'selection.json',
    'inner_predictions.csv', 'final_training_history.csv', 'environment.json',
}
ALLOWED_FILES = REQUIRED_FILES | {'timing.json', 'started.json'}


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def finite_float(value, name):
    value = float(value)
    require(math.isfinite(value), name + ' is not finite')
    return value


def close(left, right, name, tolerance=METRIC_CHECK):
    left, right = finite_float(left, name), finite_float(right, name)
    require(abs(left - right) <= tolerance, name + ' mismatch')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def integer(value, name):
    v = finite_float(value, name)
    require(v == math.floor(v), name + ' is not an integer')
    return int(v)


def integer_vector(values, name):
    numeric = np.asarray(values, dtype=np.float64)
    require(numeric.ndim == 1 and np.isfinite(numeric).all() and np.array_equal(numeric, np.floor(numeric)),
            name + ' is not a finite integer vector')
    return numeric.astype(np.int64)


def bool_value(value, name):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value in ('True', 'true', '1', 1):
        return True
    if value in ('False', 'false', '0', 0):
        return False
    raise ValueError(name + ' is not boolean')


def validate_protocol(p):
    expected = {
        'model_name': MODEL, 'seeds': list(SEEDS), 'threshold': .5,
        'lr_candidates': list(LR_VALUES), 'max_epochs': 80, 'patience': 12,
        'batch_size': 256, 'selection_metric': 'bacc', 'threads_per_worker': 2,
        'expected_parameters': 67190, 'probability_replay_tolerance': REPLAY_TOLERANCE,
    }
    for key, value in expected.items():
        require(key in p and p[key] == value, 'Frozen protocol field changed: ' + key)
    return p


def file_bindings(directory, seed, fold):
    complete_path = directory / 'complete.json'
    require(complete_path.is_file(), 'Training completion receipt is missing')
    complete = read_json(complete_path)
    require(complete.get('status') == 'TRAINED_PENDING_VALIDATION', 'Not a pending trained fold')
    require(complete.get('seed') == seed and complete.get('fold') == fold, 'Completion identity mismatch')
    require(complete.get('config_name') == MODEL and complete.get('threshold') == .5,
            'Completion model/threshold mismatch')
    require(isinstance(complete.get('files'), dict), 'Completion file map is missing')
    names, bindings = set(), {}
    root = ROOT.resolve()
    directory = directory.resolve()
    for relative, expected in complete['files'].items():
        relative_path = Path(relative)
        require(not relative_path.is_absolute() and '..' not in relative_path.parts, 'Unsafe artifact name')
        member = (root / relative_path).resolve()
        require(member.is_relative_to(directory), 'Completion member escapes this task')
        require(member.parent == directory and member.name in ALLOWED_FILES, 'Unexpected completion member')
        require(member.name not in names and member.is_file(), 'Missing or duplicate completion member')
        require(isinstance(expected, str) and len(expected) == 64, 'Invalid recorded SHA-256')
        require(sha(member) == expected, 'Completion member SHA-256 mismatch: ' + member.name)
        names.add(member.name)
        bindings[relative_path.as_posix()] = expected
    require(REQUIRED_FILES.issubset(names), 'Required artifacts missing from completion map')
    require(complete.get('checkpoint_sha256') == sha(directory / 'checkpoint.pt'),
            'Completion checkpoint hash field mismatch')
    identities = identity_bindings()
    for key in ('data_manifest_sha256', 'code_manifest_sha256', 'frozen_sha256', 'protocol_sha256'):
        require(complete.get(key) == identities[key], 'Completion manifest binding mismatch: ' + key)
    check_manifest(complete_path)
    return dict(sorted(bindings.items())), sha(complete_path)


def identity_bindings():
    return {
        'data_manifest_sha256': sha(ROOT / 'data_manifest.json'),
        'code_manifest_sha256': sha(ROOT / 'code_manifest.json'),
        'frozen_sha256': sha(ROOT / 'FROZEN.json'),
        'protocol_sha256': sha(ROOT / 'config' / 'protocol.json'),
        'validator_sha256': sha(Path(__file__)),
    }


def existing_receipt(path, seed, fold, artifact_hashes, complete_sha, identities):
    if not path.exists():
        return None
    value = read_json(path)
    require(value.get('status') == 'PASS' and value.get('seed') == seed and value.get('fold') == fold,
            'Existing validation identity/status mismatch; refusing overwrite')
    require(value.get('files') == artifact_hashes and value.get('complete_sha256') == complete_sha,
            'Validated artifacts changed; refusing overwrite')
    for key, expected in identities.items():
        require(value.get(key) == expected, 'Existing validation binding changed: ' + key)
    require(value.get('checkpoint_sha256') == sha(path.parent / 'checkpoint.pt'), 'Validated checkpoint changed')
    require(value.get('probability_replay_tolerance') == REPLAY_TOLERANCE,
            'Existing validation tolerance is different')
    require(value.get('native_cuda_validation') is True and value.get('training_called') is False,
            'Existing validation scope is not accepted')
    actual_versions = {key: str(value) if value is not None else None
                       for key, value in check_environment_versions().items()}
    require(value.get('environment_versions') == actual_versions,
            'Existing validation used another software environment')
    require(torch.cuda.is_available() and torch.cuda.get_device_name() == value['validation_environment']['gpu_name'],
            'Existing validation native GPU differs from this process')
    return value


def tensor_indices(value, name):
    require(isinstance(value, torch.Tensor), name + ' must be a tensor')
    require(value.device.type == 'cpu' and value.dtype == torch.int64 and value.ndim == 1,
            name + ' must be a CPU int64 vector')
    result = value.numpy()
    require(len(result) > 0 and np.array_equal(result, np.unique(result)), name + ' must be nonempty/sorted/unique')
    return result


def independently_derive_roles(y, groups, outer, inner, fold):
    require(groups.shape == y.shape == (3022,), 'Wrong 778 label/group dimensions')
    subjects = sorted(set(str(v) for v in groups))
    require(len(subjects) == 31, 'Wrong 778 subject count')
    labels, counts = {}, {}
    for sid in subjects:
        local_labels = np.unique(y[groups == sid])
        require(len(local_labels) == 1 and int(local_labels[0]) in (0, 1), 'Inconsistent subject labels')
        labels[sid] = int(local_labels[0])
        counts[sid] = int((groups == sid).sum())
    require(Counter(labels.values()) == {0: 16, 1: 15}, 'Wrong 778 subject class counts')
    require(len(outer) == 31 and not outer['outer_fold'].duplicated().any(), 'Incomplete outer manifest')
    require(set(outer['outer_test_subject_id'].astype(str)) == set(subjects), 'Outer subject coverage mismatch')
    selected = outer.loc[outer['outer_fold'].astype(int) == fold]
    require(len(selected) == 1, 'Outer fold identity is not unique')
    test_subject = str(selected.iloc[0]['outer_test_subject_id'])
    train_indices, test_indices = np.flatnonzero(groups != test_subject), np.flatnonzero(groups == test_subject)
    expected = set(subjects) - {test_subject}
    local = inner.loc[inner['outer_fold'].astype(int) == fold].copy()
    require(len(local) == 30 and not local['subject_id'].astype(str).duplicated().any(), 'Inner subject rows incomplete')
    require(set(local['subject_id'].astype(str)) == expected, 'Inner manifest subject coverage mismatch')
    require(set(local['outer_test_subject_id'].astype(str)) == {test_subject}, 'Inner outer-subject mismatch')
    inner_roles, subject_fold = {}, {}
    for k in range(1, 6):
        validation_subjects = set(local.loc[local['inner_validation_fold'].astype(int) == k, 'subject_id'].astype(str))
        require(validation_subjects and validation_subjects < expected, 'Empty/invalid inner validation fold')
        training_subjects = expected - validation_subjects
        tr = np.flatnonzero(np.isin(groups, sorted(training_subjects)))
        va = np.flatnonzero(np.isin(groups, sorted(validation_subjects)))
        require(not np.intersect1d(tr, va).size and not np.intersect1d(tr, test_indices).size
                and not np.intersect1d(va, test_indices).size, 'Outer test leaked into inner roles')
        require(set(y[tr].tolist()) == set(y[va].tolist()) == {0, 1}, 'Inner role lacks a class')
        inner_roles[k] = (tr, va)
        for sid in validation_subjects:
            require(sid not in subject_fold, 'Subject repeated in inner validation folds')
            subject_fold[sid] = k
    require(set(subject_fold) == expected, 'Inner OOF subjects not partitioned')
    require(np.array_equal(np.sort(np.concatenate([v[1] for v in inner_roles.values()])), train_indices),
            'Inner OOF windows do not cover the outer training set')
    return train_indices, test_indices, test_subject, expected, labels, counts, inner_roles, subject_fold


def independent_metrics(labels, probabilities):
    truth = np.asarray(labels, dtype=np.int64)
    p = np.asarray(probabilities, dtype=np.float64)
    require(truth.shape == p.shape and truth.ndim == 1 and len(truth), 'Invalid metric vectors')
    require(set(truth.tolist()) == {0, 1} and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(),
            'Invalid OOF classes/probabilities')
    pos, neg = p[truth == 1], p[truth == 0]
    auc = float(((pos[:, None] > neg).sum() + .5 * (pos[:, None] == neg).sum()) / (len(pos) * len(neg)))
    bacc = float(.5 * ((pos >= .5).mean() + (neg < .5).mean()))
    return bacc, auc


def check_candidate_fields(record, expected):
    require(integer(record['selected_epoch'], 'selected_epoch') == expected['selected_epoch'], 'Candidate epoch mismatch')
    require(integer(record['selection_stop_epoch'], 'selection_stop_epoch') == expected['selection_stop_epoch'],
            'Candidate stopping epoch mismatch')
    for key in ('learning_rate', 'best_validation_subject_balanced_accuracy', 'best_validation_subject_auc'):
        close(record[key], expected[key], key, 1e-15 if key == 'learning_rate' else METRIC_CHECK)


def validate_selection(selection, predictions, expected_subjects, labels, counts, subject_fold, p):
    candidates, history = selection['candidates'], selection['history']
    require(isinstance(candidates, list) and len(candidates) == 2, 'Exactly two LR candidates required')
    require(isinstance(history, list) and history, 'Candidate epoch history missing')
    require(sorted(float(c['learning_rate']) for c in candidates) == list(LR_VALUES), 'LR candidates changed')
    required = {'learning_rate', 'inner_fold', 'epoch', 'subject_id', 'true_label', 'prob_pd', 'n_windows', 'predicted_label'}
    require(required.issubset(predictions.columns), 'Inner OOF schema incomplete')
    integer_vector(predictions['epoch'], 'OOF epochs')
    require(np.isfinite(predictions['learning_rate'].to_numpy(dtype=float)).all(), 'Invalid OOF LR')
    require(set(predictions['learning_rate'].to_numpy(dtype=float)) == set(LR_VALUES), 'OOF LR coverage changed')
    require(set(float(h['learning_rate']) for h in history) == set(LR_VALUES), 'History LR coverage changed')
    independently_selected, candidate_audit = [], []
    for lr in LR_VALUES:
        candidate = [c for c in candidates if float(c['learning_rate']) == lr]
        require(len(candidate) == 1, 'Candidate LR is duplicated')
        local_history = [h for h in history if float(h['learning_rate']) == lr]
        local_history = sorted(local_history, key=lambda r: integer(r['epoch'], 'history epoch'))
        stop_epoch = len(local_history)
        require(1 <= stop_epoch <= p['max_epochs'], 'Candidate history length invalid')
        require([integer(r['epoch'], 'history epoch') for r in local_history] == list(range(1, stop_epoch + 1)),
                'Candidate epochs are missing/duplicated')
        local_predictions = predictions.loc[predictions['learning_rate'].astype(float) == lr].copy()
        require(set(local_predictions['epoch'].astype(int)) == set(range(1, stop_epoch + 1)), 'OOF epoch coverage mismatch')
        require(len(local_predictions) == 30 * stop_epoch, 'OOF row count does not cover all candidate epochs')
        best_epoch, best_bacc, best_auc, patience = 0, -math.inf, -math.inf, 0
        for epoch, h in enumerate(local_history, 1):
            values = local_predictions.loc[local_predictions['epoch'].astype(int) == epoch].copy()
            require(len(values) == 30 and not values['subject_id'].astype(str).duplicated().any(),
                    'OOF must contain each of the 30 outer-training subjects once per epoch')
            require(set(values['subject_id'].astype(str)) == expected_subjects, 'OOF has test or missing subject')
            for row in values.itertuples(index=False):
                sid = str(row.subject_id)
                require(integer(row.true_label, 'OOF label') == labels[sid], 'OOF label does not match dataset')
                require(integer(row.n_windows, 'OOF windows') == counts[sid], 'OOF window count mismatch')
                require(integer(row.inner_fold, 'OOF inner fold') == subject_fold[sid], 'OOF inner-fold subject mismatch')
                prob = finite_float(row.prob_pd, 'OOF probability')
                require(0 <= prob <= 1 and integer(row.predicted_label, 'OOF predicted label') == int(prob >= .5),
                        'OOF probability/threshold mismatch')
            bacc, auc = independent_metrics(values['true_label'], values['prob_pd'])
            close(h['validation_subject_balanced_accuracy'], bacc, 'OOF bAcc')
            close(h['validation_subject_auc'], auc, 'OOF AUC')
            require(finite_float(h['mean_training_loss'], 'inner training loss') >= 0, 'Negative training loss')
            improved = bacc > best_bacc + METRIC_TIE
            if abs(bacc - best_bacc) <= METRIC_TIE and auc > best_auc + METRIC_TIE:
                improved = True
            if improved:
                best_epoch, best_bacc, best_auc, patience = epoch, bacc, auc, 0
            else:
                patience += 1
            require(bool_value(h['improved'], 'improved') == improved, 'Improvement marker mismatch')
            require(integer(h['patience_counter'], 'patience') == patience, 'Patience counter mismatch')
            require(bool_value(h['selection_stopped_here'], 'stop marker') == (epoch == stop_epoch), 'Stop marker mismatch')
            if epoch < stop_epoch:
                require(patience < p['patience'], 'Selection continued after frozen stopping condition')
        if stop_epoch < p['max_epochs']:
            require(patience >= p['patience'], 'Selection stopped before frozen stopping condition')
        checked = {'learning_rate': lr, 'selected_epoch': best_epoch,
                   'selection_stop_epoch': stop_epoch,
                   'best_validation_subject_balanced_accuracy': best_bacc,
                   'best_validation_subject_auc': best_auc}
        check_candidate_fields(candidate[0], checked)
        independently_selected.append(checked)
        candidate_audit.append({'learning_rate': lr, 'epochs_checked': stop_epoch,
                                'OOF_subject_rows_checked': len(local_predictions), 'status': 'PASS'})
    # Independent reproduction of the frozen choose_candidate comparison order.
    winner = None
    for candidate in sorted(independently_selected, key=lambda r: r['learning_rate']):
        if winner is None:
            winner = candidate
            continue
        primary, bp = candidate['best_validation_subject_balanced_accuracy'], winner['best_validation_subject_balanced_accuracy']
        secondary, bs = candidate['best_validation_subject_auc'], winner['best_validation_subject_auc']
        replace = primary > bp + METRIC_TIE
        if math.isclose(primary, bp, rel_tol=0, abs_tol=METRIC_TIE):
            if secondary > bs + METRIC_TIE:
                replace = True
            elif math.isclose(secondary, bs, rel_tol=0, abs_tol=METRIC_TIE):
                if candidate['learning_rate'] < winner['learning_rate']:
                    replace = True
                elif math.isclose(candidate['learning_rate'], winner['learning_rate'], rel_tol=0, abs_tol=1e-15):
                    replace = candidate['selected_epoch'] < winner['selected_epoch']
        if replace:
            winner = candidate
    check_candidate_fields(selection['selected'], winner)
    return winner, candidate_audit


def independent_expansion(windows):
    value = np.asarray(windows, dtype=np.float32)
    require(value.ndim == 3 and value.shape[1:] == (3, 512), 'Input is not C3/Cz/C4 x512')
    frequencies = np.fft.rfftfreq(512, d=1 / 256)
    spectrum = np.fft.rfft(value.astype(np.float64), axis=-1)
    bands = []
    for lo, hi in ((1., 4.), (4., 8.), (8., 13.), (13., 30.), (30., 40.)):
        mask = ((frequencies >= lo) & (frequencies <= hi)).astype(np.float32)
        bands.append(np.fft.irfft(spectrum * mask, n=512, axis=-1).astype(np.float32))
    return np.concatenate([value] + bands, axis=1)


def native_environment(device):
    require(device.type == 'cuda' and torch.cuda.is_available(), 'Native CUDA is required; CPU fallback is forbidden')
    properties = torch.cuda.get_device_properties(device)
    return {
        'python': platform.python_version(), 'platform': platform.platform(),
        'numpy': str(np.__version__), 'pandas': str(pd.__version__), 'torch': str(torch.__version__),
        'torch_cuda': str(torch.version.cuda), 'cudnn': torch.backends.cudnn.version(),
        'gpu_name': str(properties.name), 'gpu_total_memory_bytes': int(properties.total_memory),
        'gpu_compute_capability': list(torch.cuda.get_device_capability(device)),
        'native_cuda': True, 'device_type': device.type,
        'torch_num_threads': int(torch.get_num_threads()),
        'torch_num_interop_threads': int(torch.get_num_interop_threads()),
        'CUBLAS_WORKSPACE_CONFIG': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        'deterministic_algorithms': bool(torch.are_deterministic_algorithms_enabled()),
        'cudnn_deterministic': bool(torch.backends.cudnn.deterministic),
        'cudnn_benchmark': bool(torch.backends.cudnn.benchmark),
        'cuda_matmul_allow_tf32': bool(torch.backends.cuda.matmul.allow_tf32),
        'cudnn_allow_tf32': bool(torch.backends.cudnn.allow_tf32),
    }


def validate(seed, fold):
    require(seed in SEEDS and 1 <= fold <= 31, 'Task outside the frozen grid')
    verify_frozen()
    p = validate_protocol(protocol())
    versions = {key: str(value) if value is not None else None
                for key, value in check_environment_versions().items()}
    directory = task_dir(seed, fold).resolve()
    require(directory.is_relative_to(ROOT.resolve()) and directory.is_dir(), 'Task directory missing/unsafe')
    artifact_hashes, complete_sha = file_bindings(directory, seed, fold)
    identities = identity_bindings()
    receipt_path = directory / 'validated.json'
    existing = existing_receipt(receipt_path, seed, fold, artifact_hashes, complete_sha, identities)
    if existing is not None:
        print(json.dumps({'status': 'PASS', 'seed': seed, 'fold': fold, 'existing_receipt_revalidated': True,
                          'checkpoint_sha256': existing['checkpoint_sha256']}, ensure_ascii=False), flush=True)
        return existing
    X, y, groups, outer, inner = inputs()
    require(np.asarray(X).shape == (3022, 3, 512) and np.isfinite(X).all(), 'Wrong/invalid classifier input')
    y, groups = np.asarray(y, dtype=np.int64), np.asarray(groups).astype(str)
    train_indices, test_indices, test_subject, expected, labels, counts, roles, subject_fold = independently_derive_roles(
        y, groups, outer, inner, fold)
    # No unsafe fallback is provided. Only this verified local experiment may supply checkpoints.
    checkpoint = torch.load(directory / 'checkpoint.pt', map_location='cpu', weights_only=True)
    require(isinstance(checkpoint, dict), 'Checkpoint is not a dictionary')
    require(checkpoint.get('config_name') == MODEL and checkpoint.get('seed') == seed and checkpoint.get('fold') == fold,
            'Checkpoint task identity mismatch')
    require(checkpoint.get('threshold') == .5 and checkpoint.get('n_parameters') == p['expected_parameters'],
            'Checkpoint threshold/parameter count changed')
    for key in ('data_manifest_sha256', 'code_manifest_sha256', 'frozen_sha256', 'protocol_sha256'):
        require(checkpoint.get(key) == identities[key], 'Checkpoint manifest binding mismatch: ' + key)
    require(checkpoint.get('environment_versions') == versions,
            'Checkpoint software environment differs from frozen native validation environment')
    require(np.array_equal(tensor_indices(checkpoint['train_indices'], 'train_indices'), train_indices), 'Wrong checkpoint training indices')
    require(np.array_equal(tensor_indices(checkpoint['test_indices'], 'test_indices'), test_indices), 'Wrong checkpoint test indices')
    records = checkpoint['inner_indices']
    require(isinstance(records, list) and len(records) == 5, 'Checkpoint inner role count changed')
    seen = set()
    for record in records:
        k = integer(record['inner_fold'], 'checkpoint inner fold')
        require(k in roles and k not in seen, 'Checkpoint inner folds missing/duplicated')
        seen.add(k)
        require(np.array_equal(tensor_indices(record['train_indices'], 'inner train indices'), roles[k][0]), 'Wrong inner train indices')
        require(np.array_equal(tensor_indices(record['val_indices'], 'inner validation indices'), roles[k][1]), 'Wrong inner validation indices')
    cfg = get_config(MODEL)
    require(checkpoint['model_config'] == asdict(cfg), 'Checkpoint does not use the frozen complete DS28 configuration')
    selection = read_json(directory / 'selection.json')
    inner_predictions = pd.read_csv(directory / 'inner_predictions.csv', dtype={'subject_id': str})
    selected, candidate_audit = validate_selection(selection, inner_predictions, expected, labels, counts, subject_fold, p)
    check_candidate_fields(checkpoint['selected'], selected)
    final_history = pd.read_csv(directory / 'final_training_history.csv')
    require(set(final_history.columns) == {'epoch', 'training_loss'}, 'Final history schema changed')
    epochs = [integer(v, 'final epoch') for v in final_history['epoch']]
    require(epochs == list(range(1, selected['selected_epoch'] + 1)), 'Final training history length/epochs mismatch')
    losses = [finite_float(v, 'final training loss') for v in final_history['training_loss']]
    require(all(v >= 0 for v in losses) and len(selection['final_losses']) == len(losses), 'Final losses missing/invalid')
    for left, right in zip(losses, selection['final_losses']):
        close(left, right, 'Final loss record', 1e-12)
    training_environment = read_json(directory / 'environment.json')
    require(checkpoint['environment'] == training_environment, 'Training environment record differs from checkpoint')
    # Independently reconstruct fixed band/raw input and train-only scalar statistics.
    expanded_train = independent_expansion(X[train_indices])
    mean = float(np.mean(expanded_train, dtype=np.float64))
    std = float(np.std(expanded_train, dtype=np.float64) + 1e-8)
    del expanded_train
    require(set(checkpoint['preprocess']) == {'mean', 'std'}, 'Unexpected preprocessing metadata')
    close(checkpoint['preprocess']['mean'], mean, 'Training-only mean', PREPROCESS_CHECK)
    close(checkpoint['preprocess']['std'], std, 'Training-only std', PREPROCESS_CHECK)
    expanded_test = independent_expansion(X[test_indices])
    test_features = np.asarray((expanded_test - mean) / std, dtype=np.float32)
    del expanded_test
    saved = pd.read_csv(directory / 'windows.csv', dtype={'subject_id': str})
    require(len(saved) == len(test_indices) and np.array_equal(integer_vector(saved['window_index'], 'test window indices'), test_indices),
            'Test window order/coverage mismatch')
    require(np.array_equal(saved['subject_id'].astype(str).to_numpy(), groups[test_indices]), 'Test subject mapping mismatch')
    require(np.array_equal(integer_vector(saved['true_label'], 'test labels'), y[test_indices]), 'Test label mapping mismatch')
    require((integer_vector(saved['outer_fold'], 'test outer folds') == fold).all(), 'Saved test outer fold mismatch')
    require((integer_vector(saved['selected_epoch'], 'test selected epochs') == selected['selected_epoch']).all(), 'Saved final epoch mismatch')
    require((saved['selected_learning_rate'].astype(float) == selected['learning_rate']).all(), 'Saved final LR mismatch')
    saved_probability = saved['prob_pd'].to_numpy(dtype=np.float64)
    require(np.isfinite(saved_probability).all() and ((saved_probability >= 0) & (saved_probability <= 1)).all(),
            'Saved test probabilities invalid')
    torch.set_num_threads(p['threads_per_worker'])
    device = trainer.configure_device(deterministic=True)
    environment = native_environment(device)
    require(checkpoint.get('device_name') == training_environment.get('device_name') == environment['gpu_name'],
            'Training and native validation GPU names differ; tolerance cannot be relaxed')
    require(training_environment.get('environment_versions') == versions,
            'Training environment file differs from frozen software versions')
    require(training_environment.get('torch_version') == versions['torch']
            and training_environment.get('torch_cuda_version') == versions['cuda_runtime'],
            'Training torch/CUDA environment fields disagree')
    require(training_environment.get('cudnn_version') == environment['cudnn']
            and training_environment.get('compute_capability') == environment['gpu_compute_capability'],
            'Training/native CUDA library or GPU capability changed')
    require(training_environment.get('torch_num_threads') == p['threads_per_worker']
            and training_environment.get('deterministic_algorithms') is True
            and training_environment.get('cudnn_deterministic') is True
            and training_environment.get('cudnn_benchmark') is False
            and training_environment.get('cuda_matmul_allow_tf32') is False
            and training_environment.get('cudnn_allow_tf32') is False,
            'Training deterministic CUDA/thread settings are inconsistent')
    require(environment['deterministic_algorithms'] and environment['cudnn_deterministic']
            and not environment['cudnn_benchmark'] and not environment['cuda_matmul_allow_tf32']
            and not environment['cudnn_allow_tf32'], 'Native deterministic CUDA settings changed')
    network = build_model(cfg, device)
    actual_parameters = sum(parameter.numel() for parameter in network.parameters() if parameter.requires_grad)
    require(actual_parameters == p['expected_parameters'] == 67190, 'Model parameter count changed')
    state = checkpoint['model_state_dict']
    require(isinstance(state, dict) and state, 'Missing model state')
    for name, tensor in state.items():
        require(isinstance(name, str) and isinstance(tensor, torch.Tensor) and tensor.device.type == 'cpu', 'Invalid model state item')
        if tensor.is_floating_point():
            require(torch.isfinite(tensor).all().item(), 'Model parameter contains NaN/Inf')
    network.load_state_dict(state, strict=True)
    network.eval()
    probabilities = []
    with torch.inference_mode():
        for start in range(0, len(test_features), p['batch_size']):
            batch = torch.as_tensor(test_features[start:start + p['batch_size']], device=device)
            logits = network(batch)
            require(tuple(logits.shape) == (len(batch), 2) and torch.isfinite(logits).all().item(), 'Invalid checkpoint logits')
            probabilities.append(torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy())
    replay = np.concatenate(probabilities).astype(np.float64)
    require(replay.shape == saved_probability.shape and np.isfinite(replay).all(), 'Invalid replay vector')
    error = float(np.max(np.abs(replay - saved_probability)))
    require(error <= REPLAY_TOLERANCE, 'Checkpoint probability replay exceeds frozen 1e-7 tolerance; preserve artifacts')
    require(np.array_equal(replay >= .5, saved_probability >= .5), 'Checkpoint replay changes frozen-threshold window predictions')
    subject = pd.read_csv(directory / 'subjects.csv', dtype={'subject_id': str})
    require(len(subject) == 1 and str(subject.iloc[0]['subject_id']) == test_subject, 'Saved subject coverage mismatch')
    row = subject.iloc[0]
    require(integer(row['true_label'], 'subject label') == labels[test_subject]
            and integer(row['n_windows'], 'subject windows') == len(test_indices), 'Saved subject label/window count mismatch')
    close(row['prob_pd'], float(saved_probability.mean()), 'Saved subject/window arithmetic mean', 1e-12)
    close(row['prob_pd'], float(replay.mean()), 'Checkpoint subject probability', REPLAY_TOLERANCE)
    require(integer(row['predicted_label'], 'subject predicted label') == int(float(row['prob_pd']) >= .5)
            and int(float(row['prob_pd']) >= .5) == int(float(replay.mean()) >= .5), 'Subject threshold prediction mismatch')
    del network
    torch.cuda.empty_cache()
    verify_frozen()
    final_artifacts, final_complete_sha = file_bindings(directory, seed, fold)
    require(final_artifacts == artifact_hashes and final_complete_sha == complete_sha and identity_bindings() == identities,
            'Artifacts/source changed during validation')
    require(not receipt_path.exists(), 'Validation receipt appeared concurrently; refusing overwrite')
    result = {
        'schema_version': '1.0', 'status': 'PASS', 'model': MODEL, 'seed': seed, 'fold': fold,
        'checkpoint_sha256': sha(directory / 'checkpoint.pt'), 'complete_sha256': complete_sha,
        'files': artifact_hashes, **identities,
        'n_parameters': actual_parameters, 'threshold': .5,
        'probability_replay_tolerance': REPLAY_TOLERANCE, 'max_window_probability_error': error,
        'subject_probability_error': abs(float(row['prob_pd']) - float(replay.mean())),
        'test_windows': len(test_indices), 'test_subjects': 1, 'outer_training_subjects': 30,
        'inner_candidates_checked': candidate_audit,
        'selection_rule': 'subject_bAcc_then_AUC_tie1e-12_then_smaller_LR_then_earlier_epoch',
        'selected': selected, 'final_history_epochs': len(final_history),
        'preprocess_audit': 'independent_float64_FFT_band_raw_and_train_only_scalar_stats_tolerance1e-12',
        'native_cuda_validation': True, 'validation_environment': environment,
        'training_environment': training_environment, 'environment_versions': versions,
        'training_called': False, 'threshold_optimization_called': False,
        'inner_checkpoint_predictions_replayed': False,
        'checkpoint_loading': 'weights_only=True; verified local project-generated artifact; no pickle fallback',
        'checks': ['frozen_manifests_and_files', 'exact_train_test_inner_indices', 'no_outer_test_in_inner',
                   'two_LR_full_30_subject_OOF_each_epoch', 'independent_epoch_LR_selection', 'final_history',
                   'complete_DS28_config_67190', 'train_only_preprocess', 'strict_state_load',
                   'native_CUDA_window_and_subject_replay', 'fixed_threshold_predictions'],
    }
    save(receipt_path, result)
    print(json.dumps({'status': 'PASS', 'seed': seed, 'fold': fold,
                      'max_window_probability_error': error, 'checkpoint_sha256': result['checkpoint_sha256']}, ensure_ascii=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=list(SEEDS), required=True)
    parser.add_argument('--fold', type=int, required=True)
    args = parser.parse_args()
    validate(args.seed, args.fold)


if __name__ == '__main__':
    main()
