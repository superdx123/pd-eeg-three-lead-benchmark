"""Independent native-CUDA acceptance of one newly trained DS28 W repeat/fold.

W deliberately overlaps subjects, never window roles. Two learning rates use
one train/validation window split, then refit train+validation before test.
Only safe, manifest-verified local checkpoints are loaded; no training occurs.
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
REPEATS = (1, 2, 3, 4, 5)
FOLDS = (1, 2, 3, 4, 5)
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
    require(tuple(p['repeats']) == REPEATS and tuple(p['folds']) == FOLDS, 'W grid changed')
    require(p['protocol'] == 'W_subject_overlap_window_cv', 'Wrong W protocol')
    return p


def file_bindings(directory, seed, repeat_index, fold):
    complete_path = directory / 'complete.json'
    require(complete_path.is_file(), 'Training completion receipt is missing')
    complete = read_json(complete_path)
    require(complete.get('status') == 'TRAINED_PENDING_VALIDATION', 'Not a pending trained fold')
    require(complete.get('seed') == seed and complete.get('repeat_index') == repeat_index and complete.get('fold') == fold, 'Completion identity mismatch')
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


def existing_receipt(path, seed, repeat_index, fold, artifact_hashes, complete_sha, identities):
    if not path.exists():
        return None
    value = read_json(path)
    require(value.get('status') == 'PASS' and value.get('seed') == seed and value.get('repeat_index') == repeat_index and value.get('fold') == fold,
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


def independently_derive_w_roles(y, groups, manifest, repeat_index, fold):
    require(y.shape == groups.shape == (3022,), 'Wrong W label/group dimensions')
    subjects = sorted(set(str(v) for v in groups))
    require(len(subjects) == 31, 'Wrong cohort size')
    labels = {}
    for sid in subjects:
        local = np.unique(y[groups == sid])
        require(len(local) == 1 and int(local[0]) in (0, 1), 'Inconsistent subject labels')
        labels[sid] = int(local[0])
    require(Counter(labels.values()) == {0: 16, 1: 15}, 'Wrong cohort class counts')
    require(len(manifest) == 15110 and set(manifest['repeat_index'].astype(int)) == set(REPEATS),
            'Wrong full W manifest coverage')
    selected = manifest.loc[manifest['repeat_index'].astype(int) == repeat_index].copy()
    require(len(selected) == 3022 and not selected['window_index'].duplicated().any(), 'W repeat coverage invalid')
    selected = selected.sort_values('window_index').reset_index(drop=True)
    require(np.array_equal(integer_vector(selected['window_index'], 'manifest window indices'), np.arange(3022)),
            'W manifest missing/repeated window indices')
    require(np.array_equal(selected['subject_id'].astype(str).to_numpy(), groups)
            and np.array_equal(integer_vector(selected['true_label'], 'manifest labels'), y), 'W metadata mapping mismatch')
    require(set(selected['protocol'].astype(str)) == {'W_subject_overlap_window_cv'}, 'Wrong W manifest protocol')
    require(set(integer_vector(selected['split_seed'], 'split seed')) == {11001 * repeat_index}, 'Wrong W split seed')
    assigned = integer_vector(selected['test_fold'], 'manifest test folds')
    require(set(assigned) == set(FOLDS), 'W folds missing')
    test = np.flatnonzero(assigned == fold)
    validation = np.flatnonzero(assigned == (fold % 5 + 1))
    train = np.flatnonzero((assigned != fold) & (assigned != (fold % 5 + 1)))
    for name, indices in (('train', train), ('validation', validation), ('test', test)):
        require(len(indices) and set(y[indices]) == {0, 1}, 'Empty/class-deficient W ' + name)
    require(not np.intersect1d(train, validation).size and not np.intersect1d(train, test).size
            and not np.intersect1d(validation, test).size, 'W window roles overlap')
    require(np.array_equal(np.sort(np.r_[train, validation, test]), np.arange(3022)), 'W roles do not cover all windows')
    final_train = np.sort(np.r_[train, validation])
    expected = set(str(v) for v in groups[validation])
    validation_counts = {sid: int((groups[validation] == sid).sum()) for sid in expected}
    return train, validation, final_train, test, expected, labels, validation_counts, {sid: 0 for sid in expected}


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
    expected_count = len(expected_subjects)
    require(expected_count > 1, 'Validation subject set empty')
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
        require(len(local_predictions) == expected_count * stop_epoch, 'OOF row count does not cover all candidate epochs')
        best_epoch, best_bacc, best_auc, patience = 0, -math.inf, -math.inf, 0
        for epoch, h in enumerate(local_history, 1):
            values = local_predictions.loc[local_predictions['epoch'].astype(int) == epoch].copy()
            require(len(values) == expected_count and not values['subject_id'].astype(str).duplicated().any(),
                    'W validation must contain each expected subject once per epoch')
            require(set(values['subject_id'].astype(str)) == expected_subjects, 'W validation has missing or extra subject')
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


def validate(seed, repeat_index, fold):
    require(seed in SEEDS and repeat_index in REPEATS and fold in FOLDS, 'Task outside frozen W grid')
    verify_frozen()
    p = validate_protocol(protocol())
    versions = {key: str(value) if value is not None else None
                for key, value in check_environment_versions().items()}
    directory = task_dir(seed, repeat_index, fold).resolve()
    require(directory.is_relative_to(ROOT.resolve()) and directory.is_dir(), 'Missing/unsafe W task directory')
    artifact_hashes, complete_sha = file_bindings(directory, seed, repeat_index, fold)
    identities = identity_bindings()
    receipt_path = directory / 'validated.json'
    existing = existing_receipt(receipt_path, seed, repeat_index, fold, artifact_hashes, complete_sha, identities)
    if existing is not None:
        print(json.dumps({'status': 'PASS', 'seed': seed, 'repeat_index': repeat_index, 'fold': fold,
                          'existing_receipt_revalidated': True}), flush=True)
        return existing
    X, y, groups, manifest = inputs()
    require(np.asarray(X).shape == (3022, 3, 512) and np.isfinite(X).all(), 'Wrong classifier input')
    y, groups = np.asarray(y, dtype=np.int64), np.asarray(groups).astype(str)
    selection_train, validation_indices, train_indices, test_indices, expected, labels, counts, subject_fold = independently_derive_w_roles(
        y, groups, manifest, repeat_index, fold)
    checkpoint = torch.load(directory / 'checkpoint.pt', map_location='cpu', weights_only=True)
    require(isinstance(checkpoint, dict) and checkpoint.get('config_name') == MODEL
            and checkpoint.get('protocol') == 'W_subject_overlap_window_cv'
            and checkpoint.get('seed') == seed and checkpoint.get('repeat_index') == repeat_index
            and checkpoint.get('fold') == fold and checkpoint.get('validation_fold') == fold % 5 + 1,
            'Wrong checkpoint W identity')
    require(checkpoint.get('threshold') == .5 and checkpoint.get('n_parameters') == 67190,
            'Checkpoint threshold/parameter count changed')
    for key in ('data_manifest_sha256', 'code_manifest_sha256', 'frozen_sha256', 'protocol_sha256'):
        require(checkpoint.get(key) == identities[key], 'Checkpoint manifest mismatch: ' + key)
    require(checkpoint.get('environment_versions') == versions, 'Checkpoint software environment changed')
    for key, expected_indices in (('selection_train_indices', selection_train), ('validation_indices', validation_indices),
                                  ('final_train_indices', train_indices), ('train_indices', train_indices), ('test_indices', test_indices)):
        require(np.array_equal(tensor_indices(checkpoint[key], key), expected_indices), 'Wrong checkpoint ' + key)
    inner = checkpoint['inner_indices']
    require(isinstance(inner, list) and len(inner) == 1 and inner[0]['inner_fold'] == 0, 'W must use one validation split')
    require(np.array_equal(tensor_indices(inner[0]['train_indices'], 'inner train'), selection_train)
            and np.array_equal(tensor_indices(inner[0]['val_indices'], 'inner validation'), validation_indices),
            'Checkpoint selection split mismatch')
    cfg = get_config(MODEL)
    require(checkpoint['model_config'] == asdict(cfg), 'Not complete frozen DS28 configuration')
    selection = read_json(directory / 'selection.json')
    require(selection['validation_scope'] == 'one_window_disjoint_split_with_intentional_subject_overlap',
            'W validation scope changed')
    for name, expected_count in (('selection_training_windows', len(selection_train)), ('validation_windows', len(validation_indices)),
                                 ('final_training_windows', len(train_indices)), ('test_windows', len(test_indices))):
        require(selection[name] == expected_count, 'Selection index count mismatch: ' + name)
    validation_frame = pd.read_csv(directory / 'inner_predictions.csv', dtype={'subject_id': str})
    selected, candidate_audit = validate_selection(selection, validation_frame, expected, labels, counts, subject_fold, p)
    check_candidate_fields(checkpoint['selected'], selected)
    final_history = pd.read_csv(directory / 'final_training_history.csv')
    require(set(final_history.columns) == {'epoch', 'training_loss'}, 'Final history schema changed')
    require([integer(v, 'final epoch') for v in final_history['epoch']] == list(range(1, selected['selected_epoch'] + 1)),
            'Final history does not match selected epoch')
    losses = [finite_float(v, 'final loss') for v in final_history['training_loss']]
    require(all(v >= 0 for v in losses) and len(selection['final_losses']) == len(losses), 'Invalid final losses')
    for left, right in zip(losses, selection['final_losses']):
        close(left, right, 'Final loss record', 1e-12)
    training_environment = read_json(directory / 'environment.json')
    require(checkpoint['environment'] == training_environment, 'Training environment record mismatch')
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
    require(np.array_equal(saved['subject_id'].astype(str).to_numpy(), groups[test_indices])
            and np.array_equal(integer_vector(saved['true_label'], 'test labels'), y[test_indices]), 'Test metadata mismatch')
    require((integer_vector(saved['repeat_index'], 'saved repeat') == repeat_index).all()
            and (integer_vector(saved['fold'], 'saved fold') == fold).all(), 'Saved W task identity mismatch')
    require((integer_vector(saved['selected_epoch'], 'saved epoch') == selected['selected_epoch']).all()
            and (saved['selected_learning_rate'].to_numpy(dtype=float) == selected['learning_rate']).all(),
            'Saved selected parameters mismatch')
    saved_probability = saved['prob_pd'].to_numpy(dtype=np.float64)
    require(np.isfinite(saved_probability).all() and ((saved_probability >= 0) & (saved_probability <= 1)).all(),
            'Saved probabilities invalid')
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
    subjects = pd.read_csv(directory / 'subjects.csv', dtype={'subject_id': str})
    expected_test_subjects = set(str(v) for v in groups[test_indices])
    require(len(subjects) == len(expected_test_subjects) and not subjects['subject_id'].astype(str).duplicated().any()
            and set(subjects['subject_id'].astype(str)) == expected_test_subjects, 'True subject aggregation coverage mismatch')
    subject_error = 0.
    for row in subjects.itertuples(index=False):
        sid = str(row.subject_id)
        positions = np.flatnonzero(groups[test_indices] == sid)
        require(integer(row.true_label, 'subject label') == labels[sid]
                and integer(row.n_windows, 'subject test windows') == len(positions), 'Subject label/window count mismatch')
        old_mean, replay_mean = float(saved_probability[positions].mean()), float(replay[positions].mean())
        close(row.prob_pd, old_mean, 'Saved per-subject arithmetic mean', 1e-12)
        close(row.prob_pd, replay_mean, 'Replayed subject probability', REPLAY_TOLERANCE)
        require(integer(row.predicted_label, 'subject prediction') == int(float(row.prob_pd) >= .5)
                and int(old_mean >= .5) == int(replay_mean >= .5), 'Subject threshold prediction mismatch')
        subject_error = max(subject_error, abs(float(row.prob_pd) - replay_mean))
    del network
    torch.cuda.empty_cache()
    verify_frozen()
    final_files, final_complete = file_bindings(directory, seed, repeat_index, fold)
    require(final_files == artifact_hashes and final_complete == complete_sha and identity_bindings() == identities,
            'Artifacts/source changed during validation')
    require(not receipt_path.exists(), 'Validation appeared concurrently; refusing overwrite')
    result = {
        'schema_version': '1.0', 'status': 'PASS', 'protocol': 'W_subject_overlap_window_cv',
        'model': MODEL, 'seed': seed, 'repeat_index': repeat_index, 'fold': fold, 'validation_fold': fold % 5 + 1,
        'checkpoint_sha256': sha(directory / 'checkpoint.pt'), 'complete_sha256': complete_sha,
        'files': artifact_hashes, **identities, 'n_parameters': actual_parameters, 'threshold': .5,
        'probability_replay_tolerance': REPLAY_TOLERANCE, 'max_window_probability_error': error,
        'max_subject_probability_error': subject_error, 'selection_train_windows': len(selection_train),
        'validation_windows': len(validation_indices), 'final_train_windows': len(train_indices), 'test_windows': len(test_indices),
        'validation_subjects': len(expected), 'test_subjects': len(expected_test_subjects),
        'subject_overlap_is_intentional': True, 'window_roles_disjoint': True,
        'selection_split_count': 1, 'inner_fold_marker': 0, 'candidates_checked': candidate_audit,
        'selected': selected, 'final_history_epochs': len(final_history),
        'selection_rule': 'validation_subject_bAcc_then_AUC_tie1e-12_then_smaller_LR_then_earlier_epoch',
        'preprocess_audit': 'independent_FFT_band_raw_then_train_plus_validation_only_scalar_stats',
        'native_cuda_validation': True, 'validation_environment': environment,
        'training_environment': training_environment, 'environment_versions': versions,
        'training_called': False, 'threshold_optimization_called': False,
        'validation_checkpoint_predictions_replayed': False,
        'checkpoint_loading': 'weights_only=True; verified local project-generated artifact; no pickle fallback',
        'checks': ['frozen_files', 'exact_single_train_validation_test_window_roles',
                   'test_windows_excluded_from_selection_and_refit', 'two_LR_full_validation_subjects_each_epoch',
                   'independent_selection_and_early_stopping', 'final_train_is_train_plus_validation',
                   'final_history', 'full_DS28_67190', 'train_plus_validation_only_preprocessing',
                   'strict_state_load', 'native_CUDA_window_and_true_subject_mean_replay', 'fixed_threshold_predictions'],
    }
    save(receipt_path, result)
    print(json.dumps({'status': 'PASS', 'seed': seed, 'repeat_index': repeat_index, 'fold': fold,
                      'max_window_probability_error': error, 'checkpoint_sha256': result['checkpoint_sha256']}),
          ensure_ascii=False, flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=list(SEEDS), required=True)
    parser.add_argument('--repeat', dest='repeat_index', type=int, choices=list(REPEATS), required=True)
    parser.add_argument('--fold', type=int, choices=list(FOLDS), required=True)
    args = parser.parse_args()
    validate(args.seed, args.repeat_index, args.fold)


if __name__ == '__main__':
    main()

