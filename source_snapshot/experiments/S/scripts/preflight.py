"""CPU-only engineering preflight; synthetic inputs only, never real EEG/results.

Default mode requires the completed freeze and the pinned runtime environment.
--before-freeze permits staging checks and records (rather than hides) runtime
version differences. Missing CPU dependencies may defer dynamic staging tests,
but never pass a formal preflight. No runner/validator task entry is executed.
"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
from dataclasses import asdict
import importlib
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import traceback

# This is a fresh CPU-only process. Do not initialize or enumerate any GPU.
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
sys.dont_write_bytecode = True

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
COMMON_API = {'sha', 'save', 'check_manifest', 'inputs', 'protocol', 'task_dir',
              'verify_frozen', 'environment_versions', 'check_environment_versions'}
CHECKPOINT_KEYS = {
    'schema_version', 'config_name', 'model_config', 'seed', 'fold', 'threshold',
    'n_parameters', 'train_indices', 'test_indices', 'inner_indices',
    'model_state_dict', 'preprocess', 'selected', 'environment', 'device_name',
    'environment_versions',
}
REQUIRED_OUTPUT_FILES = {'checkpoint.pt', 'windows.csv', 'subjects.csv', 'selection.json',
                         'inner_predictions.csv', 'final_training_history.csv', 'environment.json'}


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def tree(path):
    # Python's file loader accepts an optional UTF-8 BOM; retain source bytes.
    return ast.parse(Path(path).read_text(encoding='utf-8-sig'), filename=str(path))


def function_names(parsed):
    return {node.name for node in parsed.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}


def call_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = call_name(node.value)
        return (prefix + '.' if prefix else '') + node.attr
    return ''


def dict_assignment_keys(parsed, variable):
    keys = set()
    for node in ast.walk(parsed):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == variable for t in node.targets):
            if isinstance(node.value, ast.Dict):
                keys.update(k.value for k in node.value.keys if isinstance(k, ast.Constant) and isinstance(k.value, str))
        if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store) and isinstance(node.value, ast.Name) and node.value.id == variable:
            if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
                keys.add(node.slice.value)
    return keys


def static_checks():
    paths = [SCRIPTS / 'common.py', SCRIPTS / 'run_fold.py', SCRIPTS / 'validate_fold.py', Path(__file__)]
    trees = {p.name: tree(p) for p in paths}
    common, runner, validator = trees['common.py'], trees['run_fold.py'], trees['validate_fold.py']
    require(COMMON_API.issubset(function_names(common)), 'Common API is incomplete')
    require({'run', 'tagged_history', 'environment_record'}.issubset(function_names(runner)), 'Runner API missing')
    require({'validate', 'validate_selection', 'independently_derive_roles', 'independent_expansion',
             'independent_metrics', 'validate_protocol'}.issubset(function_names(validator)), 'Independent validator API missing')
    checkpoint_keys = dict_assignment_keys(runner, 'checkpoint')
    require(CHECKPOINT_KEYS.issubset(checkpoint_keys), 'Runner checkpoint schema missing: ' + ', '.join(sorted(CHECKPOINT_KEYS - checkpoint_keys)))
    manifest_keys = dict_assignment_keys(runner, 'manifest_hashes')
    require({'data_manifest_sha256', 'code_manifest_sha256', 'protocol_sha256', 'frozen_sha256'}.issubset(manifest_keys),
            'Runner manifest binding schema incomplete')
    calls = [node for node in ast.walk(runner) if isinstance(node, ast.Call)]
    require(any(call_name(c.func).split('.')[-1] == 'check_environment_versions' for c in calls),
            'Runner does not call the hard software-environment gate')
    output_literals = {node.right.value for node in ast.walk(runner)
                       if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div)
                       and isinstance(node.left, ast.Name) and node.left.id == 'directory'
                       and isinstance(node.right, ast.Constant) and isinstance(node.right.value, str)}
    require(REQUIRED_OUTPUT_FILES.issubset(output_literals), 'Runner/validator output filenames disagree')
    reads = [n for n in ast.walk(validator) if isinstance(n, ast.Call) and call_name(n.func) == 'torch.load']
    require(len(reads) == 1 and any(k.arg == 'weights_only' and isinstance(k.value, ast.Constant) and k.value.value is True
                                  for k in reads[0].keywords), 'Validator is not strict weights_only=True')
    forbidden = {'train_one_epoch', 'run_candidate_selection', 'train_final_and_predict', 'backward', 'step', 'fit'}
    require(not [n for n in ast.walk(validator) if isinstance(n, ast.Call) and call_name(n.func).split('.')[-1] in forbidden],
            'Independent validator contains a training/fitting call')
    for parsed in (runner, validator):
        # Imports/definitions/assignments are permitted at module scope; model
        # work must remain behind explicit functions and the __main__ guard.
        for node in parsed.body:
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                require(call_name(node.value.func).split('.')[-1] not in {'run', 'validate', 'main'},
                        'Task starts automatically on import')
    frozen_paths = [ROOT / 'frozen_ds' / x for x in ('dsmodel/models.py', 'dsmodel/transforms.py',
                    'dstraining/engine.py', 'dstraining/trainer.py', 'dsdata/protocol.py', 'dsdata/metrics.py')]
    for path in frozen_paths:
        require(path.is_file(), 'Frozen dependency missing: ' + str(path.relative_to(ROOT)))
        tree(path)
    hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in paths + frozen_paths}
    return {'status': 'PASS', 'source_files_AST_checked': len(paths) + len(frozen_paths),
            'source_sha256': hashes,
            'checkpoint_schema_keys_checked': sorted(CHECKPOINT_KEYS),
            'runner_environment_gate_present': True, 'validator_has_no_training_calls': True}


def import_from_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def fixture_roles(np, pd, validator):
    names = [f'SYNTHETIC_ONLY_{i:03d}' for i in range(31)]
    labels = {sid: int(i >= 16) for i, sid in enumerate(names)}
    counts = {sid: 98 if i < 15 else 97 for i, sid in enumerate(names)}
    groups = np.concatenate([np.repeat(sid, counts[sid]) for sid in names]).astype(str)
    y = np.concatenate([np.repeat(labels[sid], counts[sid]) for sid in names]).astype(np.int64)
    outer_rows, inner_rows = [], []
    for fold, held in enumerate(names, 1):
        outer_rows.append({'outer_fold': fold, 'outer_test_subject_id': held, 'true_label': labels[held]})
        remaining = [sid for sid in names if sid != held]
        for position, sid in enumerate(remaining):
            inner_rows.append({'outer_fold': fold, 'outer_test_subject_id': held,
                               'subject_id': sid, 'true_label': labels[sid],
                               'inner_validation_fold': position % 5 + 1})
    result = validator.independently_derive_roles(y, groups, pd.DataFrame(outer_rows), pd.DataFrame(inner_rows), 1)
    train, test, held, expected, label_map, count_map, roles, subject_fold = result
    require(len(groups) == 3022 and len(expected) == 30 and set(roles) == set(range(1, 6)), 'Synthetic role fixture failed')
    for tr, va in roles.values():
        require(not np.intersect1d(test, tr).size and not np.intersect1d(test, va).size, 'Synthetic role isolation failed')
    return result


def fixture_selection(pd, runner, validator, p, role_result):
    _, _, held, expected, labels, counts, _, subject_fold = role_result
    # Known analytic values, not calculated from real results. bAcc/AUC are
    # .5/.5 at epoch1, .5/1 at epoch2, and 1/1 at epoch3 onwards.
    candidates, untagged, records = [], [], []
    for lr in (.0003, .001):
        candidates.append({'learning_rate': lr, 'selected_epoch': 3, 'selection_stop_epoch': 15,
                           'best_validation_subject_balanced_accuracy': 1., 'best_validation_subject_auc': 1.})
        for epoch in range(1, 16):
            bacc, auc = ((.5, .5) if epoch == 1 else ((.5, 1.) if epoch == 2 else (1., 1.)))
            untagged.append({'epoch': epoch, 'mean_training_loss': .7 / epoch,
                            'validation_subject_balanced_accuracy': bacc, 'validation_subject_auc': auc,
                            'improved': epoch <= 3, 'patience_counter': max(0, epoch - 3),
                            'selection_stopped_here': epoch == 15})
            for sid in sorted(expected):
                label = labels[sid]
                probability = .5 if epoch == 1 else ((.6 if label else .51) if epoch == 2 else (.8 if label else .2))
                records.append({'learning_rate': lr, 'inner_fold': subject_fold[sid], 'epoch': epoch,
                                'subject_id': sid, 'true_label': label, 'prob_pd': probability,
                                'n_windows': counts[sid], 'predicted_label': int(probability >= .5)})
    history = runner.tagged_history(candidates, untagged)
    selection = {'selected': dict(candidates[0]), 'candidates': candidates, 'history': history,
                 'final_losses': [.6, .5, .4]}
    frame = pd.DataFrame(records)
    chosen, _ = validator.validate_selection(selection, frame, expected, labels, counts, subject_fold, p)
    require(chosen['learning_rate'] == .0003 and chosen['selected_epoch'] == 3, 'Synthetic tie/selection result wrong')
    rejected = []

    def must_reject(name, selected=selection, values=frame):
        try:
            validator.validate_selection(selected, values, expected, labels, counts, subject_fold, p)
        except (ValueError, KeyError, TypeError):
            rejected.append(name)
            return
        raise ValueError('Validator accepted corrupted synthetic fixture: ' + name)

    must_reject('missing_OOF_subject', values=frame.iloc[1:].copy())
    wrong = frame.copy()
    wrong.loc[0, 'subject_id'] = held
    must_reject('outer_test_subject_in_OOF', values=wrong)
    wrong = frame.copy()
    wrong.loc[0, 'true_label'] = 1 - int(wrong.loc[0, 'true_label'])
    must_reject('wrong_OOF_label', values=wrong)
    wrong = frame.copy()
    wrong.loc[0, 'inner_fold'] = int(wrong.loc[0, 'inner_fold']) % 5 + 1
    must_reject('wrong_inner_fold', values=wrong)
    wrong = frame.copy()
    wrong.loc[0, 'prob_pd'] = 1.1
    must_reject('invalid_probability', values=wrong)
    wrong = deepcopy(selection)
    wrong['history'][0]['validation_subject_auc'] = .7
    must_reject('wrong_history_metric', selected=wrong)
    wrong = deepcopy(selection)
    wrong['history'][-1]['selection_stopped_here'] = False
    must_reject('wrong_stop_marker', selected=wrong)
    wrong = deepcopy(selection)
    wrong['selected']['learning_rate'] = .001
    must_reject('wrong_LR_tie_winner', selected=wrong)
    wrong = deepcopy(selection)
    wrong['selected']['selected_epoch'] = 2
    must_reject('wrong_selected_epoch', selected=wrong)
    wrong = deepcopy(selection)
    wrong['history'][3]['patience_counter'] = 5
    must_reject('wrong_patience_counter', selected=wrong)
    return selection, {'status': 'PASS', 'synthetic_OOF_rows': len(frame),
                       'positive_fixture_passed': True, 'corrupt_fixtures_rejected': rejected}


def tensor_checkpoint_test(torch, np, runner, cfg, model, versions, selection, role_result, environment):
    train, test, _, _, _, _, roles, _ = role_result
    state = {name: value.detach().clone().cpu() for name, value in model.state_dict().items()}
    checkpoint = {
        'schema_version': 1, 'config_name': cfg.name, 'model_config': asdict(cfg),
        'seed': 11, 'fold': 1, 'threshold': .5, 'n_parameters': 67190,
        'train_indices': torch.as_tensor(train, dtype=torch.int64),
        'test_indices': torch.as_tensor(test, dtype=torch.int64),
        'inner_indices': [{'inner_fold': k, 'train_indices': torch.as_tensor(tr, dtype=torch.int64),
                           'val_indices': torch.as_tensor(va, dtype=torch.int64)} for k, (tr, va) in roles.items()],
        'model_state_dict': state, 'preprocess': {'mean': .01, 'std': .99},
        'selected': dict(selection['selected']), 'environment': environment,
        'device_name': 'SYNTHETIC_CPU_ONLY_NOT_VALIDATION', 'environment_versions': versions,
        **{key: '0' * 64 for key in ('data_manifest_sha256', 'code_manifest_sha256', 'frozen_sha256', 'protocol_sha256')},
    }
    require(CHECKPOINT_KEYS.issubset(checkpoint), 'Synthetic checkpoint schema incomplete')
    require(all(type(v) is str or v is None for v in versions.values()), 'Checkpoint versions must be native strings, not TorchVersion')
    require(hasattr(runner, 'safe_checkpoint_value'), 'Runner safe-checkpoint conversion helper missing')
    checkpoint = runner.safe_checkpoint_value(checkpoint)
    normalized = runner.safe_checkpoint_value({'integer': np.int64(3), 'float': np.float64(.1),
                                               'torch_version': torch.__version__})
    require(type(normalized['integer']) is int and type(normalized['float']) is float
            and type(normalized['torch_version']) is str, 'Runner metadata conversion is not weights_only-compatible')
    # A scoped synthetic temporary file is the only checkpoint ever opened here.
    audit = ROOT / 'audit'
    audit.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='synthetic_cpu_preflight_', dir=audit) as temporary:
        path = Path(temporary) / 'synthetic_checkpoint.pt'
        torch.save(checkpoint, path)
        restored = torch.load(path, map_location='cpu', weights_only=True)
        require(restored['model_config'] == asdict(cfg) and restored['environment_versions'] == versions,
                'Safe checkpoint roundtrip metadata failed')
        require(np.array_equal(restored['train_indices'].numpy(), train)
                and np.array_equal(restored['test_indices'].numpy(), test), 'Safe checkpoint index roundtrip failed')
        model.load_state_dict(restored['model_state_dict'], strict=True)
        require(all(torch.equal(state[name], restored['model_state_dict'][name]) for name in state),
                'Safe checkpoint parameter roundtrip failed')
    return {'status': 'PASS', 'weights_only': True, 'unsafe_fallback_used': False,
            'numpy_indices_in_checkpoint': False, 'real_checkpoint_opened': False}


def dynamic_checks(before_freeze):
    import numpy as np
    import pandas as pd
    import sklearn
    import torch

    torch.set_num_threads(1)
    require(torch.get_num_threads() == 1, 'CPU thread limit not applied')
    require(os.environ['CUDA_VISIBLE_DEVICES'] == '', 'GPU visibility was not disabled')
    sys.path.insert(0, str(SCRIPTS))
    sys.path.insert(0, str(ROOT / 'frozen_ds'))
    common = importlib.import_module('common')
    runner = import_from_file('_ds28_preflight_runner', SCRIPTS / 'run_fold.py')
    validator = import_from_file('_ds28_preflight_validator', SCRIPTS / 'validate_fold.py')
    from dsmodel import models, transforms
    from dstraining import trainer
    p = common.protocol()
    runner.check_protocol(p)
    validator.validate_protocol(p)
    versions = {'torch': str(torch.__version__), 'cuda_runtime': str(torch.version.cuda) if torch.version.cuda else None,
                'numpy': str(np.__version__), 'pandas': str(pd.__version__), 'scikit_learn': str(sklearn.__version__)}
    mismatch = {key: {'actual': versions.get(key), 'expected': value}
                for key, value in p['environment_versions'].items() if versions.get(key) != value}
    if not before_freeze:
        common.check_environment_versions()
        require(not mismatch, 'Formal CPU preflight software mismatch')
    cfg = models.get_config('DS28_BR_MS_AUG_LS_WD3')
    require(cfg.input_transform == 'band_raw' and cfg.n_channels == 18 and cfg.params == {'wd': .001}
            and cfg.augment == 'light' and cfg.label_smoothing == .05 and cfg.preprocess == 'global_scalar',
            'Frozen complete DS28 configuration mismatch')
    model = models.build_model(cfg, torch.device('cpu'))
    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    state_count = sum(tensor.numel() for tensor in model.state_dict().values())
    require(parameter_count == state_count == 67190, 'DS28 model/state parameter counts are not 67190')
    synthetic = np.random.default_rng(20261109).normal(size=(4, 3, 512)).astype(np.float32)
    independent = validator.independent_expansion(synthetic)
    original = transforms.transform_input(synthetic, 'band_raw', n_bands=5)
    require(np.array_equal(independent, original), 'Independent FFT expansion differs from frozen implementation')
    transform, meta = trainer.fit_preprocess(synthetic[:3], cfg.preprocess, cfg.input_transform, n_bands=cfg.n_bands)
    expanded = validator.independent_expansion(synthetic[:3])
    mean = float(np.mean(expanded, dtype=np.float64))
    std = float(np.std(expanded, dtype=np.float64) + 1e-8)
    require(abs(meta['mean'] - mean) <= 1e-12 and abs(meta['std'] - std) <= 1e-12, 'Synthetic train-only statistics disagree')
    expected_test = np.asarray((validator.independent_expansion(synthetic[3:]) - mean) / std, dtype=np.float32)
    require(np.array_equal(transform(synthetic[3:]), expected_test), 'Synthetic train-only preprocessing differs')
    role_result = fixture_roles(np, pd, validator)
    selection, selection_audit = fixture_selection(pd, runner, validator, p, role_result)
    environment = {'scope': 'SYNTHETIC_CPU_PREFLIGHT_ONLY', 'environment_versions': versions,
                   'torch_version': versions['torch'], 'torch_cuda_version': versions['cuda_runtime'],
                   'device_name': 'SYNTHETIC_CPU_ONLY_NOT_VALIDATION'}
    roundtrip = tensor_checkpoint_test(torch, np, runner, cfg, model, versions, selection, role_result, environment)
    del model
    return {'status': 'PASS', 'versions': versions, 'frozen_versions_match': not mismatch,
            'versions_mismatch': mismatch, 'cpu_threads': int(torch.get_num_threads()),
            'DS28_parameter_count': parameter_count, 'DS28_state_elements': state_count,
            'independent_FFT_array_equal': True, 'independent_preprocessing_array_equal': True,
            'synthetic_selection_checks': selection_audit, 'safe_checkpoint_test': roundtrip,
            'model_training_called': False, 'model_forward_called': False,
            'real_signal_arrays_opened': False, 'GPU_initialized_or_used': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--before-freeze', action='store_true')
    mode.add_argument('--remote-exact', action='store_true',
                      help='Strict frozen software/manifests with CPU-only tests; no CUDA device required')
    arguments = parser.parse_args()
    report = {'scope': 'CPU_SYNTHETIC_ENGINEERING_PREFLIGHT_ONLY',
              'before_freeze': arguments.before_freeze, 'remote_exact': arguments.remote_exact,
              'training_called': False,
              'real_cohort_read': False, 'actual_result_metrics_computed': False}
    try:
        report['static'] = static_checks()
        if not arguments.before_freeze:
            sys.path.insert(0, str(SCRIPTS))
            common = importlib.import_module('common')
            common.verify_frozen()
            report['frozen_manifest_check'] = 'PASS'
        missing = [name for name in ('numpy', 'pandas', 'sklearn', 'torch') if importlib.util.find_spec(name) is None]
        if missing:
            require(arguments.before_freeze, 'Formal preflight dependencies missing: ' + ', '.join(missing))
            report['status'] = 'STATIC_PASS_CPU_RUNTIME_DEFERRED'
            report['dynamic'] = {'status': 'NOT_RUN', 'missing_modules': missing,
                                 'action': 'Run this same CPU preflight in the prepared native server environment before training.'}
        else:
            report['dynamic'] = dynamic_checks(arguments.before_freeze)
            report['status'] = 'PASS'
        common = importlib.import_module('common')
        name = 'preflight_before_freeze.json' if arguments.before_freeze else (
            'preflight_remote_exact.json' if arguments.remote_exact else 'preflight_formal.json')
        common.save(ROOT / 'audit' / name, report)
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    except BaseException as exc:
        report['status'] = 'FAIL'
        report['error_class'] = type(exc).__name__
        report['error'] = str(exc)
        try:
            common = importlib.import_module('common')
            name = 'preflight_before_freeze.json' if arguments.before_freeze else (
                'preflight_remote_exact.json' if arguments.remote_exact else 'preflight_formal.json')
            common.save(ROOT / 'audit' / name, report)
        except Exception as receipt_error:
            report['receipt_error_class'] = type(receipt_error).__name__
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
        traceback.print_exc()
        raise SystemExit(1)


if __name__ == '__main__':
    main()
