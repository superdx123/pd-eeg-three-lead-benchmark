"""Shared I/O and immutable-input gates for the new DS28 778 S reproduction."""
from pathlib import Path
import hashlib
import json
import os

ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):
            h.update(block)
    return h.hexdigest()

def save(path,obj):
    path=Path(path).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError('OUTPUT_OUTSIDE_NEW_EXPERIMENT')
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(temporary,path)

def protocol():
    return json.loads((ROOT/'config/protocol.json').read_text(encoding='utf-8'))

def task_dir(seed,fold):
    if int(seed) not in [11,22,33,44,55] or int(fold) not in range(1,32):
        raise ValueError('TASK_NOT_IN_FIXED_GRID')
    return ROOT/'outputs'/f'seed_{int(seed)}'/f'fold_{int(fold):02d}'

def check_manifest(path):
    value=json.loads(Path(path).read_text(encoding='utf-8'))
    members=value['files'] if isinstance(value.get('files'),dict) else value
    if not members:
        raise ValueError('EMPTY_MANIFEST')
    for name,digest in members.items():
        target=(ROOT/name).resolve()
        if not target.is_relative_to(ROOT.resolve()) or not target.is_file():
            raise ValueError('MANIFEST_TARGET_NOT_LOCAL_FILE: '+name)
        if sha(target)!=digest:
            raise ValueError('MANIFEST_SHA_MISMATCH: '+name)
    return value

def freeze(paths,target):
    members={str(Path(path).resolve().relative_to(ROOT.resolve())).replace('\\','/'):sha(path) for path in sorted(paths)}
    save(target,members)
    return members

def verify_frozen():
    value=check_manifest(ROOT/'FROZEN.json')
    for name in ['data_manifest.json','code_manifest.json','config/protocol.json','config/tasks.json','input_origins.json']:
        if name not in value:
            raise ValueError('FREEZE_MISSING_REQUIRED_MEMBER: '+name)
    check_manifest(ROOT/'data_manifest.json')
    check_manifest(ROOT/'code_manifest.json')
    return value

def inputs():
    import numpy as np
    import pandas as pd
    verify_frozen()
    p=protocol()
    if p['dataset_id']!='ds002778' or p['threshold']!=0.5:
        raise ValueError('UNAUTHORIZED_DATA_OR_THRESHOLD')
    with np.load(ROOT/'data/ds002778_preprocessed_7ch_2s.npz',allow_pickle=False) as z:
        channels=[str(x) for x in z['channel_names']]
        if channels!=['FC1','FC2','C3','Cz','C4','CP1','CP2']:
            raise ValueError('STORED_CHANNEL_ORDER_MISMATCH')
        indices=[channels.index(c) for c in ['C3','Cz','C4']]
        stored=z['X']
        if stored.shape!=(3022,7,512) or stored.dtype!=np.float32:
            raise ValueError('STORED_SIGNAL_SCHEMA_MISMATCH')
        X=stored[:,indices,:]
        y=z['y'].copy()
        g=z['groups'].astype(str)
        if float(z['sfreq'].reshape(-1)[0])!=256 or float(z['window_sec'].reshape(-1)[0])!=2:
            raise ValueError('SAMPLING_OR_WINDOW_MISMATCH')
    # This is an indexing-only view of already z-scored input, never a new preprocessing pass.
    if X.shape!=(3022,3,512) or y.shape!=(3022,) or y.dtype!=np.int64 or g.shape!=(3022,):
        raise ValueError('CLASSIFIER_VIEW_SCHEMA_MISMATCH')
    if not np.isfinite(X).all() or set(np.unique(y))!={0,1} or len(np.unique(g))!=31:
        raise ValueError('SIGNAL_LABEL_OR_SUBJECT_SCHEMA_MISMATCH')
    subject_labels={sid:int(np.unique(y[g==sid]).item()) for sid in np.unique(g)}
    if list(subject_labels.values()).count(0)!=16 or list(subject_labels.values()).count(1)!=15:
        raise ValueError('SUBJECT_CLASS_COUNTS_MISMATCH')
    so=pd.read_csv(ROOT/'data/protocol_s_outer_subject_manifest.csv')
    si=pd.read_csv(ROOT/'data/protocol_s_inner_subject_manifest.csv')
    if len(so)!=31 or sorted(so.outer_fold.astype(int).tolist())!=list(range(1,32)) or so.outer_test_subject_id.duplicated().any():
        raise ValueError('OUTER_MANIFEST_MISMATCH')
    if set(so.outer_test_subject_id.astype(str))!=set(g):
        raise ValueError('OUTER_SUBJECT_MEMBERSHIP_MISMATCH')
    import sys
    sys.path.insert(0,str(ROOT/'frozen_ds'))
    from dsdata.protocol import build_s_inner_splits,get_protocol_s_outer_subjects
    for f in range(1,32):
        roles=get_protocol_s_outer_subjects(so,g,f)
        test=roles['outer_test_subject'];expected=np.flatnonzero(g!=test)
        rows=si[si.outer_fold.astype(int)==f]
        if len(rows)!=30 or rows.subject_id.duplicated().any() or set(rows.subject_id.astype(str))!=set(g[expected]):
            raise ValueError('INNER_SUBJECT_MANIFEST_MISMATCH')
        validation=[]
        for _,train,val in build_s_inner_splits(si,g,f):
            if set(g[train])&set(g[val]) or test in g[train] or test in g[val]:
                raise ValueError('INNER_SUBJECT_LEAKAGE')
            if set(y[train])!={0,1} or set(y[val])!={0,1}:
                raise ValueError('INNER_SINGLE_CLASS')
            validation.extend(val.tolist())
        if not np.array_equal(np.sort(validation),expected):
            raise ValueError('INNER_OOF_COVERAGE_MISMATCH')
    return X,y,g,so,si

def environment_versions():
    import torch,numpy,pandas,sklearn
    return {'torch':torch.__version__,'cuda_runtime':torch.version.cuda,'numpy':numpy.__version__,
            'pandas':pandas.__version__,'scikit_learn':sklearn.__version__}

def check_environment_versions():
    actual=environment_versions()
    expected=protocol()['environment_versions']
    if actual!=expected:
        raise ValueError('FROZEN_ENVIRONMENT_MISMATCH')
    return actual
