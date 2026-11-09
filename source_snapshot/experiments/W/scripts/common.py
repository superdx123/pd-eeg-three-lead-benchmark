"""Immutable ds002778 W-only input and source-role contracts; no training here."""
from pathlib import Path
import hashlib
import json
import os

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(4194304), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path,obj):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError("OUTPUT_OUTSIDE_NEW_W_EXPERIMENT")
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    os.replace(temporary,path)


def protocol():
    return json.loads((ROOT/"config/protocol.json").read_text(encoding="utf-8"))


def task_dir(seed,repeat,fold):
    if int(seed) not in (11,22,33,44,55) or int(repeat) not in range(1,6) or int(fold) not in range(1,6):
        raise ValueError("TASK_NOT_IN_FIXED_W_GRID")
    return ROOT/"outputs"/f"seed_{int(seed)}"/f"repeat_{int(repeat):02d}"/f"fold_{int(fold):02d}"


def check_manifest(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    members = value["files"] if isinstance(value.get("files"),dict) else value
    if not members:
        raise ValueError("EMPTY_MANIFEST")
    for name,digest in members.items():
        target = (ROOT/name).resolve()
        if not target.is_relative_to(ROOT.resolve()) or not target.is_file() or target.is_symlink():
            raise ValueError("MANIFEST_TARGET_NOT_LOCAL_FILE: "+name)
        if sha(target) != digest:
            raise ValueError("MANIFEST_SHA_MISMATCH: "+name)
    return value


def freeze(paths,target):
    members = {Path(path).resolve().relative_to(ROOT.resolve()).as_posix():sha(path) for path in sorted(paths)}
    save(target,members)
    return members


def verify_frozen():
    value = check_manifest(ROOT/"FROZEN.json")
    for name in ("data_manifest.json","code_manifest.json","config/protocol.json","config/tasks.json","input_origins.json"):
        if name not in value:
            raise ValueError("FREEZE_MISSING_REQUIRED_MEMBER: "+name)
    check_manifest(ROOT/"data_manifest.json")
    check_manifest(ROOT/"code_manifest.json")
    return value


def role_indices(w_manifest,repeat,fold):
    """Original source roles plus independent per-window coverage assertions."""
    import numpy as np
    import sys
    if int(repeat) not in range(1,6) or int(fold) not in range(1,6):
        raise ValueError("W_ROLE_OUTSIDE_GRID")
    sys.path.insert(0,str(ROOT/"frozen_ds"))
    from dsdata.protocol import get_protocol_w_role_indices
    roles = get_protocol_w_role_indices(w_manifest,int(repeat),int(fold),n_folds=5)
    block = w_manifest[w_manifest.repeat_index.astype(int) == int(repeat)]
    if len(block) != 3022 or block.window_index.astype(int).duplicated().any():
        raise ValueError("W_REPEAT_COVERAGE_MISMATCH")
    expected = {"test":np.sort(block.loc[block.test_fold.astype(int) == int(fold),"window_index"].to_numpy(dtype=int)),
                "validation":np.sort(block.loc[block.test_fold.astype(int) == int(fold)%5+1,"window_index"].to_numpy(dtype=int))}
    expected["train"] = np.setdiff1d(np.arange(3022),np.concatenate([expected["test"],expected["validation"]]))
    if set(roles) != {"train","validation","test"}:
        raise ValueError("W_ROLE_KEYS_MISMATCH")
    for role in expected:
        if not len(roles[role]) or not np.array_equal(np.asarray(roles[role],dtype=int),expected[role]):
            raise ValueError("FROZEN_SOURCE_W_ROLE_MISMATCH: "+role)
    combined = np.concatenate([roles["train"],roles["validation"],roles["test"]])
    if len(combined) != 3022 or not np.array_equal(np.sort(combined),np.arange(3022)):
        raise ValueError("W_WINDOW_ROLE_LEAKAGE")
    return roles


def inputs():
    import numpy as np
    import pandas as pd
    verify_frozen()
    p = protocol()
    if p["dataset_id"] != "ds002778" or p["protocol"] != "W_subject_overlap_window_cv" or p["threshold"] != 0.5:
        raise ValueError("UNAUTHORIZED_W_PROTOCOL")
    with np.load(ROOT/"data/ds002778_preprocessed_7ch_2s.npz",allow_pickle=False) as z:
        channels = z["channel_names"].astype(str).tolist()
        if channels != ["FC1","FC2","C3","Cz","C4","CP1","CP2"]:
            raise ValueError("STORED_CHANNEL_ORDER_MISMATCH")
        stored = z["X"]
        if stored.shape != (3022,7,512) or stored.dtype != np.float32:
            raise ValueError("STORED_SIGNAL_SCHEMA_MISMATCH")
        x = stored[:,[channels.index(c) for c in ("C3","Cz","C4")],:]
        y,g = z["y"].copy(),z["groups"].astype(str)
        if float(z["sfreq"].reshape(-1)[0]) != 256 or float(z["window_sec"].reshape(-1)[0]) != 2:
            raise ValueError("SAMPLING_OR_WINDOW_MISMATCH")
    if x.shape != (3022,3,512) or y.shape != (3022,) or y.dtype != np.int64 or g.shape != (3022,):
        raise ValueError("CLASSIFIER_VIEW_SCHEMA_MISMATCH")
    if not np.isfinite(x).all() or set(np.unique(y)) != {0,1} or len(np.unique(g)) != 31:
        raise ValueError("SIGNAL_LABEL_SUBJECT_SCHEMA_MISMATCH")
    labels = {sid:int(np.unique(y[g == sid]).item()) for sid in np.unique(g)}
    if list(labels.values()).count(0) != 16 or list(labels.values()).count(1) != 15:
        raise ValueError("SUBJECT_CLASS_COUNTS_MISMATCH")
    wm = pd.read_csv(ROOT/"data/protocol_w_window_manifest.csv")
    fields = ["protocol","repeat_index","split_seed","window_index","subject_id","true_label","test_fold"]
    if wm.columns.tolist() != fields or len(wm) != 15110 or set(wm.protocol.astype(str)) != {p["protocol"]}:
        raise ValueError("W_MANIFEST_SCHEMA_MISMATCH")
    if set(wm.repeat_index.astype(int)) != set(range(1,6)) or set(wm.test_fold.astype(int)) != set(range(1,6)):
        raise ValueError("W_REPEAT_FOLD_RANGE_MISMATCH")
    for repeat,split_seed in enumerate((11001,22002,33003,44004,55005),1):
        rows = wm[wm.repeat_index.astype(int) == repeat]
        idx = rows.window_index.to_numpy(dtype=int)
        if len(rows) != 3022 or not np.array_equal(np.sort(idx),np.arange(3022)) or set(rows.split_seed.astype(int)) != {split_seed}:
            raise ValueError("W_REPEAT_WINDOW_OR_SEED_MISMATCH")
        if not np.array_equal(rows.subject_id.astype(str).to_numpy(),g[idx]) or not np.array_equal(rows.true_label.to_numpy(dtype=int),y[idx]):
            raise ValueError("W_WINDOW_SUBJECT_LABEL_MAPPING_MISMATCH")
        for fold in range(1,6):
            roles = role_indices(wm,repeat,fold)
            if any(set(y[indices]) != {0,1} for indices in roles.values()):
                raise ValueError("W_ROLE_SINGLE_CLASS")
    return x,y,g,wm


def environment_versions():
    import torch,numpy,pandas,sklearn
    return {"torch":torch.__version__,"cuda_runtime":torch.version.cuda,"numpy":numpy.__version__,
            "pandas":pandas.__version__,"scikit_learn":sklearn.__version__}


def check_environment_versions():
    actual = environment_versions()
    if actual != protocol()["environment_versions"]:
        raise ValueError("FROZEN_ENVIRONMENT_MISMATCH")
    return actual
