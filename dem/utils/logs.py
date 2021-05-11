import os
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator


def find_event_files(parent_dir, version: int):
    ver = "version_" + str(version)
    path = os.path.join(parent_dir, "lightning_logs", ver)
    files = os.listdir(path)
    L = len(files)
    out = []
    for j in range(L):
        if "events.out" in files[j]:
            fp = os.path.join(path, files[j])
            out += [fp]
    return out


def find_event_file(parent_dir, version):
    files = find_event_files(parent_dir, version)
    if len(files) == 0:
        raise RuntimeError("no event files found in " + parent_dir)
    if len(files) > 1:
        raise RuntimeError("multiple event files found in " + parent_dir)
    return files[0]


def read_logged_events(parent_dir="out", version: int = 0, size_guidance=None):
    file = find_event_file(parent_dir, version)
    ea = event_accumulator.EventAccumulator(path=file, size_guidance=size_guidance)
    ea.Reload()  # loads events from file
    return ea


def read_logged_scalar(name="valid_loss", parent_dir="out", version: int = 0):
    ea = read_logged_events(parent_dir, version)
    return pd.DataFrame(ea.Scalars(name))
