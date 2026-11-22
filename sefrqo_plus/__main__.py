import argparse
import json
import os
import random
from pathlib import Path

from .models import load_generator
from .online import OnlineOptimizer
from .records import RecordStore

REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve_path(path):
    p = Path(path)
    return str(p if p.is_absolute() else REPO_ROOT / p)


def load_config(path):
    with open(path) as f:
        cfg = json.load(f)
    cfg["system_prompt"] = resolve_path(cfg["system_prompt"])
    cfg["history"] = [resolve_path(p) for p in cfg["history"]]
    cfg["records"] = resolve_path(cfg["records"])
    if cfg.get("overhead_log"):
        cfg["overhead_log"] = resolve_path(cfg["overhead_log"])
    for db in cfg["databases"]:
        db["workloads"] = [resolve_path(p) for p in db["workloads"]]
    model = cfg["model"]
    for key in ("path", "base_path"):
        if key in model and model[key].startswith(("models/", "./")):
            model[key] = resolve_path(model[key])
    return cfg


def main():
    parser = argparse.ArgumentParser(
        description="SEFRQO-Plus online query optimization"
    )
    parser.add_argument("config", help="JSON file under configs/")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["name"] = Path(args.config).stem

    if cfg["mode"] == "dynamic":
        random.seed(cfg["seed"])
    os.makedirs(os.path.dirname(cfg["records"]), exist_ok=True)
    if cfg.get("reset_records") and os.path.exists(cfg["records"]):
        os.remove(cfg["records"])
    if cfg.get("overhead_log"):
        os.makedirs(cfg["overhead_log"], exist_ok=True)
    store = RecordStore(cfg["records"], cfg.get("encoder", "text"))
    generate = load_generator(cfg["model"])
    store.create_collections(cfg["history"])

    optimizer = OnlineOptimizer(cfg, store, generate)
    if cfg["mode"] == "static":
        optimizer.run_static()
    else:
        optimizer.run_dynamic()

    os.remove(cfg["records"])


if __name__ == "__main__":
    main()
