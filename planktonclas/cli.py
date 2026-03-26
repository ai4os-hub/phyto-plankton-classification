"""
Command-line interface for planktonclas.
"""

import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime

from planktonclas import config, paths


PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PROJECT_CONFIG_NAME = "config.yaml"
DEFAULT_NOTEBOOKS_DIR = os.path.join(PACKAGE_ROOT, "notebooks")
DEFAULT_DEMO_IMAGES_DIR = os.path.join(PACKAGE_ROOT, "data", "demo-images")
DEFAULT_DEMO_SPLITS_DIR = os.path.join(PACKAGE_ROOT, "data", "dataset_files")


def _default_config_path():
    cwd_config = os.path.abspath(DEFAULT_PROJECT_CONFIG_NAME)
    if os.path.exists(cwd_config):
        return cwd_config
    return config.DEFAULT_CONFIG_PATH


def _apply_config(conf_path):
    config.set_config_path(conf_path)
    paths.CONF = config.get_conf_dict()


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _write_placeholder(path, contents=""):
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(contents)


def _copy_tree(src, dst):
    if not os.path.isdir(src):
        raise FileNotFoundError(f"Missing resource directory: {src}")
    shutil.copytree(src, dst, dirs_exist_ok=True)


def init_project(args):
    target_dir = os.path.abspath(args.directory)
    config_path = os.path.join(target_dir, DEFAULT_PROJECT_CONFIG_NAME)

    if os.path.exists(config_path) and not args.force:
        raise FileExistsError(
            f"{config_path} already exists. Use --force to overwrite it."
        )

    _ensure_dir(target_dir)
    _ensure_dir(os.path.join(target_dir, "data", "images"))
    _ensure_dir(os.path.join(target_dir, "data", "dataset_files"))
    _ensure_dir(os.path.join(target_dir, "models"))

    shutil.copyfile(config.DEFAULT_CONFIG_PATH, config_path)

    if args.demo:
        _copy_tree(DEFAULT_DEMO_IMAGES_DIR, os.path.join(target_dir, "data", "images"))
        _copy_tree(
            DEFAULT_DEMO_SPLITS_DIR,
            os.path.join(target_dir, "data", "dataset_files"),
        )
    else:
        _write_placeholder(
            os.path.join(target_dir, "data", "dataset_files", "classes.txt"),
            "# one class name per line\n",
        )
        _write_placeholder(
            os.path.join(target_dir, "data", "dataset_files", "train.txt"),
            "# relative/image/path.jpg 0\n",
        )

    print(f"Initialized project at: {target_dir}")
    print(f"Config: {config_path}")
    print(f"Images: {os.path.join(target_dir, 'data', 'images')}")
    print(f"Dataset files: {os.path.join(target_dir, 'data', 'dataset_files')}")
    print(f"Models: {os.path.join(target_dir, 'models')}")
    if args.demo:
        print("Demo data copied into data/images and data/dataset_files.")


def validate_config(args):
    conf_path = os.path.abspath(args.config or _default_config_path())
    _apply_config(conf_path)
    print(f"Configuration OK: {config.CONF_PATH}")
    print(f"Base directory: {paths.get_base_dir()}")
    print(f"Images directory: {paths.get_images_dir()}")
    print(f"Splits directory: {paths.get_splits_dir()}")
    print(f"Models directory: {paths.get_models_dir()}")


def train_model(args):
    conf_path = os.path.abspath(args.config or _default_config_path())
    _apply_config(conf_path)

    from planktonclas.train_runfile import train_fn

    conf = config.get_conf_dict()
    conf["dataset"]["num_workers"] = args.workers
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    train_fn(TIMESTAMP=timestamp, CONF=conf)


def generate_report_cmd(args):
    conf_path = os.path.abspath(args.config or _default_config_path())
    _apply_config(conf_path)

    from planktonclas.report_utils import generate_report

    summary = generate_report(timestamp=args.timestamp)
    print(f"Report generated for timestamp: {summary['timestamp']}")
    print(f"Results: {summary['results_dir']}")
    print(f"Predictions: {summary['predictions_file']}")
    print(f"Top-1 accuracy: {summary['top1_accuracy']:.3f}")
    print(f"Top-3 accuracy: {summary['top3_accuracy']:.3f}")
    print(f"Top-5 accuracy: {summary['top5_accuracy']:.3f}")
    print(f"Macro F1: {summary['macro_f1']:.3f}")
    print(f"Weighted F1: {summary['weighted_f1']:.3f}")


def run_api(args):
    conf_path = os.path.abspath(args.config or _default_config_path())
    env = os.environ.copy()
    env[config.CONFIG_ENV_VAR] = conf_path
    env["DEEPAAS_V2_MODEL"] = "planktonclas"

    command = ["deepaas-run", "--listen-ip", args.host]
    if args.port is not None:
        command.extend(["--listen-port", str(args.port)])

    raise SystemExit(subprocess.call(command, env=env))


def list_models(args):
    conf_path = os.path.abspath(args.config or _default_config_path())
    _apply_config(conf_path)

    models_dir = paths.get_models_dir()
    if not os.path.isdir(models_dir):
        print(f"No models directory found: {models_dir}")
        return

    entries = sorted(
        [
            name
            for name in os.listdir(models_dir)
            if os.path.isdir(os.path.join(models_dir, name))
        ]
    )
    if not entries:
        print(f"No models found in: {models_dir}")
        return

    print(f"Models in {models_dir}:")
    for name in entries:
        print(name)


def notebooks(args):
    print(DEFAULT_NOTEBOOKS_DIR)


def build_parser():
    parser = argparse.ArgumentParser(prog="planktonclas")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser(
        "init", help="Create a local planktonclas project structure."
    )
    init_parser.add_argument("directory", nargs="?", default=".")
    init_parser.add_argument(
        "--demo",
        action="store_true",
        help="Populate the project with demo images and demo dataset files.",
    )
    init_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing config.yaml in the target directory.",
    )
    init_parser.set_defaults(func=init_project)

    validate_parser = subparsers.add_parser(
        "validate-config", help="Validate a config file and print resolved paths."
    )
    validate_parser.add_argument("--config")
    validate_parser.set_defaults(func=validate_config)

    train_parser = subparsers.add_parser(
        "train", help="Train a model using a config file."
    )
    train_parser.add_argument("--config")
    train_parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of dataset preprocessing workers.",
    )
    train_parser.set_defaults(func=train_model)

    report_parser = subparsers.add_parser(
        "report",
        help="Generate evaluation plots and metrics for a trained run.",
    )
    report_parser.add_argument("--config")
    report_parser.add_argument(
        "--timestamp",
        help="Timestamped model directory to report on. Defaults to the latest run.",
    )
    report_parser.set_defaults(func=generate_report_cmd)

    api_parser = subparsers.add_parser(
        "api", help="Launch the DEEPaaS API with a selected config file."
    )
    api_parser.add_argument("--config")
    api_parser.add_argument("--host", default="0.0.0.0")
    api_parser.add_argument("--port", type=int, default=5000)
    api_parser.set_defaults(func=run_api)

    models_parser = subparsers.add_parser(
        "list-models", help="List models inside the configured models directory."
    )
    models_parser.add_argument("--config")
    models_parser.set_defaults(func=list_models)

    notebooks_parser = subparsers.add_parser(
        "notebooks", help="Print the package notebooks directory."
    )
    notebooks_parser.set_defaults(func=notebooks)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
