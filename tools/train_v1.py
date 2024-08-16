import sys
import time
from argparse import ArgumentParser, Namespace
from pathlib import Path

from omegaconf import OmegaConf

from anomalib.config.config import _get_now_str
from anomalib.pre_processing.utils import find_class_dirs, validate_path

sys.path.append(Path(__file__).parent.as_posix())
from train import train


def write_config(args: Namespace):
    default_config_path = Path(__file__).parent.parent / "src/anomalib/models/patchcore/config.yaml"

    config = OmegaConf.load(default_config_path)

    data_dir = validate_path(args.data_dir)




























































    config_paths = []
    for class_dir in class_dirs:
        config.project.path = args.results_dir
        config.dataset.path = class_dir.parent.as_posix()
        config.dataset.category = class_dir.name
        config.dataset.normalization = args.normalization
        config.dataset.val_split_mode = args.val_split_mode
        config.dataset.val_split_ratio = args.val_split_ratio
        config.dataset.train_batch_size = args.train_batch_size
        config.dataset.eval_batch_size = args.eval_batch_size
        config.dataset.num_workers = args.num_workers
        config.model.coreset_sampling_ratio = args.coreset_sampling_ratio
        if class_dir.name == "kamera-atas":
            config.dataset.image_size = [240, 320]
        else:
            config.dataset.image_size = [170, 340]

        config_path = Path(__file__).parent / f"configs/{class_dir.name}_{_get_now_str(time.time())}.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(OmegaConf.to_yaml(config))
        config_paths.append(config_path)

    return config_paths


def main(args: Namespace):
    config_paths = write_config(args)
    for config_path in config_paths:
        train(config_path=config_path)


def parse_args():
    parser = ArgumentParser(description='Training AI Model')

    parser.add_argument('data_dir', type=str, help='Directory where the data is stored')
    parser.add_argument('--train_batch_size', type=int, default=64, help='Training batch size (default: 64)')
    parser.add_argument('--eval_batch_size', type=int, default=32, help='Evaluation batch size (default: 32)')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for data loading (default: 8)')
    parser.add_argument('--val_split_mode', type=str, choices=['same_as_test', 'from_test', 'synthetic'],
                        default='same_as_test', help='Validation split mode (default: same_as_test)')
    parser.add_argument('--val_split_ratio', type=float, default=0.5, help='Ratio for validation split (default: 0.5)')
    parser.add_argument('--coreset_sampling_ratio', type=float, default=0.1,
                        help='Coreset sampling ratio (default: 0.1)')
    parser.add_argument('--normalization', type=str, choices=['none', 'imagenet', 'gray'], default='gray',
                        help='Normalization method (default: gray)')
    parser.add_argument('--results_dir', type=str, default=r'D:\AutoInspection\SpringSheetMetal\ai_models',
                        help=r'Directory where results will be stored (default: D:\AutoInspection\SpringSheetMetal\ai_models)')

    return parser


if __name__ == '__main__':
    main(parse_args().parse_args())
