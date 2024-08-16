from argparse import ArgumentParser, Namespace
from pathlib import Path

from anomalib.api.defect.preprocess import split_data, generate_mask_anomalib, clear_heatmap
from anomalib.pre_processing.utils import validate_path, find_class_dirs, process_data_directory


def prepare(data_dir, test_size, random_state):
    clear_heatmap(data_dir)
    split_data(data_dir, test_size=test_size, random_state=random_state)
    generate_mask_anomalib(data_dir)


def main(args: Namespace):
    print("Starting process...")
    data_dir = validate_path(args.data_dir)

    # Check if the directory exists
    if not data_dir.is_dir():
        print(f"Error: Directory '{data_dir}' does not exist.")
        return

    class_dirs = find_class_dirs(data_dir)
    for class_dir in class_dirs:
        print(f"Processing directory: {class_dir}")
        process_data_directory(
            class_dir,
            lambda dir_: prepare(dir_, args.test_size, args.random_state),
            output_dir=args.output_dir
        )
        print(f"Process completed.")


if __name__ == '__main__':
    parser = ArgumentParser(description="Process Data.")
    parser.add_argument('data_dir', type=Path, help='The directory containing the data')
    parser.add_argument('--test_size', type=float, default=0.7,
                        help='Proportion of the dataset to include in the test split (default: 0.7)')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--output_dir', type=Path, default=None, help='The output directory containing the data')
    args = parser.parse_args()
    main(args)
