from download import *
from download.constants import _SUPPORTED_DOWNLOAD_DATASETS
import sys
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, default=None, help="The full set of datasets to use for the benchmark, default to use all supported datasets.")
    parser.add_argument('--root', type=str, default=None, help="to set the download dataset directory")
    parser.add_argument('--restore', default=False, action="store_true", help="verbose mode")
    args = parser.parse_args()
    
    if not args.datasets:
        datasets = _SUPPORTED_DOWNLOAD_DATASETS
    else:
        datasets = args.datasets.split(',')
    
    for ds in datasets:
        if args.root:
            download_and_prepare_data(ds, restore=args.restore, _dataset_root_dir=args.root)
        else:
            download_and_prepare_data(ds, restore=args.restore)

if __name__ == "__main__":
    sys.exit(main())