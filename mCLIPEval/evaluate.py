from dataset import *
from models import *
from metrics import *
from dataset.constants import _SUPPORTED_DATASETS
from models.constants import _SUPPORTED_MODELS

import sys, json
import argparse
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

_FUNC_ = {
    "CLASSIFICATION": zeroshot_classification,
    "RETRIEVAL": zeroshot_retrieval,
    "COMPOSITIONALITY": zeroshot_composition
}

def evaluate(batch_size, num_workers, model_config, root=None, dataset_names=None, task_names=None, group_names=None, languages=None, verbose=False, restore=False):
    dump_metrics = {}
    if model_config:
        eval_model = EvalModel(model_config=model_config)
        model = eval_model.model
        model.initialize()
        dump_metrics['model_info'] = eval_model.__info__()
    else:
        return
    
    if not dataset_names:
        dataset_names = _SUPPORTED_DATASETS
    
    eval_datasets = EvalDataset(root=root, dataset_names=dataset_names, task_names=task_names, group_names=group_names, languages=languages, verbose=verbose)

    for dataset in eval_datasets.datasets:
        function = _FUNC_.get(dataset.group, None)
        if function:
            if verbose or restore:
                dir_name = f'eval.{model.name}'
                if not os.path.isdir(dir_name):
                    os.mkdir(dir_name)
                if os.path.exists(f'{dir_name}/{dataset.name}.json'):
                    res = json.loads(open(f'{dir_name}/{dataset.name}.json').read())
                    dump_metrics[dataset.name]=res
                    print(f'Skip {dataset.name} for {model.name}')
                    continue
            res = function(
                    model=model, 
                    dataset=dataset, 
                    batch_size=batch_size, 
                    num_workers=num_workers, 
                    verbose=verbose
                )
            if verbose or restore:
                with open(f'{dir_name}/{dataset.name}.json', "w") as f:
                    json.dump(res, f)

            dump_metrics[dataset.name]=res
    return dump_metrics
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, default=None, help="The full set of datasets to use for the benchmark, default to use all supported datasets.")
    parser.add_argument('--root', type=str, default=None, help="The root directory path where the datasets are downloaded.")
    parser.add_argument('--groups', type=str, default=None, help="The groups to choose datasets from the full set.")
    parser.add_argument('--languages', type=str, default=None, help="The languages to choose datasets from the full set.")
    parser.add_argument('--tasks', type=str, default=None, help="The tasks to choose datasets from the full set.")
    parser.add_argument('--prefix', type=str, default=None, help="Choose datasets with prefix, e.g. muti30k, imagenet.")
    
    parser.add_argument('--model_name', type=str, default=None, help="The name of model.")
    parser.add_argument('--model_script', type=str, default=None, help="The model script to use to load the model from directory.")
    parser.add_argument('--model_dir', type=str, default=None, help="The path to directory to load model config file.")
    parser.add_argument('--agency', type=str, default=None, help="The agency of model.")
    parser.add_argument('--text_encoder', type=str, default=None, help="The text encoder of model.")
    parser.add_argument('--vision_encoder', type=str, default=None, help="The vision encoder of model.")

    parser.add_argument('--output', type=str, default='output.json', help="output file where to dump the metrics.")
    
    parser.add_argument('--batch_size', type=int, default=128, help="number of samples per batch during evaluation")
    parser.add_argument('--num_workers', type=int, default=4, help="number of workers processing evaluation")
    parser.add_argument('--verbose', default=False, action="store_true", help="verbose mode")
    parser.add_argument('--restore', default=False, action="store_true", help="restore temporary evaluation results")
    
    args = parser.parse_args()

    
    if not args.model_name:
        print('Unable to load model, caused by empty model name, please specify by "--model_name".')
        return
    
    config = {}
    if args.model_dir:
        config['model_dir'] = args.model_dir
    elif args.model_name not in _SUPPORTED_MODELS:
        print('Unable to load model, caused by empty model directory, please specify by "--model_dir".')
        return
    
    if args.agency:
        config['agency'] = args.agency
    if args.text_encoder:
        config['text_encoder'] = args.text_encoder
    if args.vision_encoder:
        config['vision_encoder'] = args.vision_encoder
    if args.model_script:
        config['model_script'] = args.model_script
    model_config = {args.model_name: config}
    

    def parse_multistr_args(st):
        if not st:
            return []
        return st.split(',')

    prefix = args.prefix
    datasets = parse_multistr_args(args.datasets)
    if not datasets:
        datasets = _SUPPORTED_DATASETS
    if prefix:
        datasets = [ds for ds in datasets if ds.startswith(prefix)]

    dump_metrics = evaluate(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        model_config = model_config,
        root = args.root,
        dataset_names=datasets,
        task_names=parse_multistr_args(args.tasks), 
        group_names=parse_multistr_args(args.groups),
        languages=parse_multistr_args(args.languages), 
        verbose=args.verbose,
        restore=args.restore
    )
    print(dump_metrics)
    with open(args.output, "w") as f:
        json.dump(dump_metrics, f)

if __name__ == "__main__":
    sys.exit(main())
