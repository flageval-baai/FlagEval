from torchvision.datasets.utils import download_url, download_and_extract_archive, extract_archive
from .constants import _DATASET_ROOT_DIR, _HUGGINGFACE_AUTH_TOKEN
from datasets import DownloadManager,DownloadConfig
import os, shutil
import json 
from subprocess import call
from .voc2007 import PASCALVoc2007, PASCALVoc2007Cropped

_URLS_ = {
    'imagenet1k': "https://huggingface.co/datasets/imagenet-1k/resolve/1500f8c59b214ce459c0a593fa1c87993aeb7700/data/val_images.tar.gz",
    'imagenet-a': "https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar",
    'imagenet-r': "https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar",
    'imagenet-sketch': "https://drive.google.com/file/d/1Mj0i5HBthqH1p_yeXzsg22gZduvgoNeA",
    "imagenetv2": "https://imagenetv2public.s3-us-west-2.amazonaws.com/imagenetv2-matched-frequency.tar.gz",
    "winoground": [
        "https://huggingface.co/datasets/facebook/winoground/resolve/main/data/examples.jsonl",
        "https://huggingface.co/datasets/facebook/winoground/resolve/main/data/images.zip"
    ],
    "mscoco_captions": [
        "http://images.cocodataset.org/zips/train2014.zip",
        "http://images.cocodataset.org/zips/val2014.zip",
        "https://github.com/mehdidc/retrieval_annotations/releases/download/1.0.0/coco_test_karpathy.json"
    ],
    "flickr_30k": "https://github.com/mehdidc/retrieval_annotations/releases/download/1.0.0/flickr30k_test_karpathy.txt",
    "pcam": [
        ("https://drive.google.com/file/d/1qV65ZqZvWzuIVthK8eVDhIwrbnsJdbg_", "camelyonpatch_level_2_split_test_x.h5", "d8c2d60d490dbd479f8199bdfa0cf6ec"),
        ("https://drive.google.com/file/d/17BHrSrwWKjYsOgTMmoqrIjDy6Fa2o_gP", "camelyonpatch_level_2_split_test_y.h5", "60a7035772fbdb7f34eb86d4420cf66a")
    ],
    "objectnet": "https://objectnet.dev/downloads/objectnet-1.0.zip",
    "resisc45": "https://onedrive.live.com/download?resid=5C5E061130630A68!107&authkey=!AHHNaHIlzp_IXjs",
}
STATUS_FILE = 'status.json'

def load_status(_dataset_root_dir=_DATASET_ROOT_DIR):
    status_file = os.path.join(_dataset_root_dir, STATUS_FILE)
    if os.path.exists(status_file):
        status = json.loads(open(status_file).read())
        return status
    else:
        f = open(status_file, 'w+')
        status = {}
        f.write(json.dumps(status, ensure_ascii=False))
        f.close()
        return status

def reset_status(_dataset_root_dir=_DATASET_ROOT_DIR):
    if not os.path.exists(_dataset_root_dir):
        os.makedirs(_dataset_root_dir)
    status_file = os.path.join(_dataset_root_dir, STATUS_FILE)
    f = open(status_file, 'w+')
    f.write(json.dumps({}, ensure_ascii=False))
    f.close()

def save_to_status(dataset_names, _dataset_root_dir=_DATASET_ROOT_DIR):
    status = load_status(_dataset_root_dir=_dataset_root_dir)
    status_file = os.path.join(_dataset_root_dir, STATUS_FILE)
    update_status = {dataset:True for dataset in dataset_names}
    status.update(update_status)
    f = open(status_file, 'w+')
    f.write(json.dumps(status, ensure_ascii=False))
    f.close()
    return status

def encrypt(fpath: str, algorithm: str) -> str:
    import hashlib
    import rich.progress
    with rich.progress.open(fpath, 'rb') as f:
        hash = hashlib.new(algorithm)
        for chunk in iter(lambda: f.read(2**20), b''):
            hash.update(chunk)
        return hash.hexdigest()

def download_from_kaggle(kaggle_dataset):
    def has_kaggle():
        return call("which kaggle", shell=True) == 0
    if not has_kaggle():
        print("Kaggle is needed to download the dataset. Please install it via `pip install kaggle`")
    call(f"kaggle datasets download -d {kaggle_dataset}", shell=True)


def download_from_huggingface(dataset_name, root_dir=None):
    dc = DownloadConfig()
    dc.use_auth_token = _HUGGINGFACE_AUTH_TOKEN
    if root_dir:
        dc.cache_dir = root_dir
    dl = DownloadManager(download_config=dc)
    url = _URLS_.get(dataset_name, None)

    def get_filename_from_url(url):
        file_name = None
        if '/' in url:
            idx = url.rindex('/')
            file_name = url[idx+1:]
        return file_name
    
    if url:
        if isinstance(url, list):
            for _url in url:
                extracted_dir = dl.download_and_extract(_url)
                file_name = get_filename_from_url(_url)
                if file_name.endswith('.zip'):
                    source_dir = os.path.join(extracted_dir, file_name[:file_name.index('.zip')])
                    move_to_target_dir_and_rename(source_dir, root_dir)
                    extracted_dir = os.path.join(root_dir, source_dir[source_dir.rindex('/')+1:])
                else:
                    move_to_target_dir_and_rename(extracted_dir, root_dir, file_name)
                    extracted_dir = os.path.join(root_dir, file_name)
        else:
            extracted_dir = dl.download_and_extract(url)
    else:
        raise RuntimeError('No URLs is found.')
    return extracted_dir

def move_to_target_dir_and_rename(file_name, target_dir, new_file_name=None):
    if not os.path.exists(target_dir):
            os.makedirs(target_dir)
    if os.path.isfile(file_name):
        shutil.move(file_name, os.path.join(target_dir, new_file_name))
    else:
        shutil.move(file_name, target_dir)
        

def parse_ann_file(ann_file):
    image_list = []
    with open(ann_file) as fd:
        for line in fd:
            line = line.strip()
            if '.jpg' not in line:
                continue
            try:
                img, _ = line.split(".jpg,")
                image_list.append(f"{img}.jpg")
            except Exception as e:
                    print(line)
    return image_list

def download_and_prepare_data(dataset_name, root_dir=None, restore=True, _dataset_root_dir=_DATASET_ROOT_DIR):
    is_success = False
    _prepared_datasets = None
    if restore:
        status = load_status(_dataset_root_dir=_dataset_root_dir)
    else:
        status = {}
        reset_status(_dataset_root_dir=_dataset_root_dir)
    if status.get(dataset_name, False):
        is_success = True 
        print(f'Success to prepare dataset folder {dataset_name}.')
        return
    if dataset_name.startswith("imagenet") and '_' in dataset_name:
        dataset_name = dataset_name[:dataset_name.index('_')]
    if not root_dir:
        root_dir = os.path.join(_dataset_root_dir, dataset_name)
    if dataset_name == 'imagenet1k':
        extracted_dir = download_from_huggingface(dataset_name=dataset_name, root_dir=root_dir)
        for fname in os.listdir(extracted_dir):
            ridx = fname.rindex('_')
            dir_name = fname[ridx+1:].replace('.JPEG', '')
            new_fname = f'{fname[:ridx]}.JPEG'
            dir_name = os.path.join(root_dir, 'val', dir_name)
            move_to_target_dir_and_rename(os.path.join(extracted_dir, fname), dir_name, new_fname)
        _prepared_datasets = ["imagenet1k", "imagenet1k_cn", "imagenet1k_jp",  "imagenet1k_it"]
        is_success = True
    elif dataset_name in ['imagenet-a', 'imagenet-r']:
        url = _URLS_.get(dataset_name, None)
        download_and_extract_archive(url=url, download_root = _dataset_root_dir, extract_root = _dataset_root_dir)
        if dataset_name == 'imagenet-a':
            _prepared_datasets = ['imagenet-a', 'imagenet-a_cn', "imagenet-a_jp",  "imagenet-a_it"]
        else:
            _prepared_datasets = ['imagenet-r', 'imagenet-r_cn', "imagenet-r_jp",  "imagenet-r_it"]
        is_success = True
    elif dataset_name=='imagenet-sketch':
        url = _URLS_.get(dataset_name, None)
        download_and_extract_archive(url=url, download_root = _dataset_root_dir, extract_root = _dataset_root_dir, filename='ImageNet-Sketch.zip')
        shutil.move(os.path.join(_dataset_root_dir, 'sketch'), os.path.join(_dataset_root_dir, dataset_name))
        _prepared_datasets = ['imagenet-sketch', 'imagenet-sketch_cn', "imagenet-sketch_jp",  "imagenet-sketch_it"]
        is_success = True
    elif dataset_name == 'imagenetv2':
        url = _URLS_.get(dataset_name, None)
        download_url(url=url, root=root_dir)
        _prepared_datasets = ['imagenetv2', 'imagenetv2_cn', "imagenetv2_jp",  "imagenetv2_it"]
        is_success = True
    elif dataset_name == "caltech101":
        from torchvision.datasets import Caltech101
        ds = Caltech101(root=root_dir, download=True)
        is_success = True
    elif dataset_name == "cars":
        from torchvision.datasets import StanfordCars
        ds = StanfordCars(root=root_dir, split="test",download=True)
        is_success = True
    elif dataset_name == "cifar10":
        from torchvision.datasets import CIFAR10
        ds = CIFAR10(root=root_dir, download=True)
        is_success = True
    elif dataset_name == "cifar100":
        from torchvision.datasets import CIFAR100
        ds = CIFAR100(root=root_dir, download=True)
        is_success = True
    elif dataset_name == "country211":
        from torchvision.datasets import Country211
        ds = Country211(root=root_dir, split="test", download=True)
        is_success = True
    elif dataset_name == "dtd":
        from torchvision.datasets import DTD
        ds = DTD(root=root_dir, split="test", download=True)
        is_success = True
    elif dataset_name == "eurosat":
        from torchvision.datasets import EuroSAT
        ds = EuroSAT(root=root_dir, download=True)
        is_success = True
    elif dataset_name == "fgvc-aircraft":
        from torchvision.datasets import FGVCAircraft
        ds = FGVCAircraft(root=root_dir, annotation_level="variant", split='test', download=True)
        is_success = True
    elif dataset_name == "flowers":
        from torchvision.datasets import Flowers102
        ds = Flowers102(root=root_dir, split='test', download=True)
        is_success = True
    elif dataset_name == "food101":
        from torchvision.datasets import Food101
        ds = Food101(root=root_dir, split='test', download=True)
        is_success = True
    elif dataset_name == "gtsrb":
        from torchvision.datasets import GTSRB
        ds = GTSRB(root=root_dir, split='test', download=True)
        is_success = True
    elif dataset_name == "mnist":
        from torchvision.datasets import MNIST
        ds = MNIST(root=root_dir, train=False, download=True)
        is_success = True
    elif dataset_name == "objectnet":
        url = _URLS_.get(dataset_name, None)
        if url:
            download_and_extract_archive(url=url, download_root=root_dir, extract_root=root_dir)
    elif dataset_name == "pcam":
        urls = _URLS_.get(dataset_name, None)
        for _url, _file, _md5 in urls:
            download_url(url=_url, filename=_file, md5=_md5, root=os.path.join(root_dir, "pcam"))
        is_success = True
    elif dataset_name == "pets":
        from torchvision.datasets import OxfordIIITPet
        ds = OxfordIIITPet(root=root_dir, split='test', target_types="category", download=True)
        is_success = True
    elif dataset_name == "renderedsst2":
        from torchvision.datasets import RenderedSST2
        ds = RenderedSST2(root=root_dir, split='test', download=True)
        is_success = True
    elif dataset_name == "resisc45":
        url = _URLS_.get(dataset_name, None)
        if url:
            download_url(url=url, root=root_dir, filename='NWPU-RESISC45.rar')
            command = f'cd {root_dir};unrar x NWPU-RESISC45.rar'
            print(command)
            call(command, shell=True)
            is_success = True
    elif dataset_name == "stl10":
        from torchvision.datasets import STL10
        ds = STL10(root=root_dir, split='test', download=True)
        is_success = True
    elif dataset_name == "sun397":
        from torchvision.datasets import SUN397
        ds = SUN397(root=root_dir, download=True)
        is_success = True
    elif dataset_name == "winoground":
        download_from_huggingface(dataset_name=dataset_name, root_dir=root_dir)
        is_success = True
    elif dataset_name == "voc2007":
        ds = PASCALVoc2007Cropped(root=root_dir, set='test', download=True)
        is_success = True
    elif dataset_name == "voc2007_multilabel":
        ds = PASCALVoc2007(root=root_dir, set='test', download=True)
        is_success = True
    elif dataset_name == "fer2013":
        file_name = "fer2013.zip"
        file_path = os.path.join(root_dir, file_name)
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)
        prepared_archieve = False
        if os.path.exists(file_path):
            md5_value = encrypt(file_path, 'md5')
            if md5_value == "9f58794746ff496be12cf0bb2679e3d4":
                prepared_archieve = True
        if not prepared_archieve and os.path.exists(file_name):
            md5_value = encrypt(file_name, 'md5')
            if md5_value == "9f58794746ff496be12cf0bb2679e3d4":
                if not os.path.exists(file_path):
                    os.makedirs(file_path)
                shutil.move(file_name, file_path)
                prepared_archieve = True
        if not prepared_archieve:
            download_from_kaggle("msambare/fer2013")
            shutil.move(file_name, file_path)
            prepared_archieve = True
        if prepared_archieve and not os.path.exists(os.path.join(root_dir, 'test')):
            command = f'unzip -d {root_dir} {file_path}'
            call(command, shell=True)
        is_success = True
    elif dataset_name == "mscoco_captions":
        urls = _URLS_.get(dataset_name, None)
        for url in urls:
            if url.endswith('.json'):
                download_url(url=url, root=root_dir)
            else:
                download_and_extract_archive(url=url, download_root=root_dir, extract_root=root_dir)
        is_success = True
    elif dataset_name.startswith("mscoco_captions_cn") or dataset_name.startswith("xtd"):
        dataset_name = "mscoco_captions_cn" if dataset_name.startswith("mscoco_captions_cn") else "xtd"
        root_dir = os.path.join(_dataset_root_dir, dataset_name)
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)
        img_dir = os.path.join(root_dir, 'Images')
        ann_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mscoco')
        if dataset_name == "mscoco_captions_cn":
            shutil.copy(os.path.join(ann_dir, 'coco_captions_cn_test_1k.txt'), root_dir)
            shutil.copy(os.path.join(ann_dir, 'coco_captions_cn_test_5k.txt'), root_dir)
            ann_file = os.path.join(root_dir, 'coco_captions_cn_test_1k.txt')
            _prepared_datasets = ["mscoco_captions_cn_1k", "mscoco_captions_cn_5k"]
        else:
            for lang in ['de', 'en', 'es', 'fr', 'it', 'jp', 'ko', 'pl', 'ru', 'tr', 'zh']:
                shutil.copy(os.path.join(ann_dir, f'xtd10_{lang}_pair.txt'), root_dir)
            ann_file = os.path.join(root_dir, 'xtd10_en_pair.txt')
            _prepared_datasets = ["xtd_en", "xtd_de", "xtd_es", "xtd_fr", "xtd_it", "xtd_jp", "xtd_ko", "xtd_pl", "xtd_ru", "xtd_tr", "xtd_zh"]
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        image_list = parse_ann_file(ann_file)

        is_download_full_set = status.get('mscoco_captions', False)
        for img in image_list:
            if os.path.exists(os.path.join(img_dir, img)):
                continue
            else:
                if not is_download_full_set:
                    download_and_prepare_data("mscoco_captions", _dataset_root_dir=_dataset_root_dir)
                    is_download_full_set = True
                source_dir = 'val2014' if 'val2014' in img else 'train2014'
                source_file = os.path.join(_dataset_root_dir, 'mscoco_captions', source_dir, img)
                shutil.copy(source_file, img_dir)
        is_success = True
    elif dataset_name == "flickr30k":
        file_name = "flickr30k.zip"
        file_path = os.path.join(root_dir, file_name)
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)
        prepared_archieve = False
        if os.path.exists(file_path):
            md5_value = encrypt(file_path, 'md5')
            if md5_value == "15b5f975f6c0c144fa27591bb90ffb91":
                prepared_archieve = True
        if not prepared_archieve and os.path.exists(file_name):
            md5_value = encrypt(file_name, 'md5')
            if md5_value == "15b5f975f6c0c144fa27591bb90ffb91":
                shutil.move(file_name, file_path)
                prepared_archieve = True
        if not prepared_archieve:
            download_from_kaggle("adityajn105/flickr30k")
            shutil.move(file_name, file_path)
            prepared_archieve = True
        if prepared_archieve and not os.path.exists(os.path.join(root_dir, 'Images')):
            command = f'unzip -d {root_dir} {file_path}'
            call(command, shell=True)
        if not os.path.exists(os.path.join(root_dir, 'flickr30k_test_karpathy.txt')):
            url = _URLS_.get(dataset_name, None)
            if url:
                download_url(url=url, filename='flickr30k_test_karpathy.txt', root=root_dir)
        is_success = True
    elif dataset_name == "flickr30k_cn" or dataset_name.startswith('multi30k'):
        datasets = 'flickr30k-cn' if dataset_name == "flickr30k_cn" else "multi30k"
        root_dir = os.path.join(_dataset_root_dir, datasets)
        status = load_status(_dataset_root_dir=_dataset_root_dir)
        if not status.get('flickr30k', False):
            download_and_prepare_data('flickr30k', _dataset_root_dir=_dataset_root_dir)
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)
        img_dir = os.path.join(root_dir, "Images")
        ann_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'flickr')
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        if datasets == 'flickr30k-cn':
            ann_file = 'flickr30k_test_CNA.txt'
            shutil.copy(os.path.join(ann_dir, ann_file), root_dir)
            _prepared_datasets = ["flickr30k_cn"]
        else:
            ann_file = 'multi30k-cs_test.txt'
            for lang in ['cs', 'de', 'en','fr']:
                shutil.copy(os.path.join(ann_dir, f'multi30k-{lang}_test.txt'), root_dir)
            _prepared_datasets = ["multi30k_en", "multi30k_fr", "multi30k_de", "multi30k_cs"]
        image_list = parse_ann_file(os.path.join(root_dir, ann_file))
        
        for img in image_list:
            if os.path.exists(os.path.join(img_dir, img)):
                continue
            else:
                source_file = os.path.join(_dataset_root_dir, 'flickr30k', 'Images', img)
                shutil.copy(source_file, img_dir)
        is_success = True

    if is_success:
        if not isinstance(_prepared_datasets, list):
            _prepared_datasets = [dataset_name]
        save_to_status(_prepared_datasets, _dataset_root_dir=_dataset_root_dir)
        print(f'Success to prepare dataset folder {dataset_name}.')
    
    