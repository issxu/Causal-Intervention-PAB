# /home/weifeng/pab_x2vlm/dataset/__init__.py (???)

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image # ???? Image

from dataset.randaugment import RandomAugment
from dataset.random_erasing import RandomErasing

# ? PAB ????????,??? 'as' ???
from dataset.pab_dataset import search_train_dataset as pab_train_dataset
from dataset.pab_dataset import search_test_dataset as pab_test_dataset

# ? ICFG ????????
from dataset.icfg_dataset import icfg_train_dataset, icfg_test_dataset

# ? RSTPReid ????????
from dataset.rstp_dataset import rstp_train_dataset, rstp_test_dataset

from dataset.cuhk_dataset import search_train_dataset as cuhk_train_dataset
from dataset.cuhk_dataset import search_test_dataset as cuhk_test_dataset

from dataset.baseline_dataset import search_train_dataset as baseline_train_dataset
from dataset.baseline_dataset import search_test_dataset as baseline_test_dataset

def create_dataset(dataset, config, evaluate=False):
    """
    ???????????? ("???")
    """

    # --- ???:?????????? ---
    # CUHK-PEDES/ICFG-PEDES ????????
    reid_norm = transforms.Normalize((0.38901278, 0.3651612, 0.34836376), (0.24344306, 0.23738699, 0.23368555))
    
    # ??X2VLM/CLIP????????
    clip_norm = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    # ?????????????????
    normalize = reid_norm if dataset in ['cuhk', 'icfg', 'rstp'] else clip_norm
    print(f"Using {'ReID' if normalize == reid_norm else 'CLIP'} normalization for dataset '{dataset}'.")
    
    # ????????????
    train_transform = transforms.Compose([
        transforms.Resize((config['h'], config['w']), interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(),
        RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
        transforms.ToTensor(),
        normalize,
        RandomErasing(probability=config.get('erasing_p', 0.5), mean=[0.0, 0.0, 0.0])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((config['h'], config['w']), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ])

    # --- ???:?? 'dataset' ??,??????? ---
    
    if dataset == 'pab':
        # ??? ??????????PAB???? ???
        test_dataset = pab_test_dataset(config, test_transform)
        if evaluate:
            # ?????,Search_xvlm2.py ???? (None, test_dataset)
            return None, test_dataset
    
        train_dataset = pab_train_dataset(config, train_transform)
        return train_dataset, test_dataset
    
    elif dataset == 'icfg': # ??? ICFG ????? ???
        # ICFG ?????????? icfg_dataset.py ???
        test_dataset = icfg_test_dataset(config, test_transform)
        if evaluate:
            return None, test_dataset

        train_dataset = icfg_train_dataset(config, train_transform)
        return train_dataset, test_dataset
    
    elif dataset == 'rstp':
        test_dataset = rstp_test_dataset(config, test_transform)
        if evaluate:
            return None, test_dataset # ??????? test_dataset

        train_dataset = rstp_train_dataset(config, train_transform)
        return train_dataset, test_dataset
        
    elif dataset == 'cuhk':
        test_dataset = cuhk_test_dataset(config, test_transform)
        if evaluate:
            return None, test_dataset
        train_dataset = cuhk_train_dataset(config, train_transform)
        return train_dataset, test_dataset
        
    elif dataset == 'baseline':
        test_dataset = baseline_test_dataset(config, test_transform)
        if evaluate:
            return None, test_dataset
        train_dataset = baseline_train_dataset(config, train_transform)
        return train_dataset, test_dataset
    
    
    else:
        raise ValueError(f"Dataset '{dataset}' is not supported.")


# --- create_sampler ? create_loader ?????? ---
# (??????????????????)

def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset, shuffle in zip(datasets, shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks,
                                                      rank=global_rank, shuffle=shuffle)
        samplers.append(sampler)
    return samplers


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset, sampler, bs, n_worker, is_train, collate_fn in zip(datasets, samplers, batch_size, num_workers,
                                                                    is_trains, collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )
        loaders.append(loader)

    if len(loaders) <= 1:
        print(f"### be careful: func create_loader returns a list length of {len(loaders)}")

    return loaders