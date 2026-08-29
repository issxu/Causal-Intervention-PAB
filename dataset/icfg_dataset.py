import os
import json
import random
from PIL import Image
from torch.utils.data import Dataset
from dataset.utils import pre_caption

class icfg_train_dataset(Dataset):
    def __init__(self, config, transform):
        self.image_root = config['image_root']
        self.transform = transform
        self.max_words = config['max_words']
        self.eda_p = config.get('eda_p', 0) # ???????eda_p,????0

        # 1. ??????split?JSON??
        with open(config['train_file'], 'r') as f:
            all_anns = json.load(f)
        
        # 2. ??? split == 'train' ???
        self.ann = [item for item in all_anns if item['split'] == 'train']
        
        print(f"ICFG train dataset: Found {len(self.ann)} training entries.")

        # 3. ?? image_id ??
        self.img_ids = {}
        n = 0
        for item in self.ann:
            pid = item['id']
            if pid not in self.img_ids:
                self.img_ids[pid] = n
                n += 1

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        ann = self.ann[index]
        
        image_path = os.path.join(self.image_root, ann['file_path'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        
        pid = ann['id']

        # ?captions?????????
        cap = random.choice(ann['captions'])
        caption = pre_caption(cap, self.max_words)
        caption_eda = pre_caption(cap, self.max_words, True, self.eda_p)
        
        # ??? train_xvlm2.py ???4????
        return image, caption, caption_eda, self.img_ids[pid]


class icfg_test_dataset(Dataset):
    def __init__(self, config, transform):
        self.transform = transform
        self.image_root = config['image_root']
        self.max_words = config['max_words']
        
        # 1. ??????split?JSON??
        with open(config['test_file'], 'r') as f:
            all_anns = json.load(f)

        # 2. ??? split == 'test' ???
        self.ann = [item for item in all_anns if item['split'] == 'test']
        print(f"ICFG test dataset: Found {len(self.ann)} testing entries.")
        
        # 3. ??????? text, image, g_pids, q_pids
        self.text = []
        self.image = []
        self.g_pids = []
        self.q_pids = []
        
        # ?????????gallery?????
        gallery_images = {}
        
        for item in self.ann:
            pid = item['id']
            image_path = item['file_path']
            
            # ??????????gallery,????
            if image_path not in gallery_images:
                gallery_images[image_path] = pid
            
            # ???caption???query
            for caption in item['captions']:
                self.text.append(pre_caption(caption, self.max_words))
                self.q_pids.append(pid)
        
        # ????????gallery??
        for img_path, pid in gallery_images.items():
            self.image.append(img_path)
            self.g_pids.append(pid)

    def __len__(self):
        # ???gallery?????
        return len(self.image)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_root, self.image[index])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        
        # ??? eval.py ???3???? (pose??????)
        return image, {}, index