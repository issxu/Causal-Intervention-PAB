# ???: /home/weifeng/pab_x2vlm/dataset/rstp_dataset.py

import os
import json
import random
from PIL import Image
from torch.utils.data import Dataset
from .utils import pre_caption # ??????

# ???????RSTPReid????????

class rstp_train_dataset(Dataset):
    def __init__(self, config, transform):
        self.image_root = config['image_root']
        self.transform = transform
        self.max_words = config['max_words']
        self.eda_p = config.get('eda_p', 0) # ?? .get() ?? KeyError

        # ?????????
        with open(config['annotation_file'], 'r') as f:
            all_data = json.load(f)

        # ?? "split" ?????????
        self.ann = [item for item in all_data if item['split'] == 'train']
        
        print(f"RSTPReid :: Found {len(self.ann)} training annotations.")

        # ?? image_id ? 0-N ????? (??????????)
        self.img_ids = {}
        n = 0
        for item in self.ann:
            pid = item['id']
            if pid not in self.img_ids:
                self.img_ids[pid] = n
                n += 1
        print(f'RSTPReid :: Found {n} unique person IDs in training set.')

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        ann = self.ann[index]
        
        image_path = os.path.join(self.image_root, ann['img_path'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        person_id = ann['id']

        # ???caption???????
        cap = random.choice(ann['captions'])
        caption = pre_caption(cap, self.max_words)
        caption_eda = pre_caption(cap, self.max_words, True, self.eda_p)

        # ????????? train_dataset ??4??
        return image, caption, caption_eda, self.img_ids[person_id]


class rstp_test_dataset(Dataset):
    def __init__(self, config, transform):
        self.transform = transform
        self.image_root = config.get('image_root_test', config['image_root'])
        self.max_words = config['max_words']

        # ?????????
        with open(config['annotation_file'], 'r') as f:
            all_data = json.load(f)

        # ???????
        test_data = [item for item in all_data if item['split'] == 'test']
        print(f"RSTPReid :: Found {len(test_data)} test annotations.")

        # ???????????
        self.text = []
        self.image = []
        self.g_pids = []
        self.q_pids = []
        
        # ?? Gallery (???????????)
        unique_images = {}
        for item in test_data:
            if item['img_path'] not in unique_images:
                unique_images[item['img_path']] = item['id']
        
        self.image = sorted(unique_images.keys()) # Gallery Images
        self.g_pids = [unique_images[path] for path in self.image] # Gallery PIDs
        
        # ?? Queries (?????? captions)
        for item in test_data:
            pid = item['id']
            for caption in item['captions']:
                self.text.append(pre_caption(caption, self.max_words))
                self.q_pids.append(pid)

        print(f"RSTPReid :: Gallery size: {len(self.image)} images.")
        print(f"RSTPReid :: Query size: {len(self.text)} captions.")

    def __len__(self):
        # __len__ ???? Gallery ???,?? DataLoader ??
        return len(self.image)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_root, self.image[index])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        # ????? test_dataset ??3??
        # ?? image, pose (????), index
        return image, {}, index