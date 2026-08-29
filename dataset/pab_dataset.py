import os
import random
from random import randint, shuffle
from random import random as rand
import numpy as np
from PIL import Image
from collections import defaultdict
from torch.utils.data import Dataset

from dataset.utils import pre_caption, read_json_to_list

class TextMaskingGenerator:
    def __init__(self, tokenizer, mask_prob, mask_max, skipgram_prb=0.2, skipgram_size=3, mask_whole_word=True,
                 use_roberta=False):
        self.id2token = {i: w for w, i in tokenizer.get_vocab().items()}
        self.use_roberta = use_roberta
        for i in range(len(self.id2token)):
            assert i in self.id2token.keys()  # check
        self.cls_token_id = tokenizer.cls_token_id
        self.mask_token_id = tokenizer.mask_token_id
        self.mask_max = mask_max
        self.mask_prob = mask_prob
        self.skipgram_prb = skipgram_prb
        self.skipgram_size = skipgram_size
        self.mask_whole_word = mask_whole_word

    def get_random_word(self):
        i = randint(0, len(self.id2token) - 1)
        return i

    def __call__(self, text_ids):  # tokens: [CLS] + ...
        n_pred = min(self.mask_max, max(1, int(round(len(text_ids) * self.mask_prob))))

        # candidate positions of masked tokens
        assert text_ids[0] == self.cls_token_id
        special_pos = set([0])  # will not be masked
        cand_pos = list(range(1, len(text_ids)))

        shuffle(cand_pos)
        masked_pos = set()
        max_cand_pos = max(cand_pos)
        for pos in cand_pos:
            if len(masked_pos) >= n_pred:
                break
            if pos in masked_pos:
                continue

            def _expand_whole_word(st, end):
                new_st, new_end = st, end

                if self.use_roberta:
                    while (new_st > 1) and (self.id2token[text_ids[new_st].item()][0] != 'G'):
                        new_st -= 1
                    while (new_end < len(text_ids)) and (self.id2token[text_ids[new_end].item()][0] != 'G'):
                        new_end += 1
                else:
                    # bert, WordPiece
                    while (new_st >= 0) and self.id2token[text_ids[new_st].item()].startswith('##'):
                        new_st -= 1
                    while (new_end < len(text_ids)) and self.id2token[text_ids[new_end].item()].startswith('##'):
                        new_end += 1

                return new_st, new_end

            if (self.skipgram_prb > 0) and (self.skipgram_size >= 2) and (rand() < self.skipgram_prb):
                # ngram
                cur_skipgram_size = randint(2, self.skipgram_size)
                if self.mask_whole_word:
                    st_pos, end_pos = _expand_whole_word(
                        pos, pos + cur_skipgram_size)
                else:
                    st_pos, end_pos = pos, pos + cur_skipgram_size
            else:
                if self.mask_whole_word:
                    st_pos, end_pos = _expand_whole_word(pos, pos + 1)
                else:
                    st_pos, end_pos = pos, pos + 1

            for mp in range(st_pos, end_pos):
                if (0 < mp <= max_cand_pos) and (mp not in special_pos):
                    masked_pos.add(mp)
                else:
                    break

        masked_pos = list(masked_pos)
        n_real_pred = len(masked_pos)
        if n_real_pred > n_pred:
            shuffle(masked_pos)
            masked_pos = masked_pos[:n_pred]

        for pos in masked_pos:
            if rand() < 0.8:  # 80%
                text_ids[pos] = self.mask_token_id
            elif rand() < 0.5:  # 10%
                text_ids[pos] = self.get_random_word()

        return text_ids, masked_pos


class search_train_dataset(Dataset):
    def __init__(self, config, transform):
        self.image_root = config['image_root']
        self.transform = transform
        self.max_words = config['max_words']
        
        # [NEW] ?? EDA ??,??? 0?
        # ??? 0,???????? text_eda (??????),?????
        self.eda_p = config.get('eda_p', 0) 

        # --- 1. ???? ---
        ann_file_list = config['train_file']
        self.ann = []
        for f in ann_file_list:
            self.ann += read_json_to_list(f)
        
        print(f"Total training samples (unique images): {len(self.ann)}")
        print(f"EDA Probability: {self.eda_p}")

        # --- 2. ?? image_id ?? ---
        self.img_ids = {}
        n = 0
        for item in self.ann:
            img_id = item['image_id']
            if img_id not in self.img_ids:
                self.img_ids[img_id] = n
                n += 1
        print('Total unique image_ids:', n)

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        ann = self.ann[index]
        
        # --- 1. ???? ---
        image_path = os.path.join(self.image_root, ann['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        
        # --- 2. ????? ---
        if 'captions_list' in ann and len(ann['captions_list']) > 0:
            caption_raw = random.choice(ann['captions_list'])
        elif isinstance(ann['caption'], list):
            caption_raw = random.choice(ann['caption'])
        else:
            caption_raw = ann['caption']
            
        caption_pos = pre_caption(caption_raw, self.max_words)
        
        # [NEW] ?? EDA ??
        # ?? config ? eda_p=0,??? caption_eda ??? caption_pos
        # ????????,?????? forward ??????
        #this is cancelled in 12.27 for reconstruction about eda
        #caption_eda = pre_caption(caption_raw, self.max_words, True, self.eda_p)
        
        # --- 3. ????? (TCL ??) ---
        # 4.19??,???2???????,???eda????????
        neg_act_raw = ann.get('negative_action', caption_raw)
        neg_app_raw = ann.get('negative_appearance', caption_raw)
        
        caption_neg_act = pre_caption(neg_act_raw, self.max_words)
        caption_neg_app = pre_caption(neg_app_raw, self.max_words)

        img_id = ann['image_id']

        # [NEW] ?? 6 ???:?,??,EDA?,??1,??2,ID
        return image, caption_pos, caption_neg_act, caption_neg_app, self.img_ids[img_id]
        #return image, caption_pos, caption_eda, self.img_ids[img_id]


class search_test_dataset(Dataset):
    # ???????????????
    def __init__(self, config, transform):
        ann_file = config['test_file']
        self.transform = transform
        self.image_root = config.get('image_root_test', config['image_root'])
        self.max_words = config['max_words']

        self.ann = read_json_to_list(ann_file)

        self.be_pose_img = config.get('be_pose_img', False)
        print('test dataset -->    be_pose_img:', self.be_pose_img)

        self.text = []
        self.image = []
        self.g_pids = []
        self.q_pids = []
        
        for img_id, ann in enumerate(self.ann):
            self.g_pids.append(ann['image_id'])
            self.image.append(ann['image'])
            
            captions = ann.get('captions_list', ann.get('caption'))
            if isinstance(captions, str):
                captions = [captions]
            
            for caption in captions:
                self.q_pids.append(ann['image_id'])
                self.text.append(pre_caption(caption, self.max_words))

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_root, self.ann[index]['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        if self.be_pose_img:
            pose_path = os.path.join(self.image_root, 'pose/' + self.ann[index]['image'])
            try:
                pose = Image.open(pose_path).convert('RGB')
                pose = self.transform(pose)
            except Exception:
                pose = torch.zeros_like(image)
        else:
            pose = 0

        return image, pose, index