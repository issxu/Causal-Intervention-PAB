import os
os.environ["CUDA_VISIBLE_DEVICES"]='2'
import json
import argparse
import torch
import torch.nn.functional as F
from transformers import BertTokenizer
from models.model_search_xvlm2 import Search
# from models.model_search_xvlm2 import Search as Search2
from pathlib import Path
from dataset import create_loader, create_dataset
from ruamel.yaml import YAML
import time
import datetime
from prettytable import PrettyTable
from tqdm import tqdm
import torch
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np
import utils

yaml = YAML(typ='safe')

@torch.no_grad()
def evaluation_itm(models, device, config, args, sims_matrix, image_embeds, text_embeds, text_atts):
    for model in models:
        model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'
    print('Computing matching score')
    start_time = time.time()

    num_tasks = utils.get_world_size()
    rank = utils.get_rank()
    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)
    print(sims_matrix.shape)
    score_matrix_t2i = torch.full(sims_matrix.size(), 1000.0).to(device)
    
    for i, sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 500, header)):
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)
        encoder_output = image_embeds[topk_idx]
        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)
        
        # 使用多个模型进行推理
        scores = []
        for model in models:
            output = model.get_cross_embeds(encoder_output, encoder_att,
                                            text_embeds=text_embeds[start + i].repeat(config['k_test'], 1, 1),
                                            text_atts=text_atts[start + i].repeat(config['k_test'], 1),)[:, 0, :]
            score = model.itm_head(output)[:, 1]
            scores.append(score)
        
        # 对多个模型的输出进行平均
        score = torch.mean(torch.stack(scores), dim=0)
        score_matrix_t2i[start + i, topk_idx] = score

    min_values, _ = torch.min(score_matrix_t2i, dim=1)
    replacement_tensor = min_values.view(-1, 1).expand(-1, score_matrix_t2i.size(1))
    for i in range(sims_matrix.size(0)):
        score_matrix_t2i[i][score_matrix_t2i[i] == 1000.0] = replacement_tensor[i][score_matrix_t2i[i] == 1000.0]
    score_matrix_t2i[score_matrix_t2i == 1000.0] = replacement_tensor[score_matrix_t2i == 1000.0]
    score_matrix_t2i = (score_matrix_t2i - score_matrix_t2i.min()) / (score_matrix_t2i.max() - score_matrix_t2i.min())

    score_sim_t2i = sims_matrix.clone()
    score_sim_t2i = (score_sim_t2i - score_sim_t2i.min()) / (score_sim_t2i.max() - score_sim_t2i.min())
    score_matrix_t2i = score_matrix_t2i + 0.002 * score_sim_t2i  #

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Computing matching score time {}'.format(total_time_str))
    return score_matrix_t2i.cpu().numpy()

@torch.no_grad()
def infer_topk_images_with_itm(models, tokenizer, queries, image_loader, device, config, topk=10):
    """
    使用 ITM 分数进行文本与图像匹配，返回 top-k 图像文件名（无后缀名）
    批量处理查询
    """
    for model in models:
        model.eval()
    
    # Step 1: 提取文本特征（批量处理）
    text_inputs = tokenizer([query['caption'] for query in queries], padding='max_length', truncation=True,
                            max_length=config['max_tokens'], return_tensors="pt").to(device)
    text_embed = models[0].get_text_embeds(text_inputs.input_ids, text_inputs.attention_mask)
    text_feat = models[0].get_text_feat(text_embed)
    text_feat = F.normalize(text_feat, dim=-1)

    # Step 2: 提取图像特征（只计算一次）
    image_feats, image_embeds, image_paths = [], [], []
    for images, img_ids in image_loader:
        images = images.to(device)
        image_embed, _ = models[0].get_vision_embeds(images)        
        image_feat = models[0].get_image_feat(image_embed)
        image_feat = F.normalize(image_feat, dim=-1)

        image_embeds.append(image_embed)
        image_feats.append(image_feat)
        image_paths.extend(img_ids)

    image_feats = torch.cat(image_feats, dim=0)
    image_embeds = torch.cat(image_embeds, dim=0)

    # Step 3: 计算相似度矩阵（批量处理）
    sims_matrix = torch.matmul(image_feats, text_feat.t())  # [num_images, num_queries]
    sims_matrix = sims_matrix.t()
    
    # Step 4: 使用 ITM 计算最终匹配分数
    score_matrix = evaluation_itm(
        models, device, config, args=None,
        sims_matrix=sims_matrix, image_embeds=image_embeds,
        text_embeds=text_embed, text_atts=text_inputs.attention_mask
    )

    # Step 5: 获取每个查询的 top-k 图像索引和文件名
    topk_image_files = []
    score_matrix = torch.tensor(score_matrix) if isinstance(score_matrix, np.ndarray) else score_matrix
    for i in range(len(queries)):
        topk_indices = torch.topk(score_matrix[i], topk).indices
        topk_image_files.append([Path(image_paths[idx]).stem for idx in topk_indices])
    
    return topk_image_files

def main(args, config):
    # 初始化设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 加载分词器和多个模型
    tokenizer = BertTokenizer.from_pretrained(config['text_encoder'])
    checkpoints = ['output/my_ori_xlvm2/checkpoint_best4.pth',
    'output/my_ori_xlvm2_2/checkpoint_best4.pth',
    ]
    models = []
    for checkpoint in checkpoints:  # 假设 args.checkpoints 是一个包含多个模型路径的列表
        model = Search(config=config)
        model.load_pretrained(checkpoint)
        model = model.to(device)
        models.append(model)

    # 加载图像数据集
    print("### Loading image dataset...")
    _, image_dataset = create_dataset(config, evaluate=True)  # 使用测试集或推理集
    image_loader = create_loader([image_dataset], [None], 
                                 batch_size=[config['batch_size_test']], 
                                 num_workers=[4], 
                                 is_trains=[False], 
                                 collate_fns=[None])[0]
    
    # 读取查询 JSON 文件
    print("### Loading queries...")
    with open(args.query_file, 'r') as f:
        queries = [json.loads(line) for line in f]

    # 批量处理查询
    print("### Processing queries...")
    output_lines = []
    topk_image_files = infer_topk_images_with_itm(models, tokenizer, queries, image_loader, device, config, topk=10)
    
    for query, topk_images in zip(queries, topk_image_files):
        query_index = query['query_index']
        print(f"Query {query_index}: {topk_images}")
        output_lines.append(" ".join(topk_images))

    output_path = os.path.join(args.output_dir, 'answer.txt')
    with open(output_path, 'w') as f:
        f.write("\n".join(output_lines))
    print(f"Results saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="Path to config YAML file")
    # parser.add_argument('--checkpoints', nargs='+', required=True, help="Paths to model checkpoints")
    parser.add_argument('--query_file', type=str, required=True, help="Path to input query JSON file")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save results")
    parser.add_argument('--device', default='cuda', help="Device to use for inference (cuda or cpu)")

    args = parser.parse_args()

    # 创建输出目录
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # 读取配置文件
    config = yaml.load(open(args.config, 'r'))
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    # 执行主函数
    main(args, config)