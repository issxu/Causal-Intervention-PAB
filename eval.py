import os
import json
import numpy as np
import time
import datetime
from prettytable import PrettyTable

import torch
import torch.distributed as dist
import torch.nn.functional as F

import utils


@torch.no_grad()
def evaluation_itc(model, data_loader, tokenizer, device, config):
    model.eval()
    print('Computing features for evaluation (Original Logic)')
    start_time = time.time()

    # === ???????? ===
    texts = data_loader.dataset.text
    num_text = len(texts)
    text_bs = config['batch_size_test_text']
    text_embeds = []
    text_atts = []
    text_feats = []
    
    # ???????
    for i in range(0, num_text, text_bs):
        text = texts[i: min(num_text, i + text_bs)]
        text_input = tokenizer(text, padding='max_length', truncation=True, max_length=config['max_tokens'],
                               return_tensors="pt").to(device)
        text_embed = model.get_text_embeds(text_input.input_ids, text_input.attention_mask)
        text_feat = model.get_text_feat(text_embed)
        text_feat = F.normalize(text_feat, dim=-1)

        text_embeds.append(text_embed)
        text_atts.append(text_input.attention_mask)
        text_feats.append(text_feat)

    text_embeds = torch.cat(text_embeds, dim=0)
    text_atts = torch.cat(text_atts, dim=0)
    text_feats = torch.cat(text_feats, dim=0)

    image_embeds = []
    image_feats = []
    for image, pose, img_id in data_loader:
        image = image.to(device)
        image_embed, _ = model.get_vision_embeds(image)

        if config.get('be_pose_img', False):
            if isinstance(pose, torch.Tensor):
                pose = pose.to(device)
                if model.be_pose_conv:
                    pose = model.pose_conv(pose)
                pose_embed, _ = model.get_vision_embeds(pose)
                image_embed = model.pose_block(image_embed, pose_embed)

        image_feat = model.get_image_feat(image_embed)
        image_feat = F.normalize(image_feat, dim=-1)
        image_embeds.append(image_embed)
        image_feats.append(image_feat)

    image_embeds = torch.cat(image_embeds, dim=0)
    image_feats = torch.cat(image_feats, dim=0)

    sims_matrix = image_feats @ text_feats.t()
    sims_matrix_t2i = sims_matrix.t()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Computing features time {}'.format(total_time_str))

    return sims_matrix_t2i, image_embeds, text_embeds, text_atts


@torch.no_grad()
def evaluation_itm(model, device, config, args, sims_matrix, image_embeds, text_embeds, text_atts):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation ITM:'
    print('Computing matching score (Restoring 89.33 Logic)')
    start_time = time.time()

    num_tasks = utils.get_world_size()
    rank = utils.get_rank()
    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    # === ???? 1: ?? 1000.0 ??? ===
    score_matrix_t2i = torch.full(sims_matrix.size(), 1000.0).to(device)

    for i, sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 500, header)):
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)
        encoder_output = image_embeds[topk_idx]
        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)
        
        # ??? ITM ??,??? caption ??,?? repeat
        output = model.get_cross_embeds(
            encoder_output, 
            encoder_att,
            text_embeds=text_embeds[start + i].repeat(config['k_test'], 1, 1),
            text_atts=text_atts[start + i].repeat(config['k_test'], 1)
        )[:, 0, :]
        
        score = model.itm_head(output)[:, 1]
        score_matrix_t2i[start + i, topk_idx] = score

    # === ???? 2: ??????? ===
    # ??????????? 1000.0 ???????????,???????
    min_values, _ = torch.min(score_matrix_t2i, dim=1)
    replacement_tensor = min_values.view(-1, 1).expand(-1, score_matrix_t2i.size(1))
    
    # ??????????,???????
    # ?????????,?????? 1000.0,???? replacement_tensor ???
    # ????,????? mask ??,??????,??????
    mask = (score_matrix_t2i == 1000.0)
    score_matrix_t2i[mask] = replacement_tensor[mask]

    # === ???? 3: Min-Max Normalization ===
    score_matrix_t2i = (score_matrix_t2i - score_matrix_t2i.min()) / (score_matrix_t2i.max() - score_matrix_t2i.min())

    # === ???? 4: ITC ??? ===
    score_sim_t2i = sims_matrix.clone()
    score_sim_t2i = (score_sim_t2i - score_sim_t2i.min()) / (score_sim_t2i.max() - score_sim_t2i.min())
    
    # === ???? 5: ???? ===
    print("Applying fusion: ITM + 0.002 * ITC")
    score_matrix_t2i = score_matrix_t2i + 0.002 * score_sim_t2i

    if args.distributed:
        dist.barrier()
        torch.distributed.all_reduce(score_matrix_t2i, op=torch.distributed.ReduceOp.SUM)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Computing matching score time {}'.format(total_time_str))
    
    return score_matrix_t2i.cpu().numpy()


def mAP(scores_t2i, g_pids, q_pids, table=None):
    # ????
    similarity = torch.tensor(scores_t2i)
    indices = torch.argsort(similarity, dim=1, descending=True)
    g_pids = torch.tensor(g_pids)
    q_pids = torch.tensor(q_pids)
    pred_labels = g_pids[indices.cpu()]
    matches = pred_labels.eq(q_pids.view(-1, 1))

    all_cmc = matches[:, :10].cumsum(1)
    all_cmc[all_cmc > 1] = 1
    all_cmc = all_cmc.float().mean(0) * 100

    num_rel = matches.sum(1)
    tmp_cmc = matches.cumsum(1)
    inp = [tmp_cmc[i][match_row.nonzero()[-1]] / (match_row.nonzero()[-1] + 1.) for i, match_row in enumerate(matches)]
    mINP = torch.cat(inp).mean() * 100

    tmp_cmc = [tmp_cmc[:, i] / (i + 1.0) for i in range(tmp_cmc.shape[1])]
    tmp_cmc = torch.stack(tmp_cmc, 1) * matches
    AP = tmp_cmc.sum(1) / num_rel
    mAP_score = AP.mean() * 100

    t2i_cmc, t2i_mAP, t2i_mINP = all_cmc.numpy(), mAP_score.numpy(), mINP.numpy()

    if not table:
        table = PrettyTable(["task", "R1", "R5", "R10", "mAP", "mINP"])
        table.add_row(['t2i', t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_mAP, t2i_mINP])
        table.custom_format["R1"] = lambda f, v: f"{v:.3f}"
        table.custom_format["R5"] = lambda f, v: f"{v:.3f}"
        table.custom_format["R10"] = lambda f, v: f"{v:.3f}"
        table.custom_format["mAP"] = lambda f, v: f"{v:.3f}"
        table.custom_format["mINP"] = lambda f, v: f"{v:.3f}"
        print(table)

    return {'R1': t2i_cmc[0], 'R5': t2i_cmc[4], 'R10': t2i_cmc[9], 'mAP': t2i_mAP, 'mINP': t2i_mINP}


def save_failure_cases(scores_t2i, g_pids, q_pids, dataset, output_dir, top_k=5):
    # ????
    print(f"### Analyzing Failure Cases...")
    if isinstance(g_pids, torch.Tensor): g_pids = g_pids.cpu().numpy()
    if isinstance(q_pids, torch.Tensor): q_pids = q_pids.cpu().numpy()
    g_pids, q_pids = np.array(g_pids), np.array(q_pids)

    failures = []
    for i, scores in enumerate(scores_t2i):
        pred_indices = np.argsort(scores)[::-1]
        top1_idx = pred_indices[0]
        query_pid = q_pids[i]
        pred_pid = g_pids[top1_idx]
        
        if query_pid != pred_pid:
            # ?? text ??? list ???
            txt = dataset.text[i]
            if isinstance(txt, list): txt = txt[0]

            case_info = {
                "query_id": int(i),
                "text": str(txt),
                "pred_pid": int(pred_pid),
                "gt_pid": int(query_pid),
                "pred_img": dataset.image[top1_idx]
            }
            failures.append(case_info)
    
    with open(os.path.join(output_dir, "failure_cases.json"), "w") as f:
        json.dump(failures, f, indent=4)
    print(f"### Saved {len(failures)} failure cases.")

def retrieval_eval(scores_i2t, scores_t2i, q_pids, g_pids, queries=None, image_paths=None, args=None, topk=5):
    # ????,???? Qwen ??
    if queries is not None and image_paths is not None and args is not None:
        candidates = []
        for i, scores in enumerate(scores_t2i):
            inds = np.argsort(scores)[::-1][:topk]
            cand_paths = [image_paths[idx] for idx in inds]
            
            txt = queries[i]
            if isinstance(txt, list): txt = txt[0]

            candidates.append({
                "query_id": i,
                "text": txt,
                "top_k_images": cand_paths
            })
        
        with open(os.path.join(args.output_dir, 'candidates.json'), 'w') as f:
            json.dump(candidates, f, indent=4)
        print("### Saved candidates.json.")