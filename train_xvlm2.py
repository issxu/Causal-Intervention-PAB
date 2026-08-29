import torch
import torch.nn.functional as F
import utils

# MLM ??????
def mlm(text, text_input, tokenizer, device, mask_generator, config):
    text_masked = tokenizer(text, padding='max_length', truncation=True, max_length=config['max_tokens'],
                            return_tensors="pt").to(device)
    text_ids_masked = text_masked.input_ids
    masked_pos = torch.empty((text_ids_masked.shape[0], config['max_masks']), dtype=torch.int64, device=device)
    masked_ids = torch.empty((text_ids_masked.shape[0], config['max_masks']), dtype=torch.long, device=device)
    for index, text_id in enumerate(text_ids_masked):
        text_ids_masked_, masked_pos_ = mask_generator(text_id)
        masked_ids_ = [text_input.input_ids[index][p].item() for p in masked_pos_]
        n_pad = config['max_masks'] - len(masked_ids_)
        masked_pos_ = masked_pos_ + [0] * n_pad
        masked_pos_ = torch.tensor(masked_pos_, dtype=torch.int64).to(device)
        masked_ids_ = masked_ids_ + [-100] * n_pad
        masked_ids_ = torch.tensor(masked_ids_, dtype=torch.long).to(device)
        masked_pos[index] = masked_pos_
        masked_ids[index] = masked_ids_
    return text_ids_masked, masked_pos, masked_ids

def train_model(model, data_loader, optimizer, scaler, tokenizer, epoch, device, scheduler, config, mask_generator):
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_itc', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_itm', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_mlm', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    
    # [??] ?? loss_hard (???????)
    metric_logger.add_meter('loss_hard', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50

    # [??] ????? 5 ??? (??? text_eda)
    # ??????? search_dataset (1?+2?+?+ID)
    for i, (image, text_pos, text_neg_act, text_neg_app, idx) in enumerate(
            metric_logger.log_every(data_loader, print_freq, header)):

        image = image.to(device, non_blocking=True)
        idx = idx.to(device, non_blocking=True)

        # 1. Tokenize ???
        text_input = tokenizer(text_pos, padding='max_length', truncation=True, max_length=config['max_tokens'],
                               return_tensors="pt").to(device)
        
        # 2. Tokenize ???? (Baseline ?????)
        text_neg_act_input = tokenizer(text_neg_act, padding='max_length', truncation=True, max_length=config['max_tokens'],
                               return_tensors="pt").to(device)
        text_neg_app_input = tokenizer(text_neg_app, padding='max_length', truncation=True, max_length=config['max_tokens'],
                               return_tensors="pt").to(device)

        # MLM (?????)
        text_ids_masked, masked_pos, masked_ids = mlm(text_pos, text_input, tokenizer, device, mask_generator, config)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            # --- Part A: ???? Loss (ITC/ITM/MLM) ---
            # [??] ?? text_ids_eda,????????? EDA ??
            loss_itc, loss_itm, loss_mlm = model(
                image, text_input.input_ids, text_input.attention_mask,
                text_ids_masked=text_ids_masked, masked_pos=masked_pos, masked_ids=masked_ids,
                idx=idx
            )
            
            # --- Part B: ?? Hard Negative Mining (? TCL) ---
            # ??:?? 1???? ? 2???? ????,? CrossEntropy
            
            # 1. ????
            model_ptr = model.module if hasattr(model, 'module') else model
            
            image_embeds, _ = model_ptr.get_vision_embeds(image)
            image_feat = F.normalize(model_ptr.get_image_feat(image_embeds), dim=-1)

            def get_text_feature_safe(input_ids, attention_mask):
                embeds = model_ptr.get_text_embeds(input_ids, attention_mask)
                feat = model_ptr.get_text_feat(embeds)
                return F.normalize(feat, dim=-1)

            text_feat_pos = get_text_feature_safe(text_input.input_ids, text_input.attention_mask)
            text_feat_neg_act = get_text_feature_safe(text_neg_act_input.input_ids, text_neg_act_input.attention_mask)
            text_feat_neg_app = get_text_feature_safe(text_neg_app_input.input_ids, text_neg_app_input.attention_mask)

            # 2. ??????? [Batch, 1]
            sim_pos = (image_feat * text_feat_pos).sum(dim=1, keepdim=True)
            sim_neg_act = (image_feat * text_feat_neg_act).sum(dim=1, keepdim=True)
            sim_neg_app = (image_feat * text_feat_neg_app).sum(dim=1, keepdim=True)

            # 3. ?? 3?? logits
            # ??? [Batch, 3] -> [???, ???, ???]
            # ?? 0.07 ???? (????)
            temp_hard = 0.05 
            logits = torch.cat([sim_pos, sim_neg_act, sim_neg_app], dim=1) / temp_hard
            
            # 4. ???? 0 (????0?????????)
            labels = torch.zeros(logits.size(0), dtype=torch.long).to(device)
            
            # 5. ?? CrossEntropy Loss
            loss_hard = F.cross_entropy(logits, labels)
            
            # ? Loss (Baseline ?????????,?? CrossEntropy ????)
            # ??? 0.5 ??,??????
            loss = loss_itc + loss_itm + loss_mlm + (0.7 * loss_hard)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scale = scaler.get_scale()
        scaler.update()  
        skip_lr_sched = (scale > scaler.get_scale()) 
        if not skip_lr_sched:
            scheduler.step()
        optimizer.zero_grad()

        metric_logger.update(loss_itc=loss_itc.item())
        metric_logger.update(loss_itm=loss_itm.item())
        metric_logger.update(loss_mlm=loss_mlm.item())
        metric_logger.update(loss_hard=loss_hard.item())
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.5f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}