from torch.nn import init
from torch import nn
from models.cmp_xvlm2 import CMP
import torch

class Search(CMP):
    def __init__(self, config):
        super().__init__(config,)
        # self.log_sigma_itc = nn.Parameter(torch.tensor(-0.5))  # 对应 sigma ≈ 0.6
        # self.log_sigma_itm = nn.Parameter(torch.tensor(-0.5))
        # self.log_sigma_mlm = nn.Parameter(torch.tensor(-0.5))

    def forward(self, image, text_ids, text_atts, text_ids_masked=None, masked_pos=None, masked_ids=None,
                idx=None, text_ids_eda=None, text_atts_eda=None,
                ):

        image_embeds, image_atts = self.get_vision_embeds(image)
        text_embeds = self.get_text_embeds(text_ids, text_atts)

        image_feat, text_feat = self.get_image_feat(image_embeds), self.get_text_feat(text_embeds)
        loss_itc = self.get_contrastive_loss(image_feat, text_feat, idx=idx)
        loss_itm = self.get_matching_loss(image_embeds, image_atts, image_feat,
                                          text_embeds, text_atts, text_feat, idx=idx)

        # eda
        if text_ids_eda is not None and text_atts_eda is not None:
            text_embeds_eda = self.get_text_embeds(text_ids_eda, text_atts_eda)
            text_feat_eda = self.get_text_feat(text_embeds_eda)
            loss_itc_eda = self.get_contrastive_loss(image_feat, text_feat_eda, idx=idx)
            loss_itm_eda = self.get_matching_loss(image_embeds, image_atts, image_feat,
                                                  text_embeds_eda, text_atts_eda, text_feat_eda, idx=idx, )
            loss_itc = loss_itc + 0.8 * loss_itc_eda
            loss_itm = loss_itm + 0.8 * loss_itm_eda
    
        loss_mlm = self.get_mlm_loss(text_ids_masked, text_atts, image_embeds, image_atts,
                                         masked_pos, masked_ids, )
        # lossall = (
        #     (loss_itc / (2 * torch.exp(self.log_sigma_itc)**2) + self.log_sigma_itc) +
        #     (loss_itm / (2 * torch.exp(self.log_sigma_itm)**2) + self.log_sigma_itm) +
        #     (loss_mlm / (2 * torch.exp(self.log_sigma_mlm)**2) + self.log_sigma_mlm)
        # )
        return loss_itc, loss_itm, loss_mlm
