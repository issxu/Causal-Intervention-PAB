import torch
import torch.nn as nn
import torch.nn.functional as F

class TCL_Loss(nn.Module):
    def __init__(self, evidence_type='softplus', tau=0.1, num_classes=2):
        """
        tau: ????,????????? (?? 0.05 ~ 0.1)
        evidence_type: 'softplus' ? 'exp' ???,? NaN
        """
        super(TCL_Loss, self).__init__()
        self.evidence_type = evidence_type
        self.tau = tau
        self.num_classes = num_classes

    def get_evidence(self, similarity):
        # ? [-1, 1] ??????????
        if self.evidence_type == 'exp':
            return torch.exp(torch.clamp(similarity / self.tau, -10, 10))
        elif self.evidence_type == 'softplus':
            return F.softplus(similarity / self.tau)
        return F.relu(similarity / self.tau)

    def kl_divergence(self, alpha):
        # ??????????????(?? DECL)
        # ?????? [1, 1] ?????
        beta = torch.ones((alpha.shape[0], self.num_classes)).to(alpha.device)
        S_alpha = torch.sum(alpha, dim=1, keepdim=True)
        S_beta = torch.sum(beta, dim=1, keepdim=True)
        
        # ?? lgamma ????
        lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
        lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
        
        dg0 = torch.digamma(S_alpha)
        dg1 = torch.digamma(alpha)
        
        kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
        return kl.mean()

    def forward(self, similarity, target_type='pos'):
        # 1. ??? -> ??
        evidence = self.get_evidence(similarity)
        
        # 2. ?? Dirichlet ?? alpha
        # ?????:?? vs ????????????????????
        # alpha = evidence + 1
        
        if target_type == 'pos':
            # --- ??? (Risk Loss) ---
            # ??:?????? (???????)
            alpha = evidence + 1
            S = alpha + 1 # ????????0,??? S = (e+1) + 1
            # Digamma Loss
            loss = torch.digamma(S) - torch.digamma(alpha)
            return loss.mean()

        elif target_type == 'neg':
            # --- ??? (KL Loss) ---
            # ??:?????? (??????)
            alpha_neg = evidence + 1
            # ????? alpha ?? [alpha_neg, 1] ???? KL
            # ??? 1 ????????????
            ones = torch.ones_like(alpha_neg)
            alpha_full = torch.cat([alpha_neg, ones], dim=1) 
            return self.kl_divergence(alpha_full)