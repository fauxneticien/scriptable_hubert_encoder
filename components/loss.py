import torch

class HuBERTPretrainingLoss(torch.nn.Module):

  def __init__(self, label_pad_index=-100, pred_masked_weight=1.0, pred_nomask_weight=0.0):
    super().__init__()

    self.ce_loss = torch.nn.CrossEntropyLoss(ignore_index=label_pad_index, reduction="sum")
    self.pred_masked_weight = pred_masked_weight
    self.pred_nomask_weight = pred_nomask_weight

  def forward(self, logits_m, logits_u, masks_for_modeling, ptlabels_padded):
    loss = 0.0

    if logits_m != None and self.pred_masked_weight > 0:
      labels_m = ptlabels_padded[masks_for_modeling]
      loss += self.pred_masked_weight * self.ce_loss(logits_m, labels_m)

    if logits_u != None and self.pred_nomask_weight > 0:
      labels_u = ptlabels_padded[~masks_for_modeling]
      loss += self.pred_nomask_weight * self.ce_loss(logits_u, labels_u)

    return loss
