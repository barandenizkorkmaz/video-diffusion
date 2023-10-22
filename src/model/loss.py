import torch
import torch.nn.functional as F
from src.utils.data import to_tensor

def masked_mse_loss(input, target, mask):
    input, target, mask = to_tensor(input, target, mask)
    mask_sum = mask.sum() * input.nelement() / mask.nelement()
    return F.mse_loss(input * mask, target * mask, reduction="sum") / mask_sum


def masked_rmse_loss(input, target, mask):
    input, target, mask = to_tensor(input, target, mask)
    return torch.sqrt(masked_mse_loss(input, target, mask))


def masked_mae_loss(input, target, mask):
    input, target, mask = to_tensor(input, target, mask)
    mask_sum = mask.sum() * input.nelement() / mask.nelement()
    return F.l1_loss(input * mask, target * mask, reduction="sum") / mask_sum