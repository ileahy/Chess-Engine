import torch
import torch.nn.functional as F


def chess_loss(policy_logits, value_pred, policy_target, value_target):
    
    policy_loss = F.cross_entropy(policy_logits, policy_target)

    value_loss = F.mse_loss(value_pred.squeeze(1), value_target)

    return policy_loss, value_loss