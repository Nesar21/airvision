import torch
import torch.nn.functional as F

def masked_mae(pred, target, mask):
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device)
    return F.l1_loss(pred[mask], target[mask])

def multimodal_loss(outputs, batch, weights=None):
    la = masked_mae(outputs["aqi"],  batch["aqi"],  batch["mask_aqi"])
    lp = masked_mae(outputs["pm25"], batch["pm25"], batch["mask_pm25"])
    lt = masked_mae(outputs["pm10"], batch["pm10"], batch["mask_pm10"])

    return la + lp + lt, {"aqi": la.item(), "pm25": lp.item(), "pm10": lt.item()}
