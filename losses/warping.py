from __future__ import annotations

import numpy as np
from scipy import ndimage
import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
import cv2

from losses.utils import DiceType
from .dice_losses import Multiclass_CLDice
import typing
if typing.TYPE_CHECKING:
    from jaxtyping import Float


class HomotopyWarpingLoss(_Loss):
    """
        2D implementation of homotopy warping based on https://github.com/HuXiaoling/Warping/tree/main
        and https://arxiv.org/pdf/2112.07812
    """
    def __init__(self, softmax: bool=True, eight_connectivity: bool=False) -> None:
        super().__init__()
        self.softmax = softmax
        if eight_connectivity:
            self.fg_connectivity = 8
            self.bg_connectivity = 4
        else:
            self.fg_connectivity = 4
            self.bg_connectivity = 8

    def decide_simple_point(self, target, x, y):
        if x < 1 or y < 1 or x >= target.shape[0]-1 or y >= target.shape[1]-1:
            return target #TODO: decide what to do
        patch = target[x-1:x+2, y-1:y+2]

        ccs_fg, _ = cv2.connectedComponents(patch, self.fg_connectivity)
        ccs_bg, _ = cv2.connectedComponents(patch, self.bg_connectivity)

        label = (ccs_fg - 1) * (ccs_bg - 1)
        if label == 1:
            target[x, y] = 1 - target[x, y] #flip

        return target

    def update_simple_point(self, distance, target):
        non_zero_distance = np.nonzero(distance)
        idx = np.unravel_index(np.argsort(-distance, axis=None), distance.shape)

        for i in range(len(non_zero_distance[0])):
            x = idx[0][len(non_zero_distance[0]) - i -1]
            y = idx[0][len(non_zero_distance[0]) - i - 1]

            target = self.decide_simple_point(target, x, y)

        return target

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.float()
        assert len(target.shape) == 4
        assert len(input.shape) == 4
        B, C, H, W = target.shape

        probs = F.softmax(input, dim=1)

        pred = torch.argmax(probs, dim=1)
        if C == 2: #TODO: probably unnecessary
            target = torch.unsqueeze(target[:, 0, :, :], dim=1)

        predictions = pred.detach().cpu().numpy()
        predictions_c = predictions.copy()
        target_np = target.detach().cpu().numpy()
        target_c = target_np.copy()

        critical_points = np.zeros((B, H, W))
        for i in range(B):
            fp = ((predictions_c[i, :, :] - target_c[i, :, :]) == 1).astype(int)
            fn = ((target_c[i, :, :] - predictions_c[i, :, :]) == 1).astype(int)

            fn_distance_gt = ndimage.distance_transform_edt(target_c[i, :, :]) * fn
            fp_distance_gt = ndimage.distance_transform_edt(1 - target_c[i, :, :]) * fp

            target_warp = self.update_simple_point(fn_distance_gt, target_c[i, :, :])
            target_warp = self.update_simple_point(fp_distance_gt, target_warp)

            fn_distance_pre = ndimage.distance_transform_edt(1-predictions_c[i,:,:]) * fn # grow gt while keep unconnected
            fp_distance_pre = ndimage.distance_transform_edt(predictions_c[i,:,:]) * fp  # shrink pre while keep connected

            pre_warp = self.update_simple_point(fp_distance_pre, predictions_c[i,:,:])
            pre_warp = self.update_simple_point(fn_distance_pre, pre_warp)

            critical_points[i, :, :] = np.logical_or(np.not_equal(predictions[i,:,:], target_warp), np.not_equal(target_np[i,:,:], pre_warp)).astype(int)

        critical_points = torch.from_numpy(critical_points).to(device=input.device)
        masked_input = input * torch.unsqueeze(critical_points, dim=1)
        masked_target = (target * torch.unsqueeze(critical_points, dim=1)).long()

        warping_loss = F.cross_entropy(masked_input, torch.squeeze(masked_target, dim=1)) * len(np.nonzero(critical_points)[0])

        return warping_loss


class DiceHomotopyWarpingLoss(_Loss):
    def __init__(self,
                 dice_type: DiceType=DiceType.CLDICE,
                 cldice_alpha: float=0.5,
                 include_background: bool=True,
                 eight_connectivity: bool=False,
                 ) -> None:
        super().__init__()
        if dice_type == DiceType.DICE:
            self.DiceLoss = Multiclass_CLDice(
                softmax=True, 
                include_background=True, # irrelevant because pure Dice always uses background 
                smooth=1e-5, 
                alpha=0.0,
                convert_to_one_vs_rest=False,
                batch=True,
            )
        elif dice_type == DiceType.CLDICE:
            self.DiceLoss = Multiclass_CLDice(
                softmax=True, 
                include_background=include_background, 
                smooth=1e-5, 
                alpha=cldice_alpha, 
                iter_=5, 
                convert_to_one_vs_rest=False,
                batch=True
            )
        else:
            raise ValueError(f"Invalid dice type: {dice_type}")

        self.warpingLoss = HomotopyWarpingLoss(
            eight_connectivity=eight_connectivity,
        )
        
    def forward(self, 
                prediction: Float[torch.Tensor, "batch channel *spatial_dimensions"], 
                target: Float[torch.Tensor, "batch channel *spatial_dimensions"],
                alpha: float = 0.5, #lambda in paper
                ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # Compute multiclass BM losses
        losses = {}
        if alpha > 0:
            warping_loss = self.warpingLoss(prediction, target)
        else:
            warping_loss = torch.zeros(1, device=prediction.device)

        # Multiclass Dice loss
        dice_loss, dic = self.DiceLoss(prediction, target)
        
        losses["dice"] = dic["dice"]
        losses["cldice"] = dic["cldice"]
        losses["warping_loss"] = alpha * warping_loss

        return dice_loss + alpha * warping_loss, losses
