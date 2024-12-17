from typing import Union

import numpy as np
import torch

from scipy.ndimage import label
from skimage.metrics import variation_of_information
from monai.metrics.utils import do_metric_reduction, is_binary_tensor
from monai.utils.enums import MetricReduction

from monai.metrics.metric import CumulativeIterationMetric

class VOIMetric(CumulativeIterationMetric):
    def __init__(self, ignore_background=False, eight_connectivity=False, reverse_partitioning=False) -> None:
        super().__init__()
        self.ignore_background = ignore_background
        self.eight_connectivity = eight_connectivity
        self.reduction = MetricReduction.MEAN
        self.reverse_partitioning = reverse_partitioning

    def _compute_tensor(self, y_pred: torch.Tensor, y: torch.Tensor):  # type: ignore
        """
        Args:
            y_pred: input data to compute, typical segmentation model output.
                It must be one-hot format and first dim is batch, second dim is classes, example shape: [16, 11, 32, 32]. The values
                should be binarized. The first class is considered as background and is ignored.
            y: ground truth to compute mean dice metric. It must be one-hot format and first dim is batch.
                The values should be binarized. The first class is considered as background and is ignored.

        Raises:
            ValueError: when `y` is not a binarized tensor.
            ValueError: when `y_pred` has less than three dimensions.
        """
        is_binary_tensor(y_pred, "y_pred")
        is_binary_tensor(y, "y")

        dims = y_pred.ndimension()
        if dims != 4:
            raise ValueError(f"y_pred should have 4 dimensions (batch, channel, height, width), got {dims}.")
        
        skip_index = 1 if self.ignore_background else 0 # skip the background class if it is ignored
        
        # Convert to numpy
        y_pred = y_pred.cpu().numpy()
        y = y.cpu().numpy()

        adapted_rand_scores = torch.zeros((y_pred.shape[0], y_pred.shape[1] - skip_index))

        for i in range(y_pred.shape[0]):
            for c in range(skip_index, y_pred.shape[1]): # ignore background
                if self.reverse_partitioning:
                    structure_element = np.ones((3, 3)) if not self.eight_connectivity else None
                    pred = 1 - y_pred[i, c]
                    gt = 1 - y[i, c]
                else:
                    structure_element = np.ones((3, 3)) if self.eight_connectivity else None
                    pred = y_pred[i, c]
                    gt = y[i, c]


                pred_ccs, _ = label(pred, structure=structure_element)
                gt_ccs, _ = label(gt, structure=structure_element)

                entr1, entr2 = variation_of_information(pred_ccs, gt_ccs)

                # Compute the adapted rand score
                adapted_rand_scores[i, c - skip_index] = entr1 + entr2

        # convert back to tensor and return
        return adapted_rand_scores

    def aggregate(self, reduction: Union[MetricReduction, str, None] = None):  # type: ignore
        """
        Execute reduction logic for the output of `compute_meandice`.

        Args:
            reduction: define mode of reduction to the metrics, will only apply reduction on `not-nan` values,
                available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
                ``"mean_channel"``, ``"sum_channel"``}, default to `self.reduction`. if "none", will not do reduction.

        """
        data = self.get_buffer()
        if not isinstance(data, torch.Tensor):
            raise ValueError("the data to aggregate must be PyTorch Tensor.")

        # do metric reduction
        f, _ = do_metric_reduction(data, reduction=self.reduction or reduction) # type: ignore
        return f