# MIT License
#
# Copyright (c) 2020 CNRS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import math
from typing import IO, Text, Union

import torch
import torch.nn as nn

from pyannote.audio.models.segmentation import PyanNet
from pyannote.audio.utils.params import merge_dict
from pyannote.core.utils.generators import pairwise


class PreAllocatedLinear(nn.Module):
    """Pre-allocated linear classification layer with fixed weights.

    Reference: Class-incremental Learning with Pre-allocated Fixed Classifiers
    Link: https://arxiv.org/abs/2010.08657

    Parameters
    ----------
    out_features : int
        Output dimension corresponding to the number of classes.
    """

    def __init__(self, out_features: int):
        super().__init__()
        self.in_features = out_features - 1
        self.out_features = out_features

        # linear.weight has shape (out_dim, in_dim)
        self.linear = nn.Linear(self.in_features, self.out_features, bias=False)

        # Fix weights to k-simplex with k=out_features
        with torch.no_grad():
            # Calculate last vertex coordinates
            numerator = 1 - math.sqrt(self.out_features)
            alpha = numerator / self.in_features
            # First k-1 vertices are the standard basis in R
            self.linear.weight[:-1] = torch.eye(self.in_features)
            self.linear.weight[-1] = torch.ones(self.in_features) * alpha
            # Center polytope
            self.linear.weight -= torch.mean(self.linear.weight, dim=0)

        # Freeze the layer
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Pass forward without gradient tracking.

        Parameters
        ----------
        inputs : (batch, frame, in_features)

        Returns
        -------
        outputs : (batch, frame, out_features)
        """
        with torch.no_grad():
            return self.linear(inputs)


class OnlinePyanNet(PyanNet):
    @classmethod
    def from_trunk(
        cls,
        trunk_model_checkpoint: Union[Text, IO],
        sincnet: dict = None,
        lstm: dict = None,
    ):
        """Build a PyanNet for online learning
        using pre-trained layers as the backbone.

        No linear layers are specified so they aren't
        initialized to corresponding pre-trained linear layers.

        Parameters
        ----------
        trunk_model_checkpoint : pre-trained checkpoint to load
        sincnet : dict with the desired SincNet config (see PyanNet)
        lstm : dict with the desired LSTM config (see PyanNet)

        Returns
        -------
        an OnlinePyanNet with specified layers loaded from the checkpoint
        """
        return cls.load_from_checkpoint(
            trunk_model_checkpoint,
            strict=False,
            sincnet=sincnet,
            lstm=lstm,
            linear={"num_layers": 0},
        )

    def with_linear(self, linear: dict = None):
        """Append linear layers to the backbone layers

        Parameters
        ----------
        linear : dict with the desired linear config (see PyanNet)

        Returns
        -------
        itself with new linear layers appended
        """
        linear = merge_dict(self.LINEAR_DEFAULTS, linear)
        self.save_hyperparameters("sincnet", "lstm", "linear")

        if linear["num_layers"] < 1:
            return

        lstm_out_features: int = self.hparams.lstm["hidden_size"] * (
            2 if self.hparams.lstm["bidirectional"] else 1
        )
        self.linear = nn.ModuleList(
            [
                nn.Linear(in_features, out_features)
                for in_features, out_features in pairwise(
                    [
                        lstm_out_features,
                    ]
                    + [self.hparams.linear["hidden_size"]]
                    * self.hparams.linear["num_layers"]
                )
            ]
        )

        return self

    def freeze_trunk(self):
        """Freeze the trunk of the architecture,
        consisting of SincNet and all LSTM layers.

        Returns
        -------
        itself after freezing
        """
        self.freeze_up_to(f"lstm.{self.hparams.lstm['num_layers'] - 1}")
        return self

    def build(self):
        # Build pre-allocated classifier for class-incremental learning
        out_features = len(self.specifications.classes)
        self.classifier = PreAllocatedLinear(out_features)

        # Get input dimension
        if self.hparams.linear["num_layers"] > 0:
            in_features = self.hparams.linear["hidden_size"]
        else:
            in_features = self.hparams.lstm["hidden_size"] * (
                2 if self.hparams.lstm["bidirectional"] else 1
            )

        # Input dimensions are constrained to num_classes - 1
        # because the classifier weights need to be a simplex
        # in num_classes - 1 dimensions
        if in_features != out_features - 1:
            # Transform input features to match the required dimension
            self.classifier = nn.Sequential(
                nn.Linear(in_features, out_features - 1),
                nn.LeakyReLU(),
                self.classifier,
            )

        # Use the default activation
        self.activation = self.default_activation()
