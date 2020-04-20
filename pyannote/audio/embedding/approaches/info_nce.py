#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2020 CNRS

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# AUTHORS
# Juan Manuel CORIA - https://juanmc2005.github.io

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import SelfSupervisedRepresentationLearning
from pyannote.audio.train.generator import Subset


class InfoNCELoss(SelfSupervisedRepresentationLearning):
    """Implementation of Contrastive Predictive Coding's
       Noise Contrastive Estimation based loss.

    The loss is calculated using approximated positive and negative
    examples in the batch.
    There are 2 examples from the same speaker and N from others.
    The loss is calculated as a regular cross entropy which maximizes
    the cosine similarity between the positives, while minimizing the
    one between each positive and all the negatives.

    Parameters
    ----------
    duration : `float`
        Chunks duration, in seconds. Defaults to 1.
    min_duration : `float`, optional
        When provided, use chunks of random duration between `min_duration` and
        `duration` for training. Defaults to using fixed duration chunks.
    per_segment : `int`
        Number of batches to build for each annotation.
    negatives : `int`
        Number of negative chunks to draw per annotation.
    batch_size : `int`
        Target batch size (> 2 * negatives + 2).
        If not enough negatives can be drawn to fill the specified size
        (or the minimum size), negatives are drawn from a fallback protocol.
    per_epoch : `float`, optional
        Force total audio duration per epoch, in days.
        Defaults to total duration of protocol subset.
    fallback_protocol : `str`
        The full name of the protocol to use as fallback.
        Ex. 'VoxCeleb.SpeakerVerification.VoxCeleb2'
    fallback_subset : `Subset` {'train', 'development', 'test'}
        The subset to use from the fallback protocol.

    Reference
    ---------
    Representation Learning with Contrastive Predictive Coding
    https://arxiv.org/abs/1807.03748
    """

    def __init__(self,
                 duration: float = 1.0,
                 min_duration: float = None,
                 per_segment: int = 3,
                 negatives: int = 10,
                 batch_size: int = None,
                 per_epoch: float = None,
                 fallback_protocol: str = None,
                 fallback_subset: Subset = 'train'):
        super().__init__(
            duration=duration,
            min_duration=min_duration,
            per_segment=per_segment,
            negatives=negatives,
            batch_size=batch_size,
            per_epoch=per_epoch,
            fallback_protocol=fallback_protocol,
            fallback_subset=fallback_subset)

        self.metric = 'cosine'
        self.loss_ = nn.NLLLoss()

    def batch_loss(self, batch):
        """Compute loss for current `batch`

        Parameters
        ----------
        batch : `dict`
            ['X'] (`numpy.ndarray`)

        Returns
        -------
        batch_loss : `dict`
            ['loss'] (`torch.Tensor`) : InfoNCE loss
        """
        # get embeddings
        fX = self.embed(batch)

        # calculate the similarity between the first examples and the rest
        cos1 = F.cosine_similarity(fX[0].unsqueeze(0), fX[1:])
        cos2 = F.cosine_similarity(fX[1].unsqueeze(0), fX[2:])
        cos2 = torch.cat((cos1[0].reshape(1), cos2))
        cos = torch.stack((cos1, cos2))

        # first similarity is a positive, the rest are negatives.
        # the 'correct' answer is the positive similarity, so it is
        # maximized, while the rest is minimized
        y = torch.tensor([0, 0]).to(self.device_)

        # calculate logits
        logits = F.log_softmax(cos, dim=1)

        # calculate loss
        loss = self.loss_(logits, y)

        return {'loss': loss}