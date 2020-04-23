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
from .base import RepresentationLearning
from pyannote.audio.train.generator import Subset
from pyannote.audio.train.generator import BatchGenerator
from pyannote.audio.embedding.generators import MutualInformationBatchGenerator
from pyannote.audio.features import FeatureExtraction
from pyannote.database.protocol.protocol import Protocol
from pyannote.audio.features.wrapper import Wrappable
from pyannote.database import get_protocol
from pyannote.audio.features.utils import get_audio_duration
from pyannote.database.util import FileFinder


class InfoNCELoss(RepresentationLearning):
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
                 per_label: int = 1,
                 per_fold: int = None,
                 label_min_duration: float = 0.0,
                 per_segment: int = None,
                 negatives: int = None,
                 batch_size: int = None,
                 per_epoch: float = None,
                 fallback_protocol: str = None,
                 fallback_subset: Subset = 'train',
                 **kwargs):
        super().__init__(
            duration=duration,
            min_duration=min_duration,
            per_turn=1,
            per_label=per_label,
            per_fold=per_fold,
            per_epoch=per_epoch,
            label_min_duration=label_min_duration)

        self.metric = 'cosine'
        self.loss_ = nn.NLLLoss()
        self.per_segment = per_segment
        self.negatives = negatives
        self.batch_size = batch_size
        self.fallback_subset = fallback_subset
        # TODO how to add augmentation to this protocol?
        self.fallback_protocol = get_protocol(fallback_protocol,
                                              preprocessors={
                                                  'audio': FileFinder(),
                                                  'duration': get_audio_duration})

    def get_batch_generator(
        self,
        feature_extraction: Wrappable,
        protocol: Protocol,
        subset: Subset = "train",
        **kwargs
    ) -> BatchGenerator:
        """Get batch generator

        Parameters
        ----------
        feature_extraction : `FeatureExtraction`
        protocol : `Protocol`
        subset : {'train', 'development', 'test'}, optional

        Returns
        -------
        generator : `BatchGenerator`
        """

        return MutualInformationBatchGenerator(
            feature_extraction,
            protocol,
            self.fallback_protocol,
            self.min_duration,
            self.duration,
            self.per_segment,
            self.negatives,
            self.per_epoch,
            self.batch_size,
            subset,
            self.fallback_subset
        )

    def batch_loss(self, batch):
        """Compute loss for current `batch`

        Parameters
        ----------
        batch : `dict`
            ['X'] (`numpy.ndarray`)
            ['y'] (`numpy.ndarray`)

        Returns
        -------
        batch_loss : `dict`
            ['loss'] (`torch.Tensor`) : InfoNCE loss
        """
        # get embeddings
        fX, _ = self.embed(batch)

        # calculate the similarity between the first examples and the rest
        cos1 = F.cosine_similarity(fX[0].unsqueeze(0), fX[1:])
        cos2 = F.cosine_similarity(fX[1].unsqueeze(0), fX[2:])
        cos2 = torch.cat((cos1[0].reshape(1), cos2))
        cos = torch.stack((cos1, cos2))

        # format ground truth, which should be [0, 0],
        # as the first similarity is the one to maximize
        y = torch.tensor([0, 0]).to(self.device_)

        # calculate logits
        logits = F.log_softmax(cos, dim=1)

        # calculate loss
        loss = self.loss_(logits, y)

        return {'loss': loss}