#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2018-2020 CNRS

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
# Hervé BREDIN - http://herve.niderb.fr
# Juan Manuel CORIA - https://juanmc2005.github.io

import torch
from .base import RepresentationLearning
from pyannote.audio.train.generator import Subset
from pyannote.audio.train.generator import BatchGenerator
from pyannote.audio.embedding.generators import ContrastiveBatchGenerator
from pyannote.database.protocol.protocol import Protocol
from pyannote.audio.features.wrapper import Wrappable
from pyannote.database import get_protocol
from pyannote.audio.features.utils import get_audio_duration
from pyannote.database.util import FileFinder


class ContrastiveLoss(RepresentationLearning):
    """Contrastive loss

    Train embeddings by bringing together positive pairs (same speaker)
    and separating negative pairs (different speaker).

    Parameters
    ----------
    duration : float, optional
        Chunks duration, in seconds. Defaults to 1.
    min_duration : float, optional
        When provided, use chunks of random duration between `min_duration` and
        `duration` for training. Defaults to using fixed duration chunks.
    per_turn : int, optional
        Number of chunks per speech turn. Defaults to 1.
        If per_turn is greater than one, embeddings of the same speech turn
        are averaged before comparison. The intuition is that it might help
        learn embeddings meant to be averaged/summed.
    per_label : `int`, optional
        Number of sequences per speaker in each batch. Defaults to 1.
    per_fold : `int`, optional
        Number of different speakers per batch. Defaults to 32.
    per_epoch : `float`, optional
        Force total audio duration per epoch, in days.
        Defaults to total duration of protocol subset.
    label_min_duration : `float`, optional
        Remove speakers with less than that many seconds of speech.
        Defaults to 0 (i.e. keep them all).
    metric : {'euclidean', 'cosine', 'angular'}, optional
        Defaults to 'cosine'.
    margin: float, optional
        Margin multiplicative factor. Defaults to 0.2.

    Reference
    ---------
    Dimensionality Reduction by Learning an Invariant Mapping
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(
        self,
        duration: float = 1.0,
        min_duration: float = None,
        per_turn: int = 1,
        per_label: int = 1,
        per_fold: int = 32,
        per_epoch: float = None,
        label_min_duration: float = 0.0,
        # FIXME create a Literal type for metric
        # FIXME maybe in pyannote.core.utils.distance
        metric: str = "cosine",
        # FIXME homogeneize the meaning of margin parameter
        # FIXME it has a different meaning in ArcFace, right?
        margin: float = 0.2,
    ):

        super().__init__(
            duration=duration,
            min_duration=min_duration,
            per_turn=per_turn,
            per_label=per_label,
            per_fold=per_fold,
            per_epoch=per_epoch,
            label_min_duration=label_min_duration,
        )

        self.metric = metric
        self.margin = margin
        # FIXME see above
        self.margin_ = self.margin * self.max_distance

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
            ['loss'] (`torch.Tensor`) : Triplet loss
        """

        fX, y = self.embed(batch)

        # calculate the distances between every sample in the batch
        batch_size = fX.size(0)
        dist = self.pdist(fX).to(self.device_)

        # calculate the ground truth for each pair
        # TODO. this can be done much more cleanly with
        # pyannote.core.utils.distance.pdist(y, metric='equal')
        gt = []
        for i in range(batch_size - 1):
            for j in range(i + 1, batch_size):
                gt.append(int(y[i] != y[j]))
        gt = torch.Tensor(gt).float().to(self.device_)

        # Calculate the losses as described in the paper
        losses = (1 - gt) * torch.pow(dist, 2) + gt * torch.pow(
            torch.clamp(self.margin_ - dist, min=1e-8), 2
        )

        # FIXME: why divive by 2?
        losses = torch.sum(losses) / 2
        # Average by batch size if requested
        # FIXME: switch to torch.mean directly (size_average has been removed)
        loss = losses / dist.size(0)

        return {"loss": loss, "loss_contrastive": loss}


class SelfSupervisedContrastiveLoss(RepresentationLearning):
    """Contrastive loss for self-supervised applications

    Train embeddings by bringing together positive pairs (same speaker)
    and separating negative pairs (different speaker).
    These pairs are approximated using a custom batch generator.

    Parameters
    ----------
    duration : float, optional
        Chunks duration, in seconds. Defaults to 1.
    min_duration : float, optional
        When provided, use chunks of random duration between `min_duration` and
        `duration` for training. Defaults to using fixed duration chunks.
    per_label : `int`, optional
        Number of sequences per speaker in each batch. Defaults to 1.
    per_fold : `int`, optional
        Number of different speakers per batch. Defaults to 32.
    per_epoch : `float`, optional
        Force total audio duration per epoch, in days.
        Defaults to total duration of protocol subset.
    label_min_duration : `float`, optional
        Remove speakers with less than that many seconds of speech.
        Defaults to 0 (i.e. keep them all).
    metric : {'euclidean', 'cosine', 'angular'}, optional
        Defaults to 'cosine'.
    margin: float, optional
        Margin multiplicative factor. Defaults to 0.2.
    fallback_protocol : `str`
        If not enough negatives can be found by the batch generator, then
        fill the batch with speech chunks from this protocol.
    fallback_subset: `Subset`, optional
        The subset to use from the fallback protocol. Defaults to 'train'.

    Reference
    ---------
    Dimensionality Reduction by Learning an Invariant Mapping
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(
            self,
            duration: float = 1.0,
            min_duration: float = None,
            per_label: int = 1,
            per_fold: int = 32,
            per_epoch: float = None,
            label_min_duration: float = 0.0,
            # FIXME create a Literal type for metric
            # FIXME maybe in pyannote.core.utils.distance
            metric: str = "cosine",
            # FIXME homogeneize the meaning of margin parameter
            # FIXME it has a different meaning in ArcFace, right?
            margin: float = 0.2,
            fallback_protocol: str = None,
            fallback_subset: Subset = 'train',
            **kwargs
    ):
        super().__init__(
            duration=duration,
            min_duration=min_duration,
            per_turn=1,
            per_label=per_label,
            per_fold=per_fold,
            per_epoch=per_epoch,
            label_min_duration=label_min_duration,
        )

        self.metric = metric
        self.margin = margin
        # FIXME see above
        self.margin_ = self.margin * self.max_distance
        self.fallback_subset = fallback_subset
        self.i_positive, self.i_negative = None, None
        # FIXME this might not be the optimal place to create the protocol
        self.fallback_protocol = get_protocol(fallback_protocol,
                                              progress=True,
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
        generator = ContrastiveBatchGenerator(
            feature_extraction,
            protocol,
            self.fallback_protocol,
            self.min_duration,
            self.duration,
            self.per_label,
            self.per_fold,
            self.per_epoch,
            subset,
            self.fallback_subset)
        # we can safely initialize positive and negative indices here
        # because the batch generator is needed to calculate the first batch's loss
        self.i_positive = generator.positive_indices
        self.i_negative = generator.negative_indices
        return generator

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
            ['loss'] (`torch.Tensor`) : Triplet loss
        """
        # get batch embeddings
        fX, _ = self.embed(batch)

        # calculate the distances between every sample in the batch
        dist = self.pdist(fX).to(self.device_)

        # calculate positive losses
        pos_loss = torch.pow(dist[self.i_positive], 2)
        # calculate negative losses
        neg_loss = torch.pow(torch.clamp(self.margin_ - dist[self.i_negative], min=1e-8), 2)
        # average loss
        loss = torch.cat((pos_loss, neg_loss)).mean()

        return {"loss": loss}
