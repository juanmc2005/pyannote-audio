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

from pyannote.database.protocol.protocol import Protocol, ProtocolFile
from itertools import islice
import numpy as np
from pyannote.core import Segment, Timeline
from pyannote.core.utils.random import random_subsegment
from pyannote.audio.train.task import Task, TaskType, TaskOutput
from pyannote.audio.train.generator import BatchGenerator, Subset
from pyannote.audio.features.wrapper import Wrapper, Wrappable
from pyannote.audio.embedding.generators import SpeechSegmentGenerator
from pyannote.core.utils.distance import to_condensed


def random_chunks(segment: Segment, duration: float, n: int):
    """Draw `n` random chunks of `duration` from a specific `segment`

    :param segment: a Segment
    :param duration: chunk duration
    :param n: number of chunks to draw
    :return: an iterator over `n` random chunks in `segment`
    """
    return islice(random_subsegment(segment, duration), n)


class SelfSupervisedBatchGenerator(BatchGenerator):

    def __init__(self,
                 feature_extraction: Wrappable,
                 protocol: Protocol,
                 fallback_protocol: Protocol,
                 min_duration: float,
                 max_duration: float,
                 per_epoch: float = None,
                 subset: Subset = 'train',
                 fallback_subset: Subset = 'train',
                 fallback_label_max_duration: float = np.inf):
        """A base speech chunk batch generator class for self-supervised objectives.

        :param feature_extraction: a FeatureExtraction-compatible object
        :param protocol: a Protocol to draw files
        :param fallback_protocol: a Protocol to fill batches when
            there aren't enough negative samples for a batch
        :param min_duration: minimum chunk duration
        :param max_duration: maximum chunk duration
        :param per_epoch: epoch duration in days
        :param subset: main protocol subset {'train', 'development', 'test'}
        :param subset: fallback protocol subset {'train', 'development', 'test'}
        """
        super().__init__(feature_extraction, protocol, subset)
        self.feature_extraction = Wrapper(feature_extraction)
        self.min_duration = min_duration
        self.max_duration = max_duration

        self.data_, total_duration = self.load_metadata(protocol, subset)
        if per_epoch is None:
            per_epoch = total_duration / (24 * 60 * 60)
        self.per_epoch = per_epoch

        self.fallback_generator = SpeechSegmentGenerator(
            feature_extraction, fallback_protocol, subset=fallback_subset,
            per_label=1, per_turn=1, per_fold=1, duration=max_duration,
            label_max_duration=fallback_label_max_duration)

    def load_metadata(self, protocol: Protocol, subset: Subset):
        """Load all protocol files and return total speech duration.

        :param protocol: a Protocol instance
        :param subset: a Subset ('train', 'development', 'test')
        :return: a tuple
            - data: list of protocol files in the original order
            - total_duration: the total usable speech duration
        """
        total_duration = 0
        data = []
        for file in getattr(protocol, subset)():
            # ensure speaker turns are cropped to actual file duration
            support = Segment(start=0, end=file["duration"])
            file["annotated"] = file["annotated"].crop(support, mode="intersection")
            total_duration += sum([s.duration
                                   for s in file['annotated']
                                   if s.duration > self.max_duration])
            data.append(file)
        return data, total_duration

    def new_duration(self):
        return self.min_duration + np.random.rand() * (self.max_duration - self.min_duration)

    def is_previous_valid(self, segments: Timeline, i: int) -> bool:
        return i > 0 \
               and segments[i].start - segments[i-1].end > 0 \
               and segments[i-1].duration > self.max_duration

    def is_next_valid(self, segments: Timeline, i: int) -> bool:
        return i < len(segments) - 1 \
               and segments[i+1].start - segments[i].end > 0 \
               and segments[i+1].duration > self.max_duration

    def package(self, file: ProtocolFile, chunk: Segment):
        """Package a sample in the expected dictionary format.
           Note that a `y` key is a placeholder, as the identity
           of the speaker is not known.

        :param file: a ProtocolFile instance
        :param chunk: the chunk to package
        :return: a dict of
            - X: extracted features for `chunk`
        """
        feat = self.feature_extraction.crop(file, chunk,
                                            mode='center',
                                            fixed=chunk.duration)
        # dummy y
        return {'X': feat, 'y': np.array([0], dtype=np.long)}

    def package_fallback(self, sample: dict, duration: float):
        """Package a sample from the fallback protocol.
           This is different than a chunk generated by this class
           because we receive an already packaged sample that
           needs to be reformatted with only the `X` key.
           Additionally, samples coming from the fallback protocol
           are of `max_duration`, so an additional cropping is needed.

        :param sample: a single sample batch packaged in a dict
            with shape (1, dim)
        :param duration: the current batch duration
        :return: a dict of
            - X: extracted features from cropped sample
        """
        file = {'waveform': sample['X'].reshape(-1, 1)}
        segment = Segment(0, self.max_duration)
        chunk = next(random_subsegment(segment, duration))
        feat = self.feature_extraction.crop(file, chunk,
                                            mode='center',
                                            fixed=duration)
        # dummy y
        return {'X': feat, 'y': np.array([0], dtype=np.long)}

    @property
    def batches_per_epoch(self) -> int:
        """Lazily approximate the number of batches per epoch.

        :return: the number of batches per epoch
        """
        # duration per epoch
        duration_per_epoch = self.per_epoch * 24 * 60 * 60
        # (average) duration per batch
        duration_per_batch = 0.5 * (self.min_duration + self.max_duration) * self.batch_size
        # number of batches per epoch
        return int(np.ceil(duration_per_epoch / duration_per_batch))

    @property
    def specifications(self):
        """Lazily calculate the specifications for a batch.

        :return: a dict describing a batch
        """
        return {
            "X": {"dimension": self.feature_extraction.dimension},
            # dummy class specification
            "y": {"classes": {'0': ''}},
            "task": Task(
                type=TaskType.REPRESENTATION_LEARNING, output=TaskOutput.VECTOR
            ),
        }


class MutualInformationBatchGenerator(SelfSupervisedBatchGenerator):

    def __init__(self,
                 feature_extraction: Wrappable,
                 protocol: Protocol,
                 fallback_protocol: Protocol,
                 min_duration: float,
                 max_duration: float,
                 per_segment: int = None,
                 negatives: int = None,
                 per_epoch: float = None,
                 batch_size: int = None,
                 subset: Subset = 'train',
                 fallback_subset: Subset = 'train',
                 fallback_label_max_duration: float = np.inf):
        """A speech segment generator for non-temporal InfoNCE objectives.
        It builds batches with 2 positive examples and N-2 negative ones.

        :param feature_extraction: a FeatureExtraction-compatible object
        :param protocol: a Protocol to draw files
        :param fallback_protocol: a Protocol to fill batches when
            there aren't enough samples for a batch
        :param min_duration: minimum chunk duration
        :param max_duration: maximum chunk duration
        :param per_segment: optional, number of batches to build per segment
        :param negatives: optional, number of negative samples
            to draw from consecutive segments.
            Has to be specified if `batch_size` is None
        :param per_epoch: epoch duration in days
        :param batch_size: optional, it cannot be less than `2 * negatives + 2`.
            It will fill batches to match this size with negatives from the fallback protocol.
            Has to be specified if `negatives` is None.
            Defaults to the minimum value.
        :param subset: a protocol subset ('train', 'development', 'test')
        """
        super().__init__(feature_extraction, protocol, fallback_protocol,
                         min_duration, max_duration, per_epoch, subset,
                         fallback_subset, fallback_label_max_duration)

        if negatives is None and batch_size is None:
            msg = 'Either `negatives` or `batch_size` needs ' \
                  'to be specified for this batch generator'
            raise ValueError(msg)

        self.per_segment = per_segment
        self.n_neg = negatives
        self.target_batch_size = batch_size

        if self.n_neg is None:
            # infer number of negatives from target batch size
            self.n_neg = int(.5 * (self.target_batch_size - 2))
        elif self.target_batch_size is None:
            # infer batch size from number of negatives
            self.target_batch_size = 2 * self.n_neg + 2

    def samples(self):
        """Iterate endlessly over files (randomly)
           and segments (randomly), drawing probable
           positive and negative chunks for a self-supervised objective

        :return: an iterator over samples
        """
        while True:
            # shuffle files
            files_chosen = np.random.permutation(len(self.data_))
            for i in files_chosen:
                file = self.data_[i]
                segments = file['annotation'].get_timeline(copy=False)
                # shuffle segments
                chosen = np.random.permutation(len(segments))
                for j in chosen:
                    # skip segments shorter than maximum chunk duration
                    if segments[j].duration <= self.max_duration:
                        continue
                    # build `per_segment` batches for this segment
                    for _ in range(self.segment_batches(segments[j])):
                        for sample in self.batch_samples(file, segments, j):
                            yield sample

    def batch_samples(self, file: ProtocolFile, segments: Timeline, i: int):
        """Draw positive and negative samples to form a batch
           for the given file and segment.
           If not enough negatives can be approximated to fill
           the batch, then complete with the fallback protocol.

        :param file: a ProtocolFile
        :param segments: a Timeline with all available segments
        :param i: index corresponding to the segment to use for positives
        :return: an iterator over samples until `batch_size` is reached
        """
        # choose next batch's duration randomly
        duration = self.new_duration()

        # choose 2 random positive chunks
        yield self.package(file, next(random_subsegment(segments[i], duration)))
        yield self.package(file, next(random_subsegment(segments[i], duration)))

        # count samples in batch
        yielded = 2

        # choose random negative chunks from previous segment if possible
        if self.is_previous_valid(segments, i):
            for chunk in random_chunks(segments[i-1], duration, self.n_neg):
                yield self.package(file, chunk)
                yielded += 1

        # choose random negative chunks from next segment if possible
        if self.is_next_valid(segments, i):
            for chunk in random_chunks(segments[i+1], duration, self.n_neg):
                yield self.package(file, chunk)
                yielded += 1

        # complete batch with fallback protocol
        while yielded < self.batch_size:
            sample = next(self.fallback_generator.samples())
            yield self.package_fallback(sample, duration)
            yielded += 1

    def segment_batches(self, segment: Segment) -> int:
        """Estimate the number of batches to generate for
           a segment if `per_segment` is not specified.
           Calculate how many chunks of `max_duration` fit
           in `segment`, and divide this number by 2 to
           get the number of batches (2 positives per batch).

        :param segment: `Segment`
        :return: number of batches
        """
        if self.per_segment is None:
            # divide by 2 because we need 2 positives per batch
            return int(.5 * segment.duration / self.max_duration)
        else:
            return self.per_segment

    @property
    def batch_size(self) -> int:
        """Lazily calculate the batch size for this generator.
           An imposed minimum for the size is `2 * n_neg + 2`.
           If no `n_neg` is specified, then `target_batch_size`
           is used.

        :return: the batch size
        """
        minimum = 2 * self.n_neg + 2
        return minimum if self.target_batch_size < minimum \
                       else self.target_batch_size


class ContrastiveBatchGenerator(SelfSupervisedBatchGenerator):

    def __init__(self,
                 feature_extraction: Wrappable,
                 protocol: Protocol,
                 fallback_protocol: Protocol,
                 min_duration: float,
                 max_duration: float,
                 per_label: int,
                 per_fold: int,
                 per_epoch: float = None,
                 subset: Subset = 'train',
                 fallback_subset: Subset = 'train',
                 fallback_label_max_duration: float = np.inf):
        super().__init__(feature_extraction, protocol, fallback_protocol,
                         min_duration, max_duration, per_epoch, subset,
                         fallback_subset, fallback_label_max_duration)
        self.per_label = per_label
        self.per_fold = per_fold

    def samples(self):
        """Iterate endlessly over files (randomly)
           and segments (randomly), drawing `per_label` chunks
           from `2 * per_fold` segments to form a batch.

        :return: an iterator over samples
        """
        # choose next batch's duration randomly
        batch_size = 0
        duration = self.new_duration()
        while True:
            # shuffle files
            files_chosen = np.random.permutation(len(self.data_))
            for i in files_chosen:
                file = self.data_[i]
                segments = file['annotation'].get_timeline(copy=False)
                # shuffle segments
                segments_chosen = np.random.permutation(len(segments))
                for j in segments_chosen:
                    # skip segments shorter than maximum chunk duration
                    if segments[j].duration <= self.max_duration:
                        continue
                    # yield positive and negative samples for this segment
                    for sample in self.batch_samples(file, segments, duration, j):
                        yield sample
                        batch_size += 1
                    if batch_size == self.batch_size:
                        # choose next batch's duration randomly
                        batch_size = 0
                        duration = self.new_duration()

    def batch_samples(self, file, segments, duration, j):
        # yield positives
        for chunk in random_chunks(segments[j], duration, self.per_label):
            yield self.package(file, chunk)

        if self.is_next_valid(segments, j):
            for chunk in random_chunks(segments[j+1], duration, self.per_label):
                yield self.package(file, chunk)
        elif self.is_previous_valid(segments, j):
            for chunk in random_chunks(segments[j-1], duration, self.per_label):
                yield self.package(file, chunk)
        # otherwise yield negatives from the fallback protocol
        else:
            for sample in islice(self.fallback_generator.samples(), self.per_label):
                yield self.package_fallback(sample, duration)

    @property
    def batch_size(self) -> int:
        # batch size is always the number of chunks per segment (`per_label`)
        # times 2, because we look at a consecutive segment for negatives,
        # times `per_fold` segments that we look at each time.
        return 2 * self.per_fold * self.per_label

    @property
    def positive_indices(self) -> list:
        """Calculate the indices of positive distances in a condensed
        distance matrix, based on the batch size.
        This indices can be precalculated thanks to the specific
        structure given to batches.

        Returns
        -------
        positives : `list`
            Indices of positive distances in a condensed distance matrix
            for a batch
        """
        positives = []
        for i in range(0, self.batch_size, self.per_label):
            for j in range(i + 1, i + self.per_label):
                positives.append(to_condensed(self.batch_size, i, j))
        return positives

    @property
    def negative_indices(self) -> list:
        """Calculate the indices of negative distances in a condensed
        distance matrix, based on the batch size.
        This indices can be precalculated thanks to the specific
        structure given to batches.

        Returns
        -------
        negatives : `list`
            Indices of negative distances in a condensed distance matrix
            for a batch
        """
        negatives = []
        step = 2 * self.per_label
        for i in range(0, self.batch_size - step, step):
            for j in range(i, i + self.per_label):
                for k in range(i + self.per_label, i + step):
                    negatives.append(to_condensed(self.batch_size, j, k))
        return negatives
