#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2017-2020 CNRS

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

from typing import Optional
from typing import Text
from pyannote.database.protocol.protocol import Protocol, ProtocolFile
import itertools
import numpy as np
from pyannote.core import Segment, Timeline
from pyannote.core.utils.random import random_segment
from pyannote.core.utils.random import random_subsegment
from pyannote.audio.train.task import Task, TaskType, TaskOutput
from ..train.generator import BatchGenerator, Subset
from pyannote.audio.features.wrapper import Wrapper, Wrappable


class SpeechSegmentGenerator(BatchGenerator):
    """Generate batch of pure speech segments with associated speaker labels

    Parameters
    ----------
    feature_extraction : `pyannote.audio.features.FeatureExtraction`
        Feature extraction.
    protocol : `pyannote.database.Protocol`
    subset : {'train', 'development', 'test'}
    duration : float, optional
        Chunks duration, in seconds. Defaults to 1.
    min_duration : float, optional
        When provided, generate chunks of random duration between `min_duration`
        and `duration`. All chunks in a batch will still use the same duration.
        Defaults to generating fixed duration chunks.
    per_turn : int, optional
        Number of chunks per speech turn. Defaults to 1.
    per_label : int, optional
        Number of speech turns per speaker in each batch.
        Defaults to 3.
    per_fold : int, optional
        Number of different speakers in each batch.
        Defaults to all speakers.
    per_epoch : float, optional
        Force total audio duration per epoch, in days.
        Defaults to total duration of protocol subset.
    label_min_duration : float, optional
        Remove speakers with less than `label_min_duration` seconds of speech.
        Defaults to 0 (i.e. keep it all).
    """

    def __init__(
        self,
        feature_extraction: Wrappable,
        protocol: Protocol,
        subset: Text = "train",
        duration: float = 1.0,
        min_duration: float = None,
        per_turn: int = 1,
        per_label: int = 3,
        per_fold: Optional[int] = None,
        per_epoch: float = None,
        label_min_duration: float = 0.0,
    ):

        self.feature_extraction = Wrapper(feature_extraction)
        self.per_turn = per_turn
        self.per_label = per_label
        self.per_fold = per_fold
        self.duration = duration
        self.min_duration = duration if min_duration is None else min_duration
        self.label_min_duration = label_min_duration
        self.weighted_ = True

        total_duration = self._load_metadata(protocol, subset=subset)
        if per_epoch is None:
            per_epoch = total_duration / (24 * 60 * 60)
        self.per_epoch = per_epoch

    def _load_metadata(self, protocol: Protocol, subset: Text = "train") -> float:
        """Load training set metadata

        This function is called once at instantiation time, returns the total
        training set duration, and populates the following attributes:

        Attributes
        ----------
        data_ : dict
            Dictionary where keys are speaker labels and values are lists of
            (segments, duration, current_file) tuples where
            - segments is a list of segments by the speaker in the file
            - duration is total duration of speech by the speaker in the file
            - current_file is the file (as ProtocolFile)

        segment_labels_ : list
            Sorted list of (unique) labels in protocol.

        file_labels_ : dict of list
            Sorted lists of (unique) file-level labels in protocol

        Returns
        -------
        duration : float
            Total duration of annotated segments, in seconds.
        """

        self.data_ = {}
        segment_labels, file_labels = set(), dict()

        # loop once on all files
        for current_file in getattr(protocol, subset)():

            # keep track of unique file labels
            for key in current_file:
                if key in ["annotation", "annotated", "audio", "duration"]:
                    continue
                if key not in file_labels:
                    file_labels[key] = set()
                file_labels[key].add(current_file[key])

            # get annotation for current file
            # ensure annotation is cropped to actual file duration
            support = Segment(start=0, end=current_file["duration"])
            current_file["annotation"] = current_file["annotation"].crop(
                support, mode="intersection"
            )
            annotation = current_file["annotation"]

            # loop on each label in current file
            for label in annotation.labels():

                # get all segments with current label
                timeline = annotation.label_timeline(label)

                # remove segments shorter than maximum chunk duration
                segments = [s for s in timeline if s.duration > self.duration]

                # corner case where no segment is long enough
                # and we removed them all...
                if not segments:
                    continue

                # total duration of label in current_file (after removal of
                # short segments).
                duration = sum(s.duration for s in segments)

                # store all these in data_ dictionary
                # datum = (segment_generator, duration, current_file, features)
                datum = (segments, duration, current_file)
                self.data_.setdefault(label, []).append(datum)

        # remove labels with less than 'label_min_duration' of speech
        # otherwise those may generate the same segments over and over again
        dropped_labels = set()
        for label, data in self.data_.items():
            total_duration = sum(datum[1] for datum in data)
            if total_duration < self.label_min_duration:
                dropped_labels.add(label)

        for label in dropped_labels:
            self.data_.pop(label)

        self.file_labels_ = {k: sorted(file_labels[k]) for k in file_labels}
        self.segment_labels_ = sorted(self.data_)

        return sum(sum(datum[1] for datum in data) for data in self.data_.values())

    def samples(self):

        labels = list(self.data_)

        # batch_counter counts samples in current batch.
        # as soon as it reaches batch_size, a new random duration is selected
        # so that the next batch will use a different chunk duration
        batch_counter = 0
        batch_size = self.batch_size
        batch_duration = self.min_duration + np.random.rand() * (
            self.duration - self.min_duration
        )

        while True:

            # shuffle labels
            np.random.shuffle(labels)

            # loop on each label
            for label in labels:

                # load data for this label
                # segment_generators, durations, files, features = \
                #     zip(*self.data_[label])
                segments, durations, files = zip(*self.data_[label])

                # choose 'per_label' files at random with probability
                # proportional to the total duration of 'label' in those files
                probabilities = durations / np.sum(durations)
                chosen = np.random.choice(
                    len(files), size=self.per_label, p=probabilities
                )

                # loop on (randomly) chosen files
                for i in chosen:

                    # choose one segment at random with
                    # probability proportional to duration
                    # segment = next(segment_generators[i])
                    segment = next(random_segment(segments[i], weighted=self.weighted_))

                    # choose per_turn chunk(s) at random
                    for chunk in itertools.islice(
                        random_subsegment(segment, batch_duration), self.per_turn
                    ):

                        yield {
                            "X": self.feature_extraction.crop(
                                files[i], chunk, mode="center", fixed=batch_duration
                            ),
                            "y": self.segment_labels_.index(label),
                        }

                        # increment number of samples in current batch
                        batch_counter += 1

                        # as soon as the batch is complete, a new random
                        # duration is selected so that the next batch will use
                        # a different chunk duration
                        if batch_counter == batch_size:
                            batch_counter = 0
                            batch_duration = self.min_duration + np.random.rand() * (
                                self.duration - self.min_duration
                            )

    @property
    def batch_size(self) -> int:
        if self.per_fold is not None:
            return self.per_turn * self.per_label * self.per_fold
        return self.per_turn * self.per_label * len(self.data_)

    @property
    def batches_per_epoch(self) -> int:

        # duration per epoch
        duration_per_epoch = self.per_epoch * 24 * 60 * 60

        # (average) duration per batch
        duration_per_batch = 0.5 * (self.min_duration + self.duration) * self.batch_size

        # number of batches per epoch
        return int(np.ceil(duration_per_epoch / duration_per_batch))

    @property
    def specifications(self):
        return {
            "X": {"dimension": self.feature_extraction.dimension},
            "y": {"classes": self.segment_labels_},
            "task": Task(
                type=TaskType.REPRESENTATION_LEARNING, output=TaskOutput.VECTOR
            ),
        }


def random_chunks(segment: Segment, duration: float, n: int):
    """Draw `n` random chunks of `duration` from a specific `segment`

    :param segment: a Segment
    :param duration: chunk duration
    :param n: number of chunks to draw
    :return: an iterator over `n` random chunks in `segment`
    """
    return itertools.islice(random_subsegment(segment, duration), n)


class LifelongBatchGenerator(BatchGenerator):

    def __init__(self,
                 feature_extraction: Wrappable,
                 protocol: Protocol,
                 fallback_protocol: Protocol,
                 per_segment: int,
                 negatives: int,
                 min_duration: float,
                 max_duration: float,
                 per_epoch: float = None,
                 batch_size: int = None,
                 subset: Subset = 'train',
                 fallback_subset: Subset = 'train'):
        """A speech chunk batch generator for a lifelong step with a single file.

        :param feature_extraction: a FeatureExtraction-compatible object
        :param protocol: a Protocol to draw files
        :param fallback_protocol: a Protocol to fill batches when
            there aren't enough samples for a batch
        :param per_segment: number of batches to build per segment
        :param negatives: number of negative samples
            to draw from consecutive segments
        :param min_duration: minimum chunk duration
        :param max_duration: maximum chunk duration
        :param per_epoch: epoch duration in days
        :param batch_size: optional, it cannot be less than `2 * negatives + 2`.
            It will fill batches to match this size with negatives from the fallback protocol.
            Defaults to the minimum value.
        :param subset: a protocol subset ('train', 'development', 'test')
        """
        super().__init__(feature_extraction, protocol, subset)
        self.feature_extraction = Wrapper(feature_extraction)
        self.per_segment = per_segment
        self.n_neg = negatives
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.target_batch_size = batch_size

        self.data_, total_duration = self.load_metadata(protocol, subset)
        if per_epoch is None:
            per_epoch = total_duration / (24 * 60 * 60)
        self.per_epoch = per_epoch

        self.fallback_generator = SpeechSegmentGenerator(feature_extraction,
                                                         fallback_protocol,
                                                         subset=fallback_subset,
                                                         per_label=1,
                                                         per_turn=1,
                                                         per_fold=1,
                                                         duration=max_duration)

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

    def samples(self):
        """Iterate endlessly over files (in order)
           and segments (randomly), drawing probable
           positive and negative chunks for a self-supervised objective

        :return: an iterator over samples
        """
        while True:
            # don't shuffle files because lifelong learning is sequential
            for file in self.data_:
                segments = file['annotated']
                # shuffle segments
                chosen = np.random.choice(len(segments), size=len(segments))
                for i in chosen:
                    # skip segments shorter than maximum chunk duration
                    if segments[i].duration <= self.max_duration:
                        continue
                    # build `per_segment` batches for this segment
                    for _ in range(self.per_segment):
                        for sample in self.batch_samples(file, segments, i):
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
        duration = self.min_duration + np.random.rand() * (self.max_duration - self.min_duration)

        # choose 2 random positive chunks
        yield self.package(file, next(random_subsegment(segments[i], duration)))
        yield self.package(file, next(random_subsegment(segments[i], duration)))

        # count samples in batch
        yielded = 2

        # choose random negative chunks from previous segment if possible
        if i > 0 and segments[i-1].duration > self.max_duration:
            for chunk in random_chunks(segments[i-1], duration, self.n_neg):
                yield self.package(file, chunk)
                yielded += 1

        # choose random negative chunks from next segment if possible
        if i < len(segments) - 1 and segments[i+1].duration > self.max_duration:
            for chunk in random_chunks(segments[i+1], duration, self.n_neg):
                yield self.package(file, chunk)
                yielded += 1

        # complete batch with fallback protocol
        if yielded < self.batch_size:
            while yielded < self.batch_size:
                sample = next(self.fallback_generator.samples())
                yield self.package_fallback(sample, duration)
                yielded += 1

    def package(self, file: ProtocolFile, chunk: Segment):
        """Package a sample in the expected dictionary format.
           Note that a `y` key is not provided as the identity
           of the speaker is not known.

        :param file: a ProtocolFile instance
        :param chunk: the chunk to package
        :return: a dict of
            - X: extracted features for `chunk`
        """
        feat = self.feature_extraction.crop(file, chunk,
                                            mode='center',
                                            fixed=chunk.duration)
        return {'X': feat}

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
            - X: cropped sample with shape (dim,)
        """
        file = {'waveform': sample['X'].reshape(-1, 1)}
        segment = Segment(0, self.max_duration)
        chunk = next(random_subsegment(segment, duration))
        feat = self.feature_extraction.crop(file, chunk,
                                            mode='center',
                                            fixed=duration)
        return {'X': feat}

    @property
    def batch_size(self) -> int:
        """Lazily calculate the batch size for this generator.
           An imposed minimum for the size is `2 * n_neg + 2`.
           Although this is not strictly speaking a minimum, it
           is what's recommended, as many negatives are needed
           in a contrastive self-supervised objective.

        :return: the batch size
        """
        minimum = 2 * self.n_neg + 2
        if self.target_batch_size is None or self.target_batch_size < minimum:
            return minimum
        else:
            return self.target_batch_size

    @property
    def batches_per_epoch(self) -> int:
        """Lazily calculate the number of batches per epoch.

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
            "task": Task(
                type=TaskType.REPRESENTATION_LEARNING, output=TaskOutput.VECTOR
            ),
        }
