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

from copy import deepcopy
from typing import Callable, Iterable, List, Optional, Text

import numpy as np
import torch
from torch.nn import Parameter
from torch.optim import Optimizer
from torch_audiomentations.core.transforms_interface import BaseWaveformTransform

from pyannote.audio.core.task import Problem, Resolution, Specifications, Task
from pyannote.audio.pipelines import Segmentation as SegmentationPipeline
from pyannote.audio.tasks.segmentation.mixins import SegmentationTaskMixin
from pyannote.core import Segment, SlidingWindowFeature, Timeline
from pyannote.database import Protocol


class SpeakerTracking(SegmentationTaskMixin, Task):
    """Speaker tracking

    Speaker tracking is the process of determining if and when a (previously
    enrolled) person's voice can be heard in a given audio recording.

    Here, it is addressed with the same approach as voice activity detection,
    except {"non-speech", "speech"} classes are replaced by {"speaker1", ...,
    "speaker_N"} where N is the number of speakers in the training set.

    Parameters
    ----------
    protocol : Protocol
        pyannote.database protocol
    duration : float, optional
        Chunks duration. Defaults to 2s.
    batch_size : int, optional
        Number of training samples per batch. Defaults to 32.
    num_workers : int, optional
        Number of workers used for generating training samples.
        Defaults to multiprocessing.cpu_count() // 2.
    pin_memory : bool, optional
        If True, data loaders will copy tensors into CUDA pinned
        memory before returning them. See pytorch documentation
        for more details. Defaults to False.
    optimizer : callable, optional
        Callable that takes model parameters as input and returns
        an Optimizer instance. Defaults to `torch.optim.Adam`.
    learning_rate : float, optional
        Learning rate. Defaults to 1e-3.
    augmentation : BaseWaveformTransform, optional
        torch_audiomentations waveform transform, used by dataloader
        during training.
    """

    ACRONYM = "spk"

    def __init__(
        self,
        protocol: Protocol,
        duration: float = 2.0,
        batch_size: int = 32,
        num_workers: int = None,
        pin_memory: bool = False,
        optimizer: Callable[[Iterable[Parameter]], Optimizer] = None,
        learning_rate: float = 1e-3,
        augmentation: BaseWaveformTransform = None,
    ):

        super().__init__(
            protocol,
            duration=duration,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            optimizer=optimizer,
            learning_rate=learning_rate,
            augmentation=augmentation,
        )

        # for speaker tracking, task specification depends
        # on the data: we do not know in advance which
        # speakers should be tracked. therefore, we postpone
        # the definition of specifications.

    def setup(self, stage=None):

        super().setup(stage=stage)

        if stage == "fit":

            # build the list of speakers to be tracked.
            speakers = set()
            for f in self._train:
                speakers.update(f["annotation"].labels())

            # now that we now who the speakers are, we can
            # define the task specifications.

            # note that, since multiple speakers can be active
            # at once, the problem is multi-label classification.
            self.specifications = Specifications(
                problem=Problem.MULTI_LABEL_CLASSIFICATION,
                resolution=Resolution.FRAME,
                duration=self.duration,
                classes=sorted(speakers),
            )

    @property
    def chunk_labels(self) -> List[Text]:
        """Ordered list of labels

        Used by `prepare_chunk` so that y[:, k] corresponds to activity of kth speaker
        """
        return self.specifications.classes


class OnlineSpeakerTracking(SegmentationTaskMixin, Task):
    """Online Speaker tracking

    Speaker tracking is the process of determining if and when a (previously
    enrolled) person's voice can be heard in a given audio recording.

    Here, it is addressed with the same approach as voice activity detection,
    except {"non-speech", "speech"} classes are replaced by {"speaker1", ...,
    "speaker_N"} where N is the number of speakers in the training set.

    This class implements a variant where training data is a stream of audio
    that becomes available over time with a certain latency.

    Parameters
    ----------
    duration : float, optional
        Chunks duration. Defaults to 2s.
    batch_size : int, optional
        Number of training samples per batch. Defaults to 32.
    num_workers : int, optional
        Number of workers used for generating training samples.
        Defaults to multiprocessing.cpu_count() // 2.
    pin_memory : bool, optional
        If True, data loaders will copy tensors into CUDA pinned
        memory before returning them. See pytorch documentation
        for more details. Defaults to False.
    optimizer : callable, optional
        Callable that takes model parameters as input and returns
        an Optimizer instance. Defaults to `torch.optim.Adam`.
    learning_rate : float, optional
        Learning rate. Defaults to 1e-3.
    augmentation : BaseWaveformTransform, optional
        torch_audiomentations waveform transform, used by dataloader
        during training.
    speakers : int, optional
        Maximum number of speakers per conversation. Defaults to 20.
    sample_rate : int, optional
        The sample rate of the audio stream. Defaults to 16000.
    """

    ACRONYM = "spk"

    def __init__(
        self,
        duration: float = 2.0,
        batch_size: int = 32,
        num_workers: int = None,
        pin_memory: bool = False,
        optimizer: Callable[[Iterable[Parameter]], Optimizer] = None,
        learning_rate: float = 1e-3,
        augmentation: BaseWaveformTransform = None,
        speakers: int = 20,
        sample_rate: int = 16000,
    ):

        super().__init__(
            protocol=None,  # No need for a protocol in an online task
            duration=duration,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            optimizer=optimizer,
            learning_rate=learning_rate,
            augmentation=augmentation,
        )

        self._past_waveform: Optional[torch.Tensor] = None  # (1, samples)
        self._past_scores: Optional[SlidingWindowFeature] = None  # (frames, speakers)

        self.sample_rate = sample_rate
        self.num_past_chunks = 0
        self.chunk_size: int = round(duration * sample_rate)
        # Step size corresponding to the sliding window of the stream
        # Only 50% of the chunk duration is currently supported
        step = duration / 2
        self.step_size: int = round(step * sample_rate)

        # Task specification does not depend
        # on the data: we set an upper bound on
        # the number of speakers in each conversation.
        self.specifications = Specifications(
            problem=Problem.MULTI_LABEL_CLASSIFICATION,
            resolution=Resolution.FRAME,
            duration=self.duration,
            classes=[f"speaker_{i+1}" for i in range(speakers)],
        )

    @property
    def audio_duration(self) -> float:
        return self._past_waveform.shape[1] / self.sample_rate

    def update_database(self, chunk: torch.Tensor, scores: SlidingWindowFeature):
        # TODO save some chunks for validation
        replace = self._past_waveform is None or self._past_scores is None
        if replace:
            self._past_waveform = chunk
            self._past_scores = scores
        else:
            # Calculate first point of overlap between past audio and new audio chunk
            start_sample = self.num_past_chunks * self.step_size
            # Calculate number of samples of overlap
            overlap_samples = self._past_waveform.shape[1] - start_sample
            # Concatenate new samples at the end of the known audio stream
            self._past_waveform = torch.cat(
                [self._past_waveform, chunk[:, overlap_samples:]], dim=-1
            )

            current_scores = self._past_scores.data  # (past_frames, speakers)
            # Calculate first frame of overlap between past features and new chunk features
            start_frame, _ = self.model.introspection(start_sample)
            # Calculate number of frames of overlap
            overlap_frames = current_scores.shape[0] - start_frame
            # The new score for the overlapping half chunk is the mean between old and new
            # scores.data has shape (chunk_frames, speakers)
            current_scores[start_frame:] = (
                current_scores[start_frame:] + scores.data[:overlap_frames]
            ) / 2
            # The remaining scores are appended
            current_scores = np.vstack([current_scores, scores.data[overlap_frames:]])
            self._past_scores = SlidingWindowFeature(
                data=current_scores,
                sliding_window=self._past_scores.sliding_window,
                labels=self._past_scores.labels,
            )
        self.num_past_chunks += 1

    def setup(self, stage=None):
        # Default behavior from SegmentationMixin
        # needs a protocol to get the data from.
        # This is not the case here
        duration = self.audio_duration
        audio = {
            "waveform": self._past_waveform,
            "sample_rate": self.sample_rate,
            "duration": duration,
            "seg": self._past_scores,
        }
        self._train = [
            {
                "waveform": self._past_waveform,
                "sample_rate": self.sample_rate,
                "duration": duration,
                # Everything is annotated
                "annotated": Timeline([Segment(0, duration)]),
                # The hard labels are estimated using the
                # ever-evolving past y, coming from an
                # external local diarization model
                "annotation": SegmentationPipeline("seg")(audio),
            }
        ]
        # TODO add actual validation data
        self._validation = deepcopy(self._train)

    @property
    def chunk_labels(self) -> List[Text]:
        """Ordered list of labels

        Used by `prepare_chunk` so that y[:, k] corresponds to activity of kth speaker
        """
        return self.specifications.classes
