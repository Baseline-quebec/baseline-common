"""Phases of a training run."""

from enum import StrEnum


class Phase(StrEnum):
    """Phase of a training run; also names the data split each phase consumes."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"
