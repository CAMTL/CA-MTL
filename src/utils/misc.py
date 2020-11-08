from enum import Enum
from dataclasses import dataclass, field
from typing import List

from transformers import InputFeatures


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"


@dataclass(frozen=True)
class MultiTaskInputFeatures(InputFeatures):
    task_id: int = None


@dataclass
class MultiTaskDataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    data_dir: str = field(
        metadata={
            "help": "The input data dir. Should contain the .tsv files (or other data files) for the task."
        }
    )
    tasks: List[str] = field(
        default=None,
        metadata={
            "help": "The task file that contains the tasks to train on. If None all tasks will be used"
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )

    def __post_init__(self):
        if self.tasks is None:
            self.tasks = [
                "cola",
                "mnli",
                "rte",
                "wnli",
                "qqp",
                "sts-b",
                "sst-2",
                "qnli",
            ]
