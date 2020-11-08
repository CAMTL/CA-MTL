from typing import List, Optional, Union

import torch
from transformers import PreTrainedTokenizer
from torch.utils.data import ConcatDataset, TensorDataset

from src.utils.misc import Split, MultiTaskDataArguments
from src.data.task_dataset import TaskDataset


class MultiTaskDataset(ConcatDataset):
    def __init__(
        self,
        data_args: MultiTaskDataArguments,
        tokenizer: PreTrainedTokenizer,
        limit_length: Optional[int] = None,
        mode: Split = Split.train,
    ):
        datasets = [
            TaskDataset(
                task_name, task_id, data_args, tokenizer, limit_length=limit_length, mode=mode
            )
            for task_id, task_name in enumerate(data_args.tasks)
        ]

        super().__init__(datasets)
