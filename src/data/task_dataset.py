from typing import List, Optional, Union

import torch
from torch.utils.data.dataset import Dataset
from transformers import PreTrainedTokenizer


from src.utils.misc import Split, MultiTaskDataArguments, MultiTaskInputFeatures
from src.data.glue_utils import load_glue_task_features


class TaskDataset(Dataset):
    features: List[MultiTaskInputFeatures]

    def __init__(
        self,
        task_name: str,
        task_id: int,
        args: MultiTaskDataArguments,
        tokenizer: PreTrainedTokenizer,
        limit_length: Optional[int] = None,
        mode: Union[str, Split] = Split.train,
    ):
        self.task_name = task_name
        self.features, self.labels = load_glue_task_features(
            task_name, task_id, args, tokenizer, mode, limit_length
        )

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, i) -> MultiTaskInputFeatures:
        return self.features[i]

    def get_labels(self) -> List[str]:
        return self.labels

