import os
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Callable, Tuple

import torch
import numpy as np
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
from transformers import Trainer, TrainingArguments, EvalPrediction, glue_output_modes
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from src.data.glue_utils import compute_glue_metrics

logger = logging.getLogger(__name__)


@dataclass
class MultiTaskTrainingArguments(TrainingArguments):
    use_mt_uncertainty: bool = field(
        default=False,
        metadata={"help": "Use MT-Uncertainty sampling method"},
    )
    uniform_mt_sampling: bool = field(
        default=False,
        metadata={"help": "Sample each task an equal amount to times per epoch."},
    )
    percent_of_max_data_size: float = field(
        default=1.0,
        metadata={
            "help": "If uniform_mt_sampling=True, specify the samples per task per "
            "epoch based on the maximum dataset length. If below 0.0 or above 1.0,"
            "it will be set to the closest of 0.0 or 1.0."
        },
    )
    warmup_proportion: float = field(
        default=0.1,
        metadata={"help": "0.0 to args.lr for warmup_proportion * num_training_steps"},
    )


class MultiTaskTrainer(Trainer):
    def __init__(
        self,
        tokenizer,
        data_args,
        eval_datasets=None,
        test_datasets=None,
        *args,
        **kwargs,
    ):
        super(MultiTaskTrainer, self).__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.eval_datasets = eval_datasets
        self.test_datasets = test_datasets
        self.eval_results = {}

    def get_optimizers(
        self, num_training_steps: int
    ) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well.
        If you want to use something else, you can pass a tuple in the Trainer's init,
        or override this method in a subclass.
        """
        if self.optimizers is not None:
            return self.optimizers
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        num_warmup_steps = (
            self.args.warmup_proportion * num_training_steps
        )  # this is different from overridden function
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.args.learning_rate,
            eps=self.args.adam_epsilon,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,  # this is different from overridden function
        )
        return optimizer, scheduler

    def get_train_dataloader(self) -> DataLoader:
        if self.args.use_mt_uncertainty:
            return self._create_custom_dataloader()
        else:
            return super().get_train_dataloader()

    def _create_custom_dataloader(self):
        class MtUcertaintyIterator:
            """Sample tasks using uncertainty measure."""

            def __init__(self, my_loader):
                self.my_loader = my_loader
                self.loader_iters = [iter(loader) for loader in self.my_loader.loaders]
                self.loader_iter_sizes = [len(i) for i in self.loader_iters]
                self.max_count = len(self.my_loader)
                self.batch_count = 0

            def __iter__(self):
                return self

            def __next__(self):
                if self.batch_count == self.max_count:
                    self.batch_count = 0
                    raise StopIteration()

                test_batch = {}
                for idx, loader_iter in enumerate(self.loader_iters):
                    try:
                        batch = loader_iter.__next__()
                    except StopIteration:
                        new_loader_iter = iter(self.my_loader.loaders[idx])
                        self.loader_iters[idx] = new_loader_iter
                        batch = new_loader_iter.__next__()

                    test_batch = self.batchify_data(batch, test_batch)

                inputs = {}
                for k, v in test_batch.items():
                    if k not in ["labels"]:
                        inputs[k] = v.detach().to(self.my_loader.args.device)

                with torch.no_grad():
                    model.select_batch_mode = True
                    outputs = model(**inputs)
                    model.select_batch_mode = False

                (
                    test_batch_entropy,
                    test_batch_entropy_mean,
                    max_mean_batch_entropy,
                ) = outputs[-3:]

                for _, v in inputs.items():
                    del v  # free GPU mem
                del inputs

                test_batch_entropy_mean = (
                    test_batch_entropy_mean / max_mean_batch_entropy
                )
                test_batch_entropy = test_batch_entropy * test_batch_entropy_mean

                if "sts-b" in tasks and "mrpc" in tasks:
                    # Since sts_b is a regression task, the entropy value is always 0
                    stsb_idx = test_batch["task_id"] == tasks.index("sts-b")
                    mrpc_idx = test_batch["task_id"] == tasks.index("sst-2")
                    num_items = min(
                        len(test_batch_entropy[stsb_idx]),
                        len(test_batch_entropy[mrpc_idx]),
                    )
                    stsb_idx = stsb_idx.nonzero()[:num_items]
                    mrpc_idx = mrpc_idx.nonzero()[:num_items]
                    test_batch_entropy[stsb_idx] = test_batch_entropy[mrpc_idx]
                    test_batch_entropy_mean[stsb_idx] = test_batch_entropy_mean[
                        mrpc_idx
                    ]

                select_size = min(
                    self.my_loader.args.train_batch_size,
                    test_batch["input_ids"].shape[0],
                )  # Handled the last batch if it is lower than the batch size

                top_entropy = torch.topk(test_batch_entropy, select_size)

                for k, v in test_batch.items():
                    test_batch[k] = torch.index_select(v, 0, top_entropy.indices)

                self.batch_count += 1

                return test_batch

            @staticmethod
            def batchify_data(data, curr_batch):
                for k in data.keys():
                    if k in curr_batch.keys():
                        curr_batch[k] = torch.cat((curr_batch[k], data[k]), dim=0)
                    else:
                        curr_batch[k] = data[k]
                return curr_batch

        class CustomLoader:
            def __init__(self, loaders, datasets, loader_args):
                self.loaders = loaders
                self.dataset = datasets
                self.args = loader_args
                self.current_epoch = 0

            def __iter__(self):
                iterator = MtUcertaintyIterator(self)

                # for determinism across runs
                # https://github.com/pytorch/examples/issues/501
                for l in self.loaders:
                    if isinstance(l.sampler, DistributedSampler):
                        l.sampler.set_epoch(self.current_epoch)
                self.current_epoch += 1
                return iterator

            def __len__(self):
                loader_len = [len(loader) for loader in self.loaders]
                if self.args.uniform_mt_sampling:
                    return int(
                        self.args.percent_of_max_data_size
                        * max(loader_len)
                        * len(self.loaders)
                        / self.args.train_batch_size
                    )
                elif self.args.use_mt_uncertainty:
                    return int(
                        max(loader_len)
                        * len(self.loaders)
                        * self.args.percent_of_max_data_size
                    )
                else:
                    return sum(loader_len)

        model = self.model
        tasks = self.data_args.tasks

        data_loaders = []
        for dataset in self.train_dataset.datasets:
            train_sampler = (
                RandomSampler(dataset)
                if self.args.local_rank == -1
                else DistributedSampler(dataset)
            )

            data_loader = DataLoader(
                dataset,
                batch_size=self.args.train_batch_size,
                sampler=train_sampler,
                collate_fn=self.data_collator.collate_batch,
            )
            data_loaders.append(data_loader)

        return CustomLoader(data_loaders, self.train_dataset, self.args)

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        prediction_loss_only: Optional[bool] = None,
        context: str = None,
        do_test_if_needed: bool = True,
    ):
        datasets = eval_dataset or self.eval_datasets
        logger.info("*** Evaluate on dev ***")
        for task_name, eval_dataset in datasets.items():
            logger.info(task_name)
            self.compute_metrics = self.build_compute_metrics_fn(eval_dataset)
            eval_result = super().evaluate(
                eval_dataset=eval_dataset, prediction_loss_only=True
            )
            self.update_eval_results(eval_result, task_name)
            for key, value in self.eval_results[task_name].items():
                logger.info("  %s = %s", key, value)

    def predict(
        self,
        eval_dataset: Optional[Dataset] = None,
        prediction_loss_only: Optional[bool] = None,
    ):
        logging.info("*** Test ***")
        datasets = eval_dataset or self.test_datasets
        for task_name, test_dataset in datasets.items():
            logger.info(task_name)
            predictions = super().predict(test_dataset=test_dataset).predictions
            output_mode = glue_output_modes[task_name] 
            if output_mode == "classification":
                predictions = np.argmax(predictions, axis=1)

            output_test_file = os.path.join(
                self.args.output_dir,
                f"{task_name}_test_iter_{self.global_step}.tsv",
            )
            if self.is_world_master():
                with open(output_test_file, "w") as writer:
                    logger.info("***** Test results {} *****".format(task_name))
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        if output_mode == "regression":
                            writer.write("%d\t%3.3f\n" % (index, item))
                        else:
                            writer.write(
                                "%d\t%s\n" % (index, test_dataset.get_labels()[item])
                            )

    def update_eval_results(self, results, task_name):
        self.eval_results[task_name] = self.eval_results.get(task_name, {})
        for key, value in results.items():
            if key in self.eval_results[task_name] and 'loss' not in key and 'epoch' not in key:
                value = max(self.eval_results[task_name][key], value)
            self.eval_results[task_name][key] = value


    @staticmethod
    def build_compute_metrics_fn(
        eval_dataset
    ) -> Callable[[EvalPrediction], Dict]:
        def compute_metrics_fn(p: EvalPrediction):
            return compute_glue_metrics(eval_dataset.task_name, p)

        return compute_metrics_fn
