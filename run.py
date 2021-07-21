import os
import sys
import re
import json
import logging

import torch
from transformers import (
    HfArgumentParser,
    set_seed,
    AutoTokenizer,
    AutoConfig,
    EvalPrediction,
)

from src.model.ca_mtl import CaMtl, CaMtlArguments
from src.utils.misc import MultiTaskDataArguments, Split
from src.mtl_trainer import MultiTaskTrainer, MultiTaskTrainingArguments
from src.data.mtl_dataset import MultiTaskDataset
from src.data.task_dataset import TaskDataset

logger = logging.getLogger(__name__)


def setup_logging(training_args):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )


def parse_cmd_args():
    parser = HfArgumentParser(
        (
            CaMtlArguments,
            MultiTaskDataArguments,
            MultiTaskTrainingArguments,
        )
    )

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        (
            model_args,
            data_args,
            training_args,
        ) = parser.parse_args_into_dataclasses()

    logger.info("Training/evaluation parameters %s", training_args)

    return model_args, data_args, training_args


def create_eval_datasets(mode, data_args, tokenizer):
    eval_datasets = {}
    for task_id, task_name in enumerate(data_args.tasks):
        eval_datasets[task_name] = TaskDataset(
            task_name, task_id, data_args, tokenizer, mode=mode
        )
        if task_name == "mnli":
            # Loop to handle MNLI double evaluation (matched, mis-matched)
            eval_datasets["mnli-mm"] = TaskDataset(
                "mnli-mm", task_id, data_args, tokenizer, mode=mode
            )

    return eval_datasets


def main():
    model_args, data_args, training_args = parse_cmd_args()

    setup_logging(training_args)

    set_seed(training_args.seed)

    config = AutoConfig.from_pretrained(
        CaMtl.get_base_model(model_args.model_name_or_path),
    )

    model = CaMtl.from_pretrained(
        CaMtl.get_base_model(model_args.model_name_or_path),
        model_args,
        data_args,
        config=config)
    model.freeze_encoder_layers(model_args)

    logger.info(model)

    tokenizer = AutoTokenizer.from_pretrained(
        CaMtl.get_base_model(model_args.model_name_or_path),
    )

    logger.info("Training tasks: %s", ", ".join([t for t in data_args.tasks]))

    trainer = MultiTaskTrainer(
        tokenizer,
        data_args,
        model=model,
        args=training_args,
        train_dataset=MultiTaskDataset(data_args, tokenizer)
        if training_args.do_train
        else None,
        eval_datasets=create_eval_datasets(Split.dev, data_args, tokenizer)
        if training_args.do_eval or training_args.evaluate_during_training
        else None,
        test_datasets=create_eval_datasets(Split.test, data_args, tokenizer)
        if training_args.do_predict
        else None,
    )

    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path
            if os.path.isdir(model_args.model_name_or_path)
            else None
        )

    if training_args.do_eval:
        trainer.evaluate()

    if training_args.do_predict:
        trainer.predict()


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
