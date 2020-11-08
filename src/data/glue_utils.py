import os
import time
import logging
from filelock import FileLock
from typing import List, Optional, Union

import numpy as np
import torch
from transformers import (
    InputExample,
    PreTrainedTokenizer,
    glue_processors,
    glue_output_modes,
    glue_compute_metrics,
    RobertaTokenizer,
    RobertaTokenizerFast,
    XLMRobertaTokenizer,
)

from src.utils.misc import MultiTaskInputFeatures, Split

logger = logging.getLogger(__name__)


glue_data_folder = {
    "cola": "CoLA",
    "mnli": "MNLI",
    "mnli-mm": "MNLI",
    "mrpc": "MRPC",
    "sst-2": "SST-2",
    "sts-b": "STS-B",
    "qqp": "QQP",
    "qnli": "QNLI",
    "rte": "RTE",
    "wnli": "WNLI",
}


def convert_glue_examples_to_multi_task_features(
    examples: List[InputExample],
    tokenizer: PreTrainedTokenizer,
    task_name,
    task_id,
    max_length: Optional[int] = None,
    label_list=None,
    output_mode=None,
):
    if max_length is None:
        max_length = tokenizer.max_len

    processor = glue_processors[task_name]()
    if label_list is None:
        label_list = processor.get_labels()
        logger.info("Using label list %s for task %s" % (label_list, task_name))
    if output_mode is None:
        output_mode = glue_output_modes[task_name]
        logger.info("Using output mode %s for task %s" % (output_mode, task_name))

    label_map = {label: i for i, label in enumerate(label_list)}

    def label_from_example(example: InputExample) -> Union[int, float, None]:
        if example.label is None:
            return None
        if output_mode == "classification":
            return label_map[example.label]
        elif output_mode == "regression":
            return float(example.label)
        raise KeyError(output_mode)

    labels = [label_from_example(example) for example in examples]

    batch_encoding = tokenizer.batch_encode_plus(
        [(example.text_a, example.text_b) for example in examples],
        max_length=max_length,
        pad_to_max_length=True,
    )

    features = []
    for i in range(len(examples)):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}

        if task_name.lower() == "mnli-mm":
            task_name = "mnli"

        feature = MultiTaskInputFeatures(
            **inputs,
            label=labels[i],
            task_id=task_id,
        )
        features.append(feature)

    for i, example in enumerate(examples[:5]):
        logger.info("*** Example ***")
        logger.info("guid: %s" % (example.guid))
        logger.info("features: %s" % features[i])

    return features


def load_glue_task_features(task_name, task_id, args, tokenizer, mode, limit_length):
    processor = glue_processors[task_name]()

    # Load data features from cache or dataset file
    task_data_dir = os.path.join(args.data_dir, glue_data_folder[task_name.lower()])
    cached_features_file = os.path.join(
        task_data_dir,
        "cached_{}_{}_{}_{}".format(
            mode.value,
            tokenizer.__class__.__name__,
            str(args.max_seq_length),
            task_name,
        ),
    )

    label_list = processor.get_labels()
    if task_name in ["mnli", "mnli-mm"] and tokenizer.__class__ in (
        RobertaTokenizer,
        RobertaTokenizerFast,
        XLMRobertaTokenizer,
    ):
        # HACK(label indices are swapped in RoBERTa pretrained model)
        label_list[1], label_list[2] = label_list[2], label_list[1]

    # Make sure only the first process in distributed training processes the dataset,
    # and the others will use the cache.
    lock_path = cached_features_file + ".lock"
    with FileLock(lock_path):

        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            start = time.time()
            features = torch.load(cached_features_file)
            logger.info(
                f"Loading features from cached file {cached_features_file} [took %.3f s]",
                time.time() - start,
            )
        else:
            logger.info(f"Creating features from dataset file at {task_data_dir}")

            if mode == Split.dev:
                examples = processor.get_dev_examples(task_data_dir)
            elif mode == Split.test:
                examples = processor.get_test_examples(task_data_dir)
            else:
                examples = processor.get_train_examples(task_data_dir)
            if limit_length is not None:
                examples = examples[:limit_length]

            features = convert_glue_examples_to_multi_task_features(
                examples,
                tokenizer,
                task_name,
                task_id,
                max_length=args.max_seq_length,
                label_list=label_list,
                output_mode=glue_output_modes[task_name],
            )
            start = time.time()
            torch.save(features, cached_features_file)
            logger.info(
                "Saving features into cached file %s [took %.3f s]",
                cached_features_file,
                time.time() - start,
            )

    return features, label_list


def compute_glue_metrics(task_name, p):
    output_mode = glue_output_modes[task_name]

    if output_mode == "classification":
        preds = np.argmax(p.predictions, axis=1)
    elif output_mode == "regression":
        preds = np.squeeze(p.predictions)
    return glue_compute_metrics(task_name, preds, p.label_ids)
