from typing import Dict, List
import torch
import os
from torch.utils.data import Dataset
import transformers
import logging
from lsr.models import DualSparseEncoder
from lsr.models import DualSparseEncoder
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
import json
import ir_measures
import shutil
import math
import ir_measures
from ir_measures import *
import time
from transformers.trainer_utils import EvalLoopOutput, speed_metrics
from transformers.trainer_utils import PredictionOutput

logger = logging.getLogger(__name__)
LOSS_NAME = "loss.pt"


class HFTrainer(transformers.trainer.Trainer):
    """Customized Trainer from Huggingface's Trainer"""

    def __init__(self, *args, index=None, loss=None, **kwargs) -> None:
        super(HFTrainer, self).__init__(*args, **kwargs)
        self.loss = loss
        self.customed_log = defaultdict(lambda: 0.0)
        self.tokenizer = self.data_collator.tokenizer
        self.index = index

    def _maybe_log_save_evaluate(
        self, tr_loss, model, trial, epoch, ignore_keys_for_eval
    ):
        if self.control.should_log:
            log = {}
            for metric in self.customed_log:
                log[metric] = (
                    self._nested_gather(
                        self.customed_log[metric]).mean().item()
                )
                log[metric] = round(
                    (
                        log[metric]
                        / (self.state.global_step - self._globalstep_last_logged)
                        / self.args.gradient_accumulation_steps
                    ),
                    4,
                )
            self.log(log)
            for metric in self.customed_log:
                self.customed_log[metric] -= self.customed_log[metric]
            self.control.should_log = True
        super()._maybe_log_save_evaluate(
            tr_loss, model, trial, epoch, ignore_keys_for_eval
        )

    def _load_optimizer_and_scheduler(self, checkpoint):
        super()._load_optimizer_and_scheduler(checkpoint)
        if checkpoint is None:
            return
        if os.path.join(checkpoint, LOSS_NAME):
            self.loss.load_state_dict(torch.load(
                os.path.join(checkpoint, LOSS_NAME)))

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute loss
        """
        loss_output, q_reg, d_reg, to_log = model(self.loss, **inputs)
        for log_metric in to_log:
            self.customed_log[log_metric] += to_log[log_metric]
        return loss_output + q_reg + d_reg

    def save_model(self, model_dir=None, _internal_call=False):
        """Save model checkpoint"""
        logger.info("Saving model checkpoint to %s", model_dir)
        if model_dir is None:
            model_dir = os.path.join(self.args.output_dir, "model")
        self.model.save_pretrained(model_dir)
        if self.tokenizer is not None:
            tokenizer_path = os.path.join(model_dir, "tokenizer")
            self.tokenizer.save_pretrained(tokenizer_path)
        loss_path = os.path.join(model_dir, LOSS_NAME)
        logger.info("Saving loss' state to %s", loss_path)
        torch.save(self.loss.state_dict(), loss_path)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        """Load from a checkpoint to continue traning"""
        # Load model from checkpoint
        logger.info("Loading model's weight from %s", resume_from_checkpoint)
        self.model.load_state_dict(
            DualSparseEncoder.from_pretrained(
                resume_from_checkpoint).state_dict()
        )

    def evaluation_loop(self, query_dataloader, doc_dataloader, qrels, run_file):
        eval_dir = Path(self.args.output_dir)/"inference_float"
        if not eval_dir.is_dir():
            eval_dir.mkdir()
        queries_path = eval_dir/"queries.jsonl"
        docs_dir = eval_dir/"docs"
        index_dir = eval_dir/"index"
        run_path = eval_dir/run_file
        if queries_path.is_file():
            os.remove(queries_path)
        if docs_dir.is_dir():
            shutil.rmtree(docs_dir)
        if index_dir.is_dir():
            shutil.rmtree(index_dir)
        self.model.eval()
        qid2rep = defaultdict(dict)
        with open(queries_path, "w") as fquery:
            for batch_queries in tqdm(query_dataloader, desc=f"Encoding queries and saving raw weights to {queries_path}"):
                batch_query_ids = batch_queries["query_ids"]
                with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
                    batch_query_tok_ids, batch_query_tok_weights, _ = self.model.encode_queries(
                        **self._prepare_inputs(batch_queries["queries"]), to_dense=False).to_sparse()
                    batch_query_tok_ids = batch_query_tok_ids.to(
                        "cpu").tolist()
                    batch_query_tok_weights = batch_query_tok_weights.to(
                        "cpu").tolist()
                for qid, tokid_list, tokweight_list in zip(batch_query_ids, batch_query_tok_ids, batch_query_tok_weights):
                    w2w = {str(tok_id): tok_weight for tok_id,
                           tok_weight in zip(tokid_list, tokweight_list) if tok_weight > 0}
                    row = {"id":  qid, "vector": w2w}
                    row = json.dumps(row)
                    fquery.write(row+"\n")
                    qid2rep[qid] = w2w
        num_partition = 60
        iter_per_patition = math.ceil(len(doc_dataloader)/num_partition)
        doc_iter = iter(doc_dataloader)
        for part_idx in tqdm(list(range(num_partition)), desc="Encoding documents and saving raw weights to {docs_dir}"):
            docs_path = docs_dir/f"docs_{part_idx}.jsonl"
            with open(docs_path, "w") as fdoc:
                for _ in range(iter_per_patition):
                    try:
                        batch_docs = next(doc_iter)
                        doc_ids = batch_docs["doc_ids"]
                        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
                            doc_tok_ids, doc_tok_weights, _ = self.model.encode_docs(
                                **self._prepare_inputs(batch_docs["doc_groups"]), to_dense=False).to_sparse()
                            doc_tok_ids = doc_tok_ids.to(
                                "cpu").tolist()
                            doc_tok_weights = doc_tok_weights.to(
                                "cpu").tolist()
                        for doc_id, tokid_list, tokweight_list in zip(doc_ids, doc_tok_ids, doc_tok_weights):
                            tokid_list = [str(tokid) for tokid in tokid_list]
                            vector = {tokid: tokweight for tokid,
                                      tokweight in zip(tokid_list, tokweight_list) if tokweight > 0}
                            doc_json = {"id": doc_id, "vector": vector}
                            fdoc.write(json.dumps(doc_json)+"\n")
                    except:
                        pass
        self.index.index(docs_dir, index_dir)
        run = self.index.retrieve(queries_path, run_path)
        if qrels is None:
            metrics = {}
        else:
            metrics = ir_measures.calc_aggregate(
                [MRR@10, R@5, R@10, R@100, R@1000, R@100000, NDCG@10, NDCG@20, MAP], qrels, run)
        self.model.train()
        return run, metrics

    def evaluate(
            self,
            eval_dataset,
            ignore_keys,
            metric_key_prefix: str = "eval"):
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (`Dataset`, *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns
                not accepted by the `model.forward()` method are automatically removed. It must implement the `__len__`
                method.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        doc_data_loader = self.get_eval_dataloader(
            eval_dataset.docs)
        query_dataloader = self.get_eval_dataloader(
            eval_dataset.queries)
        start_time = time.time()
        run, metrics = self.evaluation_loop(
            query_dataloader, doc_data_loader, eval_dataset.qrels, "eval_run.trec")
        metrics = {
            f"{metric_key_prefix}_{str(m)}": metrics[m] for m in metrics}
        metrics.update(speed_metrics(metric_key_prefix,
                                     start_time))
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, metrics)
        self._memory_tracker.stop_and_update_metrics(metrics)
        return metrics

    def predict(
        self, test_dataset: Dataset, ignore_keys=None, metric_key_prefix: str = "test"
    ):
        """
        Run prediction and returns predictions and potential metrics.

        Depending on the dataset and your use case, your test dataset may contain labels. In that case, this method
        will also return metrics, like in `evaluate()`.

        Args:
            test_dataset (`Dataset`):
                Dataset to run the predictions on. If it is an `datasets.Dataset`, columns not accepted by the
                `model.forward()` method are automatically removed. Has to implement the method `__len__`
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"test"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "test_bleu" if the prefix is "test" (default)

        <Tip>

        If your predictions or labels have different sequence length (for instance because you're doing dynamic padding
        in a token classification task) the predictions will be padded (on the right) to allow for concatenation into
        one array. The padding index is -100.

        </Tip>

        Returns: *NamedTuple* A namedtuple with the following keys:

            - predictions (`np.ndarray`): The predictions on `test_dataset`.
            - label_ids (`np.ndarray`, *optional*): The labels (if the dataset contained some).
            - metrics (`Dict[str, float]`, *optional*): The potential dictionary of metrics (if the dataset contained
              labels).
        """
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()
        if test_dataset is None:
            test_dataset = self.test_dataset
        doc_data_loader = self.get_test_dataloader(
            test_dataset.docs)
        query_dataloader = self.get_test_dataloader(
            test_dataset.queries)
        print(
            f"Prediction queries: {len(test_dataset.queries)} with {len(query_dataloader)} batches")
        print(
            f"Prediction docs: {len(test_dataset.docs)} with {len(doc_data_loader)} batches")
        start_time = time.time()
        run_path = self.args.output_dir + "/test_run.trec"
        test_run, metrics = self.evaluation_loop(
            query_dataloader, doc_data_loader, test_dataset.qrels, run_path=run_path)
        metrics = {
            f"{metric_key_prefix}_{str(m)}": metrics[m] for m in metrics}
        metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
            )
        )
        self.log(metrics)
        self.control = self.callback_handler.on_predict(
            self.args, self.state, self.control, metrics)
        self._memory_tracker.stop_and_update_metrics(metrics)
        result_path = self.args.output_dir + "/test_result.json"
        logger.info(f"Saving test metrics to {result_path}")
        json.dump(metrics, open(result_path, "w"))
        return PredictionOutput(predictions=test_run,  label_ids=None, metrics=metrics)
