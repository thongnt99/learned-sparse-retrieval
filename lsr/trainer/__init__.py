import torch
import os
import transformers
import logging
from lsr.models import DualSparseEncoder
from lsr.models import DualSparseEncoder
from collections import defaultdict

logger = logging.getLogger(__name__)

LOSS_NAME = "loss.pt"


class HFTrainer(transformers.trainer.Trainer):
    """Customized Trainer from Huggingface's Trainer"""

    def __init__(
        self, *args, loss=None, train_only_bias_and_layer_norm=False, **kwargs
    ) -> None:
        super(HFTrainer, self).__init__(*args, **kwargs)
        self.loss = loss
        self.customed_log = defaultdict(lambda: 0.0)
        self.tokenizer = self.data_collator.tokenizer
        self.model = torch.compile(self.model)

    def _maybe_log_save_evaluate(
        self, tr_loss, model, trial, epoch, ignore_keys_for_eval
    ):
        if self.control.should_log:
            log = {}
            for metric in self.customed_log:
                log[metric] = (
                    self._nested_gather(self.customed_log[metric]).mean().item()
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
            self.loss.load_state_dict(os.path.join(checkpoint, LOSS_NAME))

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
            DualSparseEncoder.from_pretrained(resume_from_checkpoint).state_dict()
        )
