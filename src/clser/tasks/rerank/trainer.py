#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2026-05-27
# @Author : GJason
# @File : trainer.py

import torch
import torch.nn.functional as F
from transformers import Trainer


class RerankCollator:

    def __init__(self, tokenizer, pad_to_multiple_of=None):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features):
        flat = [
            {"input_ids": ids, "attention_mask": mask}
            for feature in features
            for ids, mask in zip(feature["input_ids"], feature["attention_mask"])
        ]
        return self.tokenizer.pad(
            flat,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )


class RerankTrainer(Trainer):

    def __init__(self, *args, model_type, loss_type, temperature, train_group_size, eval_group_size,
                 yes_token_id=None, no_token_id=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_type = model_type
        self.loss_type = loss_type
        self.temperature = temperature
        self.train_group_size = train_group_size
        self.eval_group_size = eval_group_size
        self.yes_token_id = yes_token_id
        self.no_token_id = no_token_id

    def _scores(self, model, inputs):
        if self.model_type == "cross_encoder":
            return model(**inputs).logits.squeeze(-1)
        logits = model(**inputs, logits_to_keep=1).logits[:, -1, :]
        return logits[:, self.yes_token_id] - logits[:, self.no_token_id]

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        group_size = self.train_group_size if model.training else self.eval_group_size
        scores = self._scores(model, inputs).float().view(-1, group_size)

        if self.loss_type == "listwise":
            target = torch.zeros(scores.size(0), dtype=torch.long, device=scores.device)
            loss = F.cross_entropy(scores / self.temperature, target)
        else:
            target = torch.zeros_like(scores)
            target[:, 0] = 1.0
            loss = F.binary_cross_entropy_with_logits(scores, target)

        return (loss, {"scores": scores}) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
        if prediction_loss_only:
            return loss.detach(), None, None
        scores = outputs["scores"].detach()
        labels = torch.zeros(scores.size(0), dtype=torch.long, device=scores.device)
        return loss.detach(), scores, labels
