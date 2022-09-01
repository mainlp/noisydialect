from data import DUMMY_POS

import copy
import time

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.nn.utils import clip_grad_norm_
from transformers import BertConfig, BertModel, BertPreTrainedModel, \
    BertForTokenClassification
from transformers.models.bert.modeling_bert import BertForMaskedLM


class Model:

    def __init__(self, pretrained_model_name_or_path,
                 pos2idx, classifier_dropout,
                 print_model_structures=False, print_config=True,
                 ):
        # self.mode = "PRETRAINING"  # ["PRETRAINING", "FINETUNING", "EVAL"]
        """
        According to the DBMDZ model configuration, their model is of type
        BertForMaskedLM rather than BertForPretraining, which also
        encompasses next sentence prediction
        https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-dbmdz-cased-config.json
        )
        """
        self.pretraining_model = BertForMaskedLM.from_pretrained(
            pretrained_model_name_or_path)
        if print_model_structures:
            print("Pretraining model")
            print(self.pretraining_model)
        config = copy.deepcopy(self.pretraining_model.config)
        config.__setattr__("label2id", pos2idx)
        config.__setattr__("id2label", {pos2idx[pos]: pos for pos in pos2idx})
        config.__setattr__("architectures", ["BertForTokenClassification"])
        self.finetuning_model = BertForTokenClassification(config)
        self.finetuning_model.bert = self.pretraining_model.bert
        if print_config or print_model_structures:
            print("Finetuning model")
            if print_config:
                print(config)
            if print_model_structures:
                print(self.finetuning_model)
        # TODO do this more elegantly/low-level

    def continue_pretraining(self):
        pass  # TODO

    def finetune(self, device, iter_train, iter_dev, iter_test,
                 optimizer, scheduler, n_epochs, tokenizer, dummy_idx):
        for epoch in range(n_epochs):
            print("============")
            print(f"Epoch {epoch + 1}/{n_epochs} started at " + self.now())
            self.train_classifier(device, iter_train, optimizer, scheduler,
                                  tokenizer, dummy_idx)
            self.eval(device, iter_dev, dummy_idx, "dev")
            self.eval(device, iter_test, dummy_idx, "test")

    @staticmethod
    def now():
        return time.strftime("%Y/%m/%d %H:%M:%S", time.localtime())

    def train_classifier(self, device, iter_train, optimizer, scheduler,
                         tokenizer, dummy_idx, sanity_mod=1000):
        self.finetuning_model.train()
        n_batches = len(iter_train)
        train_loss = 0
        for i, batch in enumerate(iter_train):
            self.finetuning_model.zero_grad()  # Clear old gradients

            loss, logits = self.forward_finetuning(
                input_ids=batch[0].to(device),
                attention_mask=batch[2].to(device),
                labels=batch[1].to(device),
                dummy_idx=dummy_idx)
            train_loss += loss.item()
            loss.backward()  # Calculate gradients
            clip_grad_norm_(self.finetuning_model.parameters(), 1.0)

            optimizer.step()  # Update model parameters
            scheduler.step()  # Update learning rate

            if i % sanity_mod == 0:
                self.sanity_check(ids=batch[0], labels=batch[1],
                                  logits=logits.detach().cpu().numpy(),
                                  mask=batch[2], tokenizer=tokenizer)

            if i % 10 == 0:
                print(f"Batch {i:>2}/{n_batches} {self.now()} loss: {out.loss.item()}")

        print(f"Mean train loss for epoch: {train_loss / n_batches}")

    def eval_classifier(self, device, iter_test, dummy_idx, eval_type):
        self.finetuning_model.eval()
        n_batches = len(iter_test)
        eval_loss = 0
        for i, batch in enumerate(iter_test):
            with torch.no_grad():
                loss, logits = self.forward_finetuning(
                    input_ids=batch[0].to(device),
                    attention_mask=batch[2].to(device),
                    labels=batch[1].to(device),
                    dummy_idx=dummy_idx)
                eval_loss += loss.item()

                # TODO additional metrics
        print(f"Mean {eval_type} loss for epoch: {eval_loss / n_batches}")

    def forward_finetuning(self, input_ids, attention_mask, labels,
                           dummy_idx):
        # Forward pass, as for BertForTokenClassification,
        # but ignoring the [DUMMY] tags when computing the
        # CrossEntropyLoss
        outputs = self.finetuning_model.bert(input_ids, attention_mask)
        sequence_output = outputs[0]
        sequence_output = self.finetuning_model.dropout(sequence_output)
        logits = self.finetuning_model.classifier(sequence_output)
        loss = CrossEntropyLoss(ignore_index=dummy_idx)(
            logits.view(-1, self.finetuning_model.num_labels),
            labels.view(-1))
        return loss, logits

    def sanity_check(self, ids, labels, logits, mask, tokenizer):
        tokens = tokenizer.convert_ids_to_tokens(ids[mask.astype(bool)])
        gs_labels = [self.finetuning_model.config.id2label[y] for y in labels]
        pred_labels_enc = np.argmax(logits, axis=1)
        pred_labels = [self.finetuning_model.config.id2label[y]
                       for y in pred_labels_enc]
        print("TOKEN\tGOLD\tPREDICTED\tWRONG/IGNORED")
        for token, gs_label, pred_label in zip(
                tokens, gs_labels, pred_labels):
            print(f"{token}\t{gs_label}\t{pred_label}\t", end="")
            if gs_label == DUMMY_POS:
                print("--")
            elif gs_label != pred_label:
                print("/!\\")
            else:
                print()

    def save_finetuned(self, path):
        self.finetuning_model.save_pretrained(path)

    def load_finetuned(self, path):
        self.finetuning_model = BertForTokenClassification.from_pretrained(
            path)
