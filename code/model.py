from data import DUMMY_POS

import copy
import time

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
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
            self.eval_classifier(device, iter_dev, dummy_idx, "dev")
            if iter_test:
                self.eval_classifier(device, iter_test, dummy_idx, "test")

    def predict(self, device, iter_test, dummy_idx, out_file):
        self.eval_classifier(device, iter_test, dummy_idx, "test",
                             out_file=out_file)

    @staticmethod
    def now():
        return time.strftime("%Y/%m/%d %H:%M:%S", time.localtime())

    def train_classifier(self, device, iter_train, optimizer, scheduler,
                         tokenizer, dummy_idx, sanity_mod=1000):
        self.finetuning_model.train()
        n_batches = len(iter_train)
        train_loss = 0
        y_true = np.empty(0, dtype=int)
        y_pred = np.empty(0, dtype=int)
        for i, batch in enumerate(iter_train):
            self.finetuning_model.zero_grad()  # Clear old gradients

            loss, logits = self.forward_finetuning(
                input_ids=batch[0].to(device),
                attention_mask=batch[1].to(device),
                labels=batch[2].to(device),
                dummy_idx=dummy_idx)
            train_loss += loss.item()
            loss.backward()  # Calculate gradients
            clip_grad_norm_(self.finetuning_model.parameters(), 1.0)

            optimizer.step()  # Update model parameters
            scheduler.step()  # Update learning rate

            b_y_gs = batch[2].detach().cpu().numpy()
            b_logits = logits.detach().cpu().numpy()
            y_true = np.append(y_true, b_y_gs)
            y_pred = np.append(y_pred, np.argmax(b_logits, axis=1))

            if i % 10 == 0:
                print(f"Batch {i:>2}/{n_batches} {self.now()}", end="\t")
                print(f"loss: {loss.item():.4f}", end="\t")

            if i % sanity_mod == 0:
                b_x = batch[0].detach().cpu().numpy()
                b_mask = batch[1].detach().cpu().numpy()
                self.sanity_check(b_x[0], b_y_gs[0], b_logits[0], b_mask[0],
                                  tokenizer)

        print(f"Mean train loss for epoch: {train_loss / n_batches:.4f}")
        self.print_scores(y_true, y_pred, dummy_idx)

    def eval_classifier(self, device, iter_test, dummy_idx, eval_type,
                        tokenizer, sanity_mod=1000, out_file=None):
        self.finetuning_model.eval()
        n_batches = len(iter_test)
        eval_loss = 0
        y_true = np.empty(0, dtype=int)
        y_pred = np.empty(0, dtype=int)
        x = np.empty(0, dtype=np.int64)
        for i, batch in enumerate(iter_test):
            with torch.no_grad():
                loss, logits = self.forward_finetuning(
                    input_ids=batch[0].to(device),
                    attention_mask=batch[1].to(device),
                    labels=batch[2].to(device) if len(batch) > 2 else None,
                    dummy_idx=dummy_idx)
                b_x = batch[0].detach().cpu().numpy()
                b_mask = batch[1].detach().cpu().numpy()
                if len(batch) > 2:
                    b_y_gs = batch[2].detach().cpu().numpy()
                else:
                    b_y_gs = None
                b_logits = logits.detach().cpu().numpy()
                if len(batch) > 2:
                    eval_loss += loss.item()
                    y_true = np.append(y_true, b_y_gs)
                x = np.append(x, b_x)
                y_pred = np.append(y_pred, np.argmax(b_logits, axis=1))
                if i % sanity_mod == 0:
                    self.sanity_check(b_x[0], b_y_gs[0], b_logits[0],
                                      b_mask[0], tokenizer)

        if eval_loss > 0.000:
            print(f"Mean {eval_type} loss: {eval_loss / n_batches}")
            self.print_scores(y_true, y_pred, dummy_idx)

        if out_file:
            with open(out_file, "w", encoding='utf8') as f:
                if eval_loss > 0.000:
                    for tok_id, gs_id, pred_id in zip(x, y_true, y_pred):
                        f.write(tokenizer.convert_ids_to_tokens(int(tok_id)))
                        f.write("\t")
                        f.write(self.finetuning_model.config.id2label[gs_id])
                        f.write("\t")
                        f.write(self.finetuning_model.config.id2label[pred_id])
                        f.write("\n")
                else:
                    for tok_id, pred_id in zip(x, y_pred):
                        f.write(tokenizer.convert_ids_to_tokens(int(tok_id)))
                        f.write("\t")
                        f.write(self.finetuning_model.config.id2label[pred_id])
                        f.write("\n")

    def forward_finetuning(self, input_ids, attention_mask, labels,
                           dummy_idx):
        # Forward pass, as for BertForTokenClassification,
        # but ignoring the [DUMMY] tags when computing the
        # CrossEntropyLoss
        outputs = self.finetuning_model.bert(input_ids, attention_mask)
        sequence_output = outputs[0]
        sequence_output = self.finetuning_model.dropout(sequence_output)
        logits = self.finetuning_model.classifier(sequence_output)
        if labels is not None:
            loss = CrossEntropyLoss(ignore_index=dummy_idx)(
                logits.view(-1, self.finetuning_model.num_labels),
                labels.view(-1))
        else:
            loss = None
        return loss, logits

    def sanity_check(self, ids, labels, logits, mask, tokenizer):
        # Assumes all IDs belong just to one sentence
        tokens, gs_labels, pred_labels = self.decode(
            ids, labels, logits, mask, tokenizer)
        print("TOKEN\tGOLD\tPREDICTED\tWRONG/IGNORED")
        for token, gs_label, pred_label in zip(
                tokens, gs_labels, pred_labels):
            print(f"{token}\t{gs_label}\t{pred_label}\t", end="")
            if gs_label == DUMMY_POS:
                print("--")
            elif labels is not None and gs_label != pred_label:
                print("/!\\")
            else:
                print()

    def print_scores(self, y_true, y_pred, dummy_idx):
        mask = np.where(y_true == dummy_idx, 0, 1)
        acc = accuracy_score(y_true, y_pred, sample_weight=mask)
        f1_micro = f1_score(y_true, y_pred, sample_weight=mask,
                            average='micro')
        f1_macro = f1_score(y_true, y_pred, sample_weight=mask,
                            average='macro')
        print(f"Accuracy: {acc:.2f}")
        print(f"F1 micro: {f1_micro:.2f}")
        print(f"F1 macro: {f1_macro:.2f}")

    def decode(self, tok_ids, label_ids, logits, mask, tokenizer):
        # Assumes all IDs belong just to one sentence
        tokens = tokenizer.convert_ids_to_tokens(tok_ids[mask.astype(bool)])
        if label_ids is None:
            gs_labels = ["?" for _ in mask]
        else:
            # print(label_ids)
            gs_labels = [self.finetuning_model.config.id2label[y]
                         for y in label_ids]
        pred_labels = [self.finetuning_model.config.id2label[y]
                       for y in np.argmax(logits, axis=1)]
        return tokens, gs_labels, pred_labels

    def save_finetuned(self, path):
        self.finetuning_model.save_pretrained(path)

    def load_finetuned(self, path):
        self.finetuning_model = BertForTokenClassification.from_pretrained(
            path)
