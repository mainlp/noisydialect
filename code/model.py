import copy
import time

import numpy as np
import torch
from torch.nn import Dropout, Linear
from torch.nn.utils import clip_grad_norm_
from transformers import BertConfig, BertModel, BertPreTrainedModel, \
    BertForTokenClassification
from transformers.models.bert.modeling_bert import BertForMaskedLM


class Model:

    def __init__(self, pretrained_model_name_or_path,
                 pos2idx, classifier_dropout,
                 print_model_structures=False
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
        print("Finetuning model")
        print(config)
        if print_model_structures:
            print(self.finetuning_model)
        # TODO do this more elegantly/low-level

    def continue_pretraining(self):
        pass  # TODO

    def finetune(self, device, iter_train, iter_dev, iter_test,
                 optimizer, scheduler, n_epochs, tokenizer):
        for epoch in range(n_epochs):
            print("============")
            print(f"Epoch {epoch + 1}/{n_epochs} started at " + self.now())
            self.train_classifier(device, iter_train, optimizer, scheduler, tokenizer) # TODO
            self.eval(device, iter_dev, "dev")
            self.eval(device, iter_test, "test")

    @staticmethod
    def now():
        return time.strftime("%Y/%m/%d %H:%M:%S", time.localtime())

    def train_classifier(self, device, iter_train, optimizer, scheduler,
                         tokenizer):
        self.finetuning_model.train()
        train_loss = 0

        n_batches = len(iter_train)
        for i, batch in enumerate(iter_train):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            self.finetuning_model.zero_grad()  # Clear old gradients

            # Forward pass
            # TODO: override! custom loss function taking into account the
            # subword tokens!!!
            out = self.finetuning_model(b_input_ids, b_input_mask,
                                        labels=b_labels)
            train_loss += out.loss.item()

            out.loss.backward()  # Calculate gradients
            clip_grad_norm_(self.finetuning_model.parameters(), 1.0)

            optimizer.step()  # Update model parameters
            scheduler.step()  # Update learning rate

            # words, x, is_heads, tags, y, seqlens = batch
            # _y = y
            # optimizer.zero_grad()
            # logits, y, _ = model(x, y)
            # # (n_sent, max_len, vocab_size) ->  (n_sent * max_len, vocab_size)
            # logits = logits.view(-1, logits.shape[-1])
            # # (n_sent, max_len) -> (n_sent * max_len,)
            # y = y.view(-1)  # (N*T,)

            # loss = criterion(logits, y)
            # loss.backward()

            # optimizer.step()

            if i % 1000 == 0:
                # Does everything look alright?
                ids = b_input_ids[0].detach().cpu().numpy()
                mask = b_input_mask[0].detach().cpu().numpy()
                tokens = tokenizer.convert_ids_to_tokens(
                    ids[mask.astype(bool)])
                gs_labels_enc = b_labels[0].detach().cpu().numpy()
                gs_labels = [self.finetuning_model.config.id2label[y]
                             for y in gs_labels_enc]
                logits = out.logits[0].detach().cpu().numpy()
                pred_labels_enc = np.argmax(logits, axis=1)
                pred_labels = [self.finetuning_model.config.id2label[y]
                               for y in pred_labels_enc]
                print("TOKEN\tGOLD\tPREDICTED")
                for token, gs_label, pred_label in zip(
                    tokens, gs_labels, pred_labels):
                    # chr(92) = \
                    print(f"{token}\t{gs_label}\t{pred_label}\t{'' if gs_label == pred_label else '/!' + chr(92)}")

            if i % 10 == 0:
                print(f"Batch {i:>2}/{n_batches} {self.now()} loss: {out.loss.item()}")
        print(f"Mean train loss for epoch: {train_loss / n_batches}")

    def eval_classifier(self, device, iter_test, eval_type):
        self.finetuning_model.eval()
        n_batches = len(iter_test)
        for i, batch in enumerate(iter_test):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            eval_loss = 0
            with torch.no_grad():
                # TODO: override! custom loss function taking into account the
                # subword tokens!!!
                out = self.finetuning_model(b_input_ids, b_input_mask,
                                            labels=b_labels)
                eval_loss += out.loss.item()

                # TODO additional metrics
        print(f"Mean {eval_type} loss for epoch: {eval_loss / n_batches}")

    def save_finetuned(self, path):
        self.finetuning_model.save_pretrained(path)

    def load_finetuned(self, path):
        self.finetuning_model = BertForTokenClassification.from_pretrained(
            path)
