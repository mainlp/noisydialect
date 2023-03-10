from C_data import DUMMY_POS

import copy
import sys

import numpy as np
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score, f1_score
import torch
from torch.nn import CrossEntropyLoss
from transformers import BertForTokenClassification, BertForMaskedLM, \
    BertForPreTraining, RobertaForMaskedLM, RobertaForTokenClassification, \
    XLMRobertaForMaskedLM, XLMRobertaForTokenClassification


def filter_predictions(y_pred, y_true, dummy_idx):
    mask = np.nonzero(np.where(y_true == dummy_idx, 0, 1))
    gold_filtered = np.asarray(y_true)[mask]
    pred_filtered = np.asarray(y_pred)[mask]
    return gold_filtered, pred_filtered


def score(y_pred, y_true, dummy_idx):
    # Simply using the mask as sample_weight is not sufficient for
    # calculating the F1 score as the dummy class (which we want to
    # completely ignore) would still be included in the calculation
    # of the class-based averages. This would produce F1 macro scores
    # that are slightly too low.
    gold_filtered, pred_filtered = filter_predictions(y_pred, y_true,
                                                      dummy_idx)
    acc = accuracy_score(gold_filtered, pred_filtered)
    f1_macro = f1_score(gold_filtered, pred_filtered,
                        average='macro', zero_division=0)
    return acc, f1_macro


class Classifier(pl.LightningModule):
    def __init__(self, pretrained_model_name_or_path, plm_type,
                 pos2idx, classifier_dropout, learning_rate,
                 subtok2weight=None,
                 val_data_names=["dev"], test_data_names=["test"],
                 print_model_structures=False, print_config=True,
                 ):
        super().__init__()
        """
        According to the DBMDZ model configuration, their model is of type
        BertForMaskedLM rather than BertForPretraining, which also
        encompasses next sentence prediction
        https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-dbmdz-cased-config.json
        )
        TODO This should be modified when/if I add a continued pretraining option
        """
        if plm_type == "BertForMaskedLM":
            self.pretraining_model = BertForMaskedLM.from_pretrained(
                pretrained_model_name_or_path)
        elif plm_type == "BertForPreTraining":
            self.pretraining_model = BertForPreTraining.from_pretrained(
                pretrained_model_name_or_path)
        elif plm_type == "RobertaForMaskedLM":
            self.pretraining_model = RobertaForMaskedLM.from_pretrained(
                pretrained_model_name_or_path)
        elif plm_type == "XLMRobertaForMaskedLM":
            self.pretraining_model = XLMRobertaForMaskedLM.from_pretrained(
                pretrained_model_name_or_path)
        else:
            print("Could not recognize PLM type: " + plm_type)
            print("Quitting.")
            sys.exit(1)
        if print_model_structures:
            print("Pretraining model")
            print(self.pretraining_model)
        config = copy.deepcopy(self.pretraining_model.config)
        config.__setattr__("label2id", pos2idx)
        config.__setattr__("id2label", {pos2idx[pos]: pos for pos in pos2idx})
        if plm_type in ("BertForMaskedLM", "BertForPreTraining"):
            config.__setattr__("architectures",
                               ["BertForTokenClassification"])
            self.finetuning_model = BertForTokenClassification(config)
        elif plm_type == "RobertaForMaskedLM":
            config.__setattr__("architectures",
                               ["RobertaForTokenClassification"])
            self.finetuning_model = RobertaForTokenClassification(config)
        elif plm_type == "XLMRobertaForMaskedLM":
            config.__setattr__("architectures",
                               ["XLMRobertaForTokenClassification"])
            self.finetuning_model = XLMRobertaForTokenClassification(config)
        if plm_type in ("BertForMaskedLM", "BertForPreTraining"):
            self.finetuning_model.bert = self.pretraining_model.bert
            self.is_roberta = False
        else:
            self.finetuning_model.roberta = self.pretraining_model.roberta
            self.is_roberta = True
        if print_config or print_model_structures:
            print("Finetuning model")
            if print_config:
                print(config)
            if print_model_structures:
                print(self.finetuning_model)
        self.dummy_idx = pos2idx[DUMMY_POS]
        self.learning_rate = learning_rate
        # Monitoring performance:
        self.val_data_names = val_data_names
        # nested: validation set -> epoch
        self.val_preds = [[] for _ in val_data_names]
        self.val_gold = [[] for _ in val_data_names]
        self.set_test_names(test_data_names)
        self.epoch = 0

    def set_test_names(self, test_data_names):
        self.test_data_names = test_data_names
        self.test_preds = [[] for _ in test_data_names]
        self.test_gold = [[] for _ in test_data_names]

    def forward(self, input_ids, attention_mask):
        if self.is_roberta:
            outputs = self.finetuning_model.roberta(input_ids, attention_mask)
        else:
            outputs = self.finetuning_model.bert(input_ids, attention_mask)
        sequence_output = outputs[0]
        sequence_output = self.finetuning_model.dropout(sequence_output)
        logits = self.finetuning_model.classifier(sequence_output)
        return logits

    def loss(self, logits, labels):
        return CrossEntropyLoss(ignore_index=self.dummy_idx)(
            logits.view(-1, self.finetuning_model.num_labels),
            labels.view(-1))

    def detach_and_score(self, logits, labels):
        y_pred = np.argmax(logits.detach().clone().cpu().numpy(),
                           axis=2).flatten()
        y_true = labels.detach().clone().cpu().numpy().flatten()
        acc, f1_macro = self.score(y_pred, y_true)
        return acc, f1_macro, y_pred, y_true

    def score(self, y_pred, y_true):
        return score(y_pred, y_true, self.dummy_idx)

    def on_train_start(self):
        self.epoch = 0

    def on_train_epoch_start(self):
        self.epoch += 1

    def training_step(self, train_batch, batch_idx):
        x, mask, y = train_batch
        logits = self.forward(x, mask)
        loss = self.loss(logits, y)
        acc, f1_macro, _, _ = self.detach_and_score(logits, y)
        self.log_dict({"train_loss_batch": loss, "train_acc_batch": acc,
                       "train_f1_batch": f1_macro}, prog_bar=True)
        return loss

    def on_validation_epoch_start(self):
        self.cur_val_preds = [[] for _ in self.val_data_names]
        self.cur_val_gold = [[] for _ in self.val_data_names]

    def validation_step(self, val_batch, batch_idx, dataloader_idx=0):
        x, mask, y = val_batch
        logits = self.forward(x, mask)
        loss = self.loss(logits, y)
        acc, f1_macro, y_pred, y_true = self.detach_and_score(logits, y)
        self.cur_val_preds[dataloader_idx] += y_pred.tolist()
        self.cur_val_gold[dataloader_idx] += y_true.tolist()
        val_name = self.val_data_names[dataloader_idx]
        self.log_dict({val_name + "_loss_batch": loss,
                       val_name + "_acc_batch": acc,
                       val_name + "_f1_batch": f1_macro}, prog_bar=True)

    def on_validation_epoch_end(self):
        for i, val_name in enumerate(self.val_data_names):
            print(i, val_name)
            cur_preds = np.asarray(self.cur_val_preds[i])
            cur_gold = np.asarray(self.cur_val_gold[i])
            acc, f1_macro = self.score(cur_preds, cur_gold)
            print("acc", acc)
            print("f1 macro", f1_macro)
            self.val_preds[i].append(cur_preds)
            self.val_gold[i].append(cur_gold)
            self.log_dict({f"{val_name}_acc_epoch{self.epoch}": acc,
                           f"{val_name}_f1_epoch{self.epoch}": f1_macro})

    def on_test_start(self):
        self.set_test_names(self.test_data_names)

    def on_test_epoch_start(self):
        self.cur_test_preds = [[] for _ in self.test_data_names]
        self.cur_test_gold = [[] for _ in self.test_data_names]

    def test_step(self, test_batch, batch_idx, dataloader_idx=0):
        x, mask, y = test_batch
        logits = self.forward(x, mask)
        acc, f1_macro, y_pred, y_true = self.detach_and_score(logits, y)
        self.cur_test_preds[dataloader_idx] += y_pred.tolist()
        self.cur_test_gold[dataloader_idx] += y_true.tolist()
        test_name = self.test_data_names[dataloader_idx]
        self.log_dict({test_name + "_acc_batch": acc,
                       test_name + "_f1_batch": f1_macro}, prog_bar=True)

    def on_test_epoch_end(self):
        for i, test_name in enumerate(self.test_data_names):
            print(i, test_name)
            cur_preds = np.asarray(self.cur_test_preds[i])
            cur_gold = np.asarray(self.cur_test_gold[i])
            acc, f1_macro = self.score(cur_preds, cur_gold)
            print("acc", acc)
            print("f1 macro", f1_macro)
            self.test_preds[i] = cur_preds
            self.test_gold[i] = cur_gold
            self.log_dict({f"{test_name}_acc_epoch{self.epoch}": acc,
                           f"{test_name}_f1_epoch{self.epoch}": f1_macro})

    def test_results(self):
        return [self.score(self.test_preds[i], self.test_gold[i])
                for i in range(len(self.test_data_names))]

    def get_test_predictions(self):
        return self.test_preds, self.test_gold

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch[0], batch[1])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
