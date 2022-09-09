from data import DUMMY_POS

import copy

import numpy as np
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score, f1_score
import torch
from torch.nn import CrossEntropyLoss
from transformers import BertForTokenClassification
from transformers.models.bert.modeling_bert import BertForMaskedLM


class Classifier(pl.LightningModule):
    def __init__(self, pretrained_model_name_or_path,
                 pos2idx, classifier_dropout,
                 print_model_structures=False, print_config=True,
                 ):
        super().__init__()
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
        self.dummy_idx = pos2idx[DUMMY_POS]
        self.learning_rate = learning_rate

    def forward(self, input_ids, attention_mask):
        outputs = self.finetuning_model.bert(input_ids, attention_mask)
        sequence_output = outputs[0]
        sequence_output = self.finetuning_model.dropout(sequence_output)
        logits = self.finetuning_model.classifier(sequence_output)
        return logits

    def loss(self, logits, labels):
        return CrossEntropyLoss(ignore_index=self.dummy_idx)(
            logits.view(-1, self.finetuning_model.num_labels),
            labels.view(-1))

    def score(self, logits, labels):
        y_true = labels.detach().cpu().numpy().flatten()
        y_pred = np.argmax(logits.detach().cpu().numpy(), axis=2).flatten()
        mask = np.where(y_true == self.dummy_idx, 0, 1)
        acc = accuracy_score(y_true, y_pred, sample_weight=mask)
        f1_macro = f1_score(y_true, y_pred, sample_weight=mask,
                            average='macro', zero_division=0)
        return acc, f1_macro

    def training_step(self, train_batch, batch_idx):
        x, mask, y = train_batch
        logits = self.forward(x, mask)
        loss = self.loss(logits, y)
        acc, f1_macro = self.score(logits, y)
        self.log_dict({"train_loss": loss,
                       "train_acc": acc,
                       "train_f1": f1_macro}, prog_bar=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, mask, y = val_batch
        logits = self.forward(x, mask)
        loss = self.loss(logits, y)
        acc, f1_macro = self.score(logits, y)
        self.log_dict({"val_loss": loss,
                       "val_acc": acc,
                       "val_f1": f1_macro},
                      prog_bar=True)

    def test_step(self, test_batch, batch_idx):
        x, mask, y = test_batch
        logits = self.forward(x, mask)
        loss = self.loss(logits, y)
        acc, f1_macro = self.score(logits, y)
        self.log_dict({"test_loss": loss,
                       "test_acc": acc,
                       "test_f1": f1_macro}, prog_bar=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
