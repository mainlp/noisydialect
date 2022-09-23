from data import DUMMY_POS
from embeddings import CombinedEmbeddings

import copy

import numpy as np
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score, f1_score
import torch
from torch.nn import CrossEntropyLoss
from transformers import BertForTokenClassification, BertModel
from transformers.modeling_outputs import \
    BaseModelOutputWithPoolingAndCrossAttentions
from transformers.models.bert.modeling_bert import BertForMaskedLM


class Classifier(pl.LightningModule):
    def __init__(self, pretrained_model_name_or_path,
                 pos2idx, classifier_dropout, learning_rate,
                 use_sca_embeddings=False, subtok2weight=None,
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
        if use_sca_embeddings:
            sca_bert = BertForCombinedEmbeddings(self.pretraining_model.bert)
            self.pretraining_model.bert = sca_bert
            self.finetuning_model.bert = sca_bert
            sca_embeddings = CombinedEmbeddings(sca_bert.embeddings,
                                                subtok2weight)
            self.pretraining_model.embeddings = sca_embeddings
            self.pretraining_model.bert.embeddings = sca_embeddings
            self.finetuning_model.embeddings = sca_embeddings
            self.finetuning_model.bert.embeddings = sca_embeddings
        else:
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
        y_true = labels.detach().clone().cpu().numpy().flatten()
        y_pred = np.argmax(logits.detach().clone().cpu().numpy(),
                           axis=2).flatten()
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
        self.log_dict({"train_loss": loss, "train_acc": acc, "train_f1": f1_macro}, prog_bar=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, mask, y = val_batch
        logits = self.forward(x, mask)
        loss = self.loss(logits, y)
        acc, f1_macro = self.score(logits, y)
        self.log_dict({"val_loss": loss, "val_acc": acc, "val_f1": f1_macro}, prog_bar=True)

    def test_step(self, test_batch, batch_idx):
        x, mask, y = test_batch
        logits = self.forward(x, mask)
        loss = self.loss(logits, y)
        acc, f1_macro = self.score(logits, y)
        self.log_dict({"test_loss": loss, "test_acc": acc, "test_f1": f1_macro}, prog_bar=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch[0], batch[1])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


class BertForCombinedEmbeddings(BertModel):
    def __init__(self, bert_orig):
        self.__dict__.update(bert_orig.__dict__)

    # Overrides the BertModel method:
    # https://github.com/huggingface/transformers/blob/v4.22.1/src/transformers/models/bert/modeling_bert.py#L874
    # and merely changes how the input size is derived from the
    # input_ids matrix
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            # !! The sole change is here (removing the extra dimension
            # used for storing the token "siblings"):
            # input_shape = input_ids.size()
            input_shape = input_ids.size()[:-1]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )
