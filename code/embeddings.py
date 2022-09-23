import torch
from transformers.models.bert.modeling_bert import BertEmbeddings
# TODO look into torch.nn.EmbeddingBag


class CombinedEmbeddings(BertEmbeddings):
    def __init__(self, pretrained_embeddings, subtok2weight=None):
        self.__dict__.update(pretrained_embeddings.__dict__)
        self.subtok2weight = subtok2weight

    # Overrides the BertEmbeddings method
    # https://github.com/huggingface/transformers/blob/2c8b508ccabea6638aa463a137852ff3b64be036/src/transformers/models/bert/modeling_bert.py#L209
    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        past_key_values_length=0,
    ):
        if input_ids is not None:
            # This is changed (we need to remove the extra dimension
            # used for storing the list of token "siblings"):
            # input_shape = input_ids.size()  # original
            input_shape = input_ids.size()[:-1]
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:,past_key_values_length : seq_length + past_key_values_length]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            # The contents of this if-block are changed from the
            # BertEmbeddings version:
            # inputs_embeds = self.word_embeddings(input_ids)  # original
            no_bags = input_ids.size()[-1] == 1
            inputs_embeds = torch.zeros(
                [d for d in input_shape] + [self.word_embeddings.embedding_dim],
                device=input_ids.device)
            for batch in range(input_shape[0]):
                for i, id_list in enumerate(input_ids[batch]):
                    if id_list[0] == 0 or no_bags:
                        # 0 = [PAD]; confirmed via assert in Data class
                        inputs_embeds[batch][i] = self.word_embeddings(
                            id_list[0])
                    elif id_list[1] == 0:
                        # Any 0s after the first position just mean that
                        # the list of token "siblings" is over.
                        # If there's only one non-0 entry, we don't need
                        # to do any averaging.
                        inputs_embeds[batch][i] = self.word_embeddings(
                            id_list[0])
                    else:
                        end_idx = len(id_list)
                        for j in range(2, len(id_list)):
                            if id_list[j] == 0:
                                end_idx = j
                                break
                        embedding_bag = self.word_embeddings(id_list[:end_idx])
                        # If we have weights for the subtokens, weigh each
                        # embedding according to its weight, otherwise just
                        # use the mean.
                        if self.subtok2weight:
                            weights = []
                            found_weights = False
                            for idx in id_list[:end_idx]:
                                try:
                                    weights.append(self.subtok2weight[
                                        idx.item()])
                                    found_weights = True
                                except KeyError:
                                    weights.append(0.0)
                            if found_weights:
                                for weight, emb in zip(weights, embedding_bag):
                                    inputs_embeds[batch][i] += weight * emb
                                else:
                                    inputs_embeds[batch][i] = embedding_bag \
                                        .mean(axis=0)
                        else:
                            inputs_embeds[batch][i] = embedding_bag.mean(
                                axis=0)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
