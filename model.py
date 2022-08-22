import cust_logger

from argparse import ArgumentParser
import sys
import time

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import AdamW, AutoTokenizer, BertForTokenClassification, \
    get_linear_schedule_with_warmup
import numpy as np
import torch

# Used for CLS, SEP, and for word-medial/final subtokens
DUMMY_POS = "<DUMMY>"


def add_sent_data(x, toks, pos, input_mask, real_pos,
                  cur_toks, cur_pos,
                  max_len, i, tokenizer):
    cur_toks = ["[CLS"] + cur_toks[:max_len - 2] + ["SEP"]
    toks.append(cur_toks)
    x[i][:len(cur_toks)] = tokenizer.convert_tokens_to_ids(cur_toks)
    cur_pos = ([DUMMY_POS]
               + cur_pos[:max_len - 2]
               + (max_len - len(cur_pos) - 2) * [DUMMY_POS]
               + [DUMMY_POS])
    pos.append(cur_pos)
    real_pos[i][:len(cur_pos)] = [0 if p == DUMMY_POS else 1 for p in cur_pos]
    input_mask[i][:len(cur_pos)] = len(cur_pos) * [1]
    return x, toks, pos, real_pos


def read_input_data(filename, tokenizer, n_sents, max_len,
                    tag2idx=None, encoding='utf8', verbose=True):
    assert max_len >= 2
    if verbose:
        print("Reading data from " + filename)
    with open(filename, encoding=encoding) as f_in:
        x = np.zeros((n_sents, max_len), dtype=np.float64)
        toks, pos = [], []
        input_mask = np.zeros((n_sents, max_len))
        # real_pos = 1 if full token or beginning of a token,
        # 0 if subword token from later on in the word with dummy tag
        real_pos = np.zeros((n_sents, max_len))
        cur_toks, cur_pos = [], []
        i = 0
        for line in f_in:
            line = line.strip()
            if not line:
                if cur_toks:
                    x, toks, pos, real_pos = add_sent_data(
                        x, toks, pos, input_mask, real_pos, cur_toks, cur_pos,
                        max_len, i, tokenizer)
                    i += 1
                    cur_toks, cur_pos = [], []
                    if n_sents == i:
                        break
                    if verbose and i % 500 == 0:
                        print(i)
                continue
            *words, word_pos = line.split()
            word_toks = []
            for word in words:
                word_toks += tokenizer.tokenize(word)
            cur_toks += word_toks
            cur_pos.append(word_pos)
            cur_pos += [DUMMY_POS for _ in range(1, len(word_toks))]
        if cur_toks:
            add_sent_data(x, toks, pos, input_mask, real_pos, cur_toks,
                          cur_pos, max_len, i, tokenizer)
    if not tag2idx:
        # BERT doesn't want labels that are already onehot-encoded
        tag2idx = {tag: idx for idx, tag in enumerate(
            {tok_pos for sent_pos in pos for tok_pos in sent_pos})}
    y = np.empty((n_sents, max_len))
    for i, sent_pos in enumerate(pos):
        y[i] = [tag2idx[tok_pos] for tok_pos in sent_pos]
    assert len(toks) == x.shape[0] == len(pos), \
        f"{len(toks)} == {x.shape[0]} == {len(pos)}"
    if verbose:
        print(f"{len(toks)} sentences")
        for i in zip(toks[0], pos[0], x[0]):
            print(i)
        print("\n")
    return toks, x, y, input_mask, real_pos, tag2idx


def train_and_validate(model, device, train_iter, test_iter, optimizer,
                       scheduler, n_epochs):
    for epoch in range(n_epochs):
        print("============")
        print(f"Epoch {epoch + 1} / {n_epochs} started at "
              + time.strftime("%Y%M%d-%H%M%S", time.localtime()))
        train(model, device, train_iter, optimizer, scheduler)
        eval(model, device, test_iter)


def train(model, device, train_iter, optimizer, scheduler):
    model.train()  # Switch to training mode
    train_loss = 0

    n_batches = len(train_iter)
    for i, batch in enumerate(train_iter):
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        model.zero_grad()  # Clear old gradients

        # Forward pass
        # TODO: override! custom loss function taking into account the
        # subword tokens!!!
        loss, logits, _, _ = model(b_input_ids, b_input_mask,
                                   labels=b_labels)
        train_loss += loss.item()

        loss.backward()  # Calculate gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

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

        if i % 10 == 0:
            print(f"Batch {i:>2} / {n_batches}, loss: {loss.item()}")
    print(f"Mean train loss for epoch: {train_loss / n_batches}")


def eval(model, device, test_iter):
    model.eval()
    n_batches = len(test_iter)
    for i, batch in enumerate(test_iter):
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        eval_loss = 0
        with torch.no_grad():
            # TODO: override! custom loss function taking into account the
            # subword tokens!!!
            loss, logits, _, _ = model(b_input_ids, b_input_mask,
                                       labels=b_labels)
            eval_loss += loss.item()

            # TODO additional metrics
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            # metrics here!
    print(f"Mean validation loss for epoch: {eval_loss / n_batches}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-r", dest="train_file",
                        help="training data file",
                        default="data/hamburg-dependency-treebank/"
                                "train_DHT_STTS.txt")
    parser.add_argument("-e", dest="test_file",
                        help="test data file",
                        default="data/NOAH-corpus/test_GSW_STTS.txt")
    parser.add_argument("-nr", dest="n_sents_train",
                        help="no. of sentences in the training data",
                        default=206794, type=int)
    parser.add_argument("-ne", dest="n_sents_test",
                        help="no. of sentences in the test data",
                        default=7320, type=int)
    parser.add_argument("-m", dest="max_len",
                        help="max. number of tokens per sentence "
                        "(incl. CLS and SEP)",
                        default=54, type=int)

    args = parser.parse_args()
    sys.stdout = cust_logger.Logger("eda")
    for arg in vars(args):
        print(arg, getattr(args, arg))

    tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-german-cased")

    (toks_train, x_train, y_train,
        input_mask_train, real_pos_train, tag2idx) = read_input_data(
        args.train_file, tokenizer, args.n_sents_train, args.max_len)
    (toks_test, x_test, y_test,
        input_mask_test, real_pos_test, _) = read_input_data(
        args.test_file, tokenizer, args.n_sents_test, args.max_len,
        tag2idx=tag2idx)

    model = BertForTokenClassification.from_pretrained(
        "dbmdz/bert-base-german-cased", num_labels=len(tag2idx))
    if torch.cuda.is_available():
        model.cuda()  # TODO
        device = 'cuda'
    else:
        device = 'cpu'
    optimizer = AdamW(model.parameters())
    n_epochs = 2  # TODO move various settings to the console args
    batch_size = 32

    train_data = TensorDataset(x_train, input_mask_train, y_train)
    train_iter = DataLoader(train_data, sampler=RandomSampler(train_data),
                            batch_size=batch_size)
    test_data = TensorDataset(x_test, input_mask_test, y_test)
    test_iter = DataLoader(test_data, sampler=RandomSampler(test_data),
                           batch_size=batch_size)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0,
        num_training_steps=len(train_iter) * n_epochs)

    train_and_validate(model, device, train_iter, test_iter, optimizer,
                       scheduler, n_epochs)
