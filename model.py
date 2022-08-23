import cust_logger

from argparse import ArgumentParser
import sys
import time

from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from transformers import AutoTokenizer, BertForTokenClassification, \
    get_linear_schedule_with_warmup
import matplotlib.pyplot as plt
import numpy as np
import torch

# Used for CLS, SEP, and for word-medial/final subtokens
DUMMY_POS = "<DUMMY>"


def add_sent_data(x, toks, pos, input_mask, real_pos,
                  cur_toks, cur_pos,
                  max_len, i, tokenizer):
    cur_toks = ["[CLS"] + cur_toks[:max_len - 2] + ["SEP"]
    toks.append(cur_toks)
    input_mask[i][:len(cur_toks)] = len(cur_toks) * [1]
    x[i][:len(cur_toks)] = tokenizer.convert_tokens_to_ids(cur_toks)
    cur_pos = ([DUMMY_POS]
               + cur_pos[:max_len - 2]
               + (max_len - len(cur_pos) - 2) * [DUMMY_POS]
               + [DUMMY_POS])
    pos.append(cur_pos)
    real_pos[i][:len(cur_pos)] = [0 if p == DUMMY_POS else 1 for p in cur_pos]
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
        for i in zip(toks[0], x[0], pos[0], y[0], input_mask[0], real_pos[0]):
            print(i)
        print("\n")
    return toks, Tensor(x).to(torch.int64), Tensor(y).to(torch.int64), \
        Tensor(input_mask).to(torch.int64), real_pos, tag2idx


def visualize(matrix, name):
    plt.clf()
    plt.pcolormesh(matrix)
    plt.savefig(f"figs/{name}.png")


def train_and_validate(model, device, train_iter, dev_iter, test_iter,
                       optimizer, scheduler, n_epochs):
    for epoch in range(n_epochs):
        print("============")
        print(f"Epoch {epoch + 1}/{n_epochs} started at " + now())
        train(model, device, train_iter, optimizer, scheduler)
        eval(model, device, dev_iter)
        eval(model, device, test_iter)


def now():
    return time.strftime("%Y/%m/%d %H:%M:%S", time.localtime())


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
#         loss, logits, _, _ = model(b_input_ids, b_input_mask,
#                                    labels=b_labels)
        out = model(b_input_ids, b_input_mask, labels=b_labels)
        train_loss += out.loss.item()

        out.loss.backward()  # Calculate gradients
        clip_grad_norm_(model.parameters(), 1.0)

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
            print(f"Batch {i:>2}/{n_batches} {now()} loss: {out.loss.item()}")
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
            out = model(b_input_ids, b_input_mask, labels=b_labels)
            eval_loss += out.loss.item()

            # TODO additional metrics
            logits = out.logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            # metrics here!
    print(f"Mean validation loss for epoch: {eval_loss / n_batches}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-r", dest="td_file",
                        help="training/development data file",
                        default="data/hamburg-dependency-treebank/"
                                "train_DHT_STTS.txt")
    parser.add_argument("-e", dest="test_file",
                        help="test data file",
                        default="data/NOAH-corpus/test_GSW_STTS.txt")
    parser.add_argument("-nr", dest="n_sents_td",
                        help="no. of sentences in the training/dev data",
                        default=206794, type=int)
    parser.add_argument("-ne", dest="n_sents_test",
                        help="no. of sentences in the test data",
                        default=7320, type=int)
    parser.add_argument("-m", dest="max_len",
                        help="max. number of tokens per sentence "
                        "(incl. CLS and SEP)",
                        default=54, type=int)
    parser.add_argument("-d", dest="dev_ratio",
                        help="dev:dev+train ratio",
                        default=0.1, type=float)

    args = parser.parse_args()
    sys.stdout = cust_logger.Logger("eda")
    for arg in vars(args):
        print(arg, getattr(args, arg))

    tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-german-cased")

    (toks_td, x_td, y_td,
        input_mask_td, real_pos_td, tag2idx) = read_input_data(
        args.td_file, tokenizer, args.n_sents_td, args.max_len)
    (toks_train, toks_dev, x_train, x_dev,
        y_train, y_dev, input_mask_train, input_mask_dev,
        real_pos_train, real_pos_dev) = train_test_split(
        toks_td, x_td, y_td, input_mask_td, real_pos_td,
        test_size=args.dev_ratio)
    visualize(x_train, f"x_train_{x_train.shape[0]}")
    visualize(y_train, f"y_train_{x_train.shape[0]}")
    visualize(real_pos_train, f"real_pos_train_{x_train.shape[0]}")
    visualize(input_mask_train, f"input_mask_train_{x_train.shape[0]}")
    visualize(x_dev, f"x_dev_{x_dev.shape[0]}")
    visualize(y_dev, f"y_dev_{x_dev.shape[0]}")
    visualize(real_pos_dev, f"real_pos_dev_{x_dev.shape[0]}")
    visualize(input_mask_dev, f"input_mask_dev_{x_dev.shape[0]}")
    (toks_test, x_test, y_test,
        input_mask_test, real_pos_test, _) = read_input_data(
        args.test_file, tokenizer, args.n_sents_test, args.max_len,
        tag2idx=tag2idx)
    visualize(x_test, f"x_test_{args.n_sents_test}")
    visualize(y_test, f"y_test_{args.n_sents_test}")
    visualize(y_train, f"y_train_{args.n_sents_test}")
    visualize(real_pos_test, f"real_pos_test_{args.n_sents_test}")
    visualize(input_mask_test, f"input_mask_test_{args.n_sents_test}")

    model = BertForTokenClassification.from_pretrained(
        "dbmdz/bert-base-german-cased", num_labels=len(tag2idx))
    print(model)
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
    dev_data = TensorDataset(x_dev, input_mask_dev, y_dev)
    dev_iter = DataLoader(dev_data, sampler=RandomSampler(dev_data),
                          batch_size=batch_size)
    test_data = TensorDataset(x_test, input_mask_test, y_test)
    test_iter = DataLoader(test_data, sampler=RandomSampler(test_data),
                           batch_size=batch_size)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0,
        num_training_steps=len(train_iter) * n_epochs)

    train_and_validate(model, device, train_iter, dev_iter, test_iter,
                       optimizer, scheduler, n_epochs)
