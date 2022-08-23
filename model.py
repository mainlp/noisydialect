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


def read_raw_input(filename, n_sents, encoding="utf8", verbose=True):
    """
    Reads the original (non-BERT) tokens and their labels
    """
    if verbose:
        print("Reading data from " + filename)
    toks, pos = [], []
    with open(filename, encoding=encoding) as f_in:
        cur_toks, cur_pos = [], []
        i = 0
        for line in f_in:
            line = line.strip()
            if not line:
                if cur_toks:
                    toks.append(cur_toks)
                    pos.append(cur_pos)
                    i += 1
                    cur_toks, cur_pos = [], []
                    if n_sents == i:
                        break
                    if verbose and i % 500 == 0:
                        print(i)
                continue
            *words, word_pos = line.split()
            cur_toks += [word for word in words]
            cur_pos.append(word_pos)
            # Treat multi-word tokens (that will be split up anyway)
            # like subtokens
            cur_pos += [DUMMY_POS for _ in range(1, len(words))]
        if cur_toks:
            toks.append(cur_toks)
            pos.append(cur_pos)
    assert len(toks) == len(pos), f"{len(toks)} == {len(pos)}"
    return toks, pos


def input2tensors(toks_orig, pos_orig, tokenizer, T,
                  tag2idx=None, encoding='utf8', verbose=True):
    assert T >= 2
    N = len(toks_orig)
    x = np.zeros((N, T), dtype=np.float64)
    toks, pos = [], []
    input_mask = np.zeros((N, T))
    # real_pos = 1 if full token or beginning of a token,
    # 0 if subword token from later on in the word with dummy tag
    real_pos = np.zeros((N, T))
    cur_toks, cur_pos = [], []
    for i, (sent_toks, sent_pos) in enumerate(zip(toks_orig, pos_orig)):
        if verbose and i % 500 == 0:
            print(i)
        cur_toks = ["[CLS]"]
        cur_pos = [DUMMY_POS]
        for token, pos_tag in zip(sent_toks, sent_pos):
            subtoks = tokenizer.tokenize(token)
            cur_toks += subtoks
            cur_pos += [pos_tag]
            cur_pos += [DUMMY_POS for _ in range(1, len(subtoks))]
        cur_toks = cur_toks[:T - 1] + ["SEP"]
        toks.append(cur_toks)
        input_mask[i][:len(cur_toks)] = len(cur_toks) * [1]
        x[i][:len(cur_toks)] = tokenizer.convert_tokens_to_ids(cur_toks)
        cur_pos = (cur_pos[:T - 1]
                   + [DUMMY_POS]  # SEP
                   + (T - len(cur_pos) - 1) * [DUMMY_POS]  # padding
                   )
        pos.append(cur_pos)
        real_pos[i][:len(cur_pos)] = [0 if p == DUMMY_POS
                                      else 1 for p in cur_pos]
    if not tag2idx:
        # BERT doesn't want labels that are already onehot-encoded
        tag2idx = {tag: idx for idx, tag in enumerate(
            {tok_pos for sent_pos in pos for tok_pos in sent_pos})}
    y = np.empty((N, T))
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


def subtok_ratio(toks_orig, toks_bert):
    return (len(toks_bert) - len(toks_orig)) / len(toks_bert)


def visualize(matrix, name):
    plt.clf()
    plt.pcolormesh(matrix)
    plt.savefig(f"figs/{name}.png")


def train_and_validate(model, device, iter_train, iter_dev, iter_test,
                       optimizer, scheduler, n_epochs):
    for epoch in range(n_epochs):
        print("============")
        print(f"Epoch {epoch + 1}/{n_epochs} started at " + now())
        train(model, device, iter_train, optimizer, scheduler)
        eval(model, device, iter_dev)
        eval(model, device, iter_test)


def now():
    return time.strftime("%Y/%m/%d %H:%M:%S", time.localtime())


def train(model, device, iter_train, optimizer, scheduler):
    model.train()  # Switch to training mode
    train_loss = 0

    n_batches = len(iter_train)
    for i, batch in enumerate(iter_train):
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


def eval(model, device, iter_test):
    model.eval()
    n_batches = len(iter_test)
    for i, batch in enumerate(iter_test):
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
    parser.add_argument("-r", dest="file_td",
                        help="training/development data file",
                        default="data/hamburg-dependency-treebank/"
                                "train_DHT_STTS.txt")
    parser.add_argument("-e", dest="file_test",
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
    # parser.add_argument("-q", "--quiet", action="store_false", dest="verbose",
    #                     default=True, help="no messages to stdout")

    args = parser.parse_args()
    sys.stdout = cust_logger.Logger("eda")
    for arg in vars(args):
        print(arg, getattr(args, arg))

    tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-german-cased")

    # Prepare training/validation data: (modified) HRL tokens
    toks_td, pos_td = read_raw_input(args.file_td, args.n_sents_td)
    # TODO modify the tokens
    toks_orig_train, toks_orig_dev, pos_train, pos_dev = train_test_split(
        toks_td, pos_td, test_size=args.dev_ratio)
    (toks_train, x_train, y_train,
        input_mask_train, real_pos_train, tag2idx) = input2tensors(
        toks_orig_train, pos_train, tokenizer, args.max_len)
    print("Subtoken ratio (training)",
          subtok_ratio(toks_orig_train, toks_train))
    (toks_dev, x_dev, y_dev,
        input_mask_dev, real_pos_dev, _) = input2tensors(
        toks_orig_dev, pos_dev, tokenizer, args.max_len, tag2idx=tag2idx)
    print("Subtoken ratio (development)",
          subtok_ratio(toks_orig_dev, toks_dev))
    visualize(x_train, f"x_train_{x_train.shape[0]}")
    visualize(y_train, f"y_train_{x_train.shape[0]}")
    visualize(real_pos_train, f"real_pos_train_{x_train.shape[0]}")
    visualize(input_mask_train, f"input_mask_train_{x_train.shape[0]}")
    visualize(x_dev, f"x_dev_{x_dev.shape[0]}")
    visualize(y_dev, f"y_dev_{x_dev.shape[0]}")
    visualize(real_pos_dev, f"real_pos_dev_{x_dev.shape[0]}")
    visualize(input_mask_dev, f"input_mask_dev_{x_dev.shape[0]}")

    # Prepare test data: LRL tokens
    toks_orig_test, pos_test = read_raw_input(args.file_test,
                                              args.n_sents_test)
    (toks_test, x_test, y_test,
        input_mask_test, real_pos_test, _) = input2tensors(
        toks_orig_test, pos_test, tokenizer, args.max_len, tag2idx=tag2idx)
    print("Subtoken ratio (test)",
          subtok_ratio(toks_orig_test, toks_orig_test))
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

    data_train = TensorDataset(x_train, input_mask_train, y_train)
    iter_train = DataLoader(data_train, sampler=RandomSampler(data_train),
                            batch_size=batch_size)
    data_dev = TensorDataset(x_dev, input_mask_dev, y_dev)
    iter_dev = DataLoader(data_dev, sampler=RandomSampler(data_dev),
                          batch_size=batch_size)
    data_test = TensorDataset(x_test, input_mask_test, y_test)
    iter_test = DataLoader(data_test, sampler=RandomSampler(data_test),
                           batch_size=batch_size)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0,
        num_training_steps=len(iter_train) * n_epochs)

    train_and_validate(model, device, iter_train, iter_dev, iter_test,
                       optimizer, scheduler, n_epochs)
