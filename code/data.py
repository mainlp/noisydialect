import copy
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import TensorDataset

# Used for CLS, SEP, and for word-medial/final subtokens
DUMMY_POS = "<DUMMY>"


def read_raw_input(filename, max_sents, encoding="utf8",
                   verbose=True):
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
                    if max_sents == i:
                        break
                    if verbose and i % 1000 == 0:
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


class Data:
    def __init__(self,
                 name,
                 toks_orig=None, pos_orig=None,
                 toks_bert=None, x=None, y=None,
                 input_mask=None, pos_mask=None,
                 pos2idx=None,
                 # If initializing from files:
                 load_parent_dir=None,
                 raw_data_path=None, max_sents=-1,
                 ):
        self.name = name
        self.toks_orig = toks_orig
        self.toks_bert = toks_bert
        self.pos_orig = pos_orig
        self.x = x
        self.y = y
        self.input_mask = input_mask
        self.pos_mask = pos_mask
        self.pos2idx = pos2idx
        if load_parent_dir:
            self.load(load_parent_dir)
        elif raw_data_path:
            self.read_raw_input(raw_data_path, max_sents)
        print(self)

    def subtok_ratio(self):
        try:
            return (len(self.toks_bert) - len(self.toks_orig)) / \
                len(self.toks_bert)
        except TypeError:
            return -1

    @staticmethod
    def visualize(matrix, name):
        plt.clf()
        plt.pcolormesh(matrix)
        plt.savefig(f"../figs/{name}.png")

    def tensor_dataset(self):
        return TensorDataset(torch.Tensor(self.x).to(torch.int64),
                             torch.Tensor(self.input_mask).to(torch.int64),
                             torch.Tensor(self.y).to(torch.int64))

    def alphabet(self):
        return {c for sent in self.toks_orig for tok in sent for c in tok
                if c != ' '}

    # ---- Initializing ----

    def read_raw_input(self, filename, max_sents, encoding="utf8",
                       verbose=True):
        self.toks_orig, self.pos_orig = read_raw_input(
            filename, max_sents, encoding, verbose)

    def add_random_noise(self, noise_lvl_min, noise_lvl_max, alphabet,
                         noise_types=["add_char", "delete_char",
                                        "replace_char"]):
        """
        Aepli & Sennrich 2022,
        Wang, Ruder & Neubig 2021
        """
        toks_noisy = []
        for sent_toks in self.toks_orig:
            percentage_noisy = random.randrange(
                round(100 * noise_lvl_min), round(100 * noise_lvl_max)) / 100
            idx_noisy = random.sample(
                range(len(sent_toks)),
                k=round(percentage_noisy * len(sent_toks)))
            sent_toks_noisy = []
            for i, tok in enumerate(sent_toks):
                if i in idx_noisy:
                    sent_toks_noisy.append(
                        getattr(self, random.sample(noise_types, 1)[0])(
                            tok, alphabet))
                else:
                    sent_toks_noisy.append(tok)
            toks_noisy.append(sent_toks_noisy)
        return toks_noisy

    @staticmethod
    def add_char(word, alphabet):
        idx = random.randrange(-1, len(word))
        if idx == -1:
            return random.sample(alphabet, 1)[0] + word
        return word[:idx + 1] + random.sample(alphabet, 1)[0] + word[idx + 1:]

    @staticmethod
    def delete_char(word, alphabet):
        idx = random.randrange(0, len(word))
        return word[:idx] + word[idx + 1:]

    @staticmethod
    def replace_char(word, alphabet):
        idx = random.randrange(0, len(word))
        return word[:idx] + random.sample(alphabet, 1)[0] + word[idx + 1:]

    def prepare_xy(self, tokenizer, T, verbose=True):
        assert T >= 2
        N = len(self.toks_orig)
        self.x = np.zeros((N, T), dtype=np.float64)
        self.toks_bert, pos = [], []
        self.input_mask = np.zeros((N, T))
        # real_pos = 1 if full token or beginning of a token,
        # 0 if subword token from later on in the word with dummy tag
        self.real_pos = np.zeros((N, T))
        cur_toks, cur_pos = [], []
        for i, (sent_toks, sent_pos) in enumerate(
                zip(self.toks_orig, self.pos_orig)):
            if verbose and i % 1000 == 0:
                print(i)
            cur_toks = ["[CLS]"]
            cur_pos = [DUMMY_POS]
            for token, pos_tag in zip(sent_toks, sent_pos):
                subtoks = tokenizer.tokenize(token)
                cur_toks += subtoks
                cur_pos += [pos_tag]
                cur_pos += [DUMMY_POS for _ in range(1, len(subtoks))]
            cur_toks = cur_toks[:T - 1] + ["SEP"]
            self.toks_bert.append(cur_toks)
            self.input_mask[i][:len(cur_toks)] = len(cur_toks) * [1]
            self.x[i][:len(cur_toks)] = tokenizer.convert_tokens_to_ids(
                cur_toks)
            cur_pos = (cur_pos[:T - 1]
                       + [DUMMY_POS]  # SEP
                       + (T - len(cur_pos) - 1) * [DUMMY_POS]  # padding
                       )
            pos.append(cur_pos)
            self.real_pos[i][:len(cur_pos)] = [0 if p == DUMMY_POS
                                               else 1 for p in cur_pos]
        if not self.pos2idx:
            # BERT doesn't want labels that are already onehot-encoded
            self.pos2idx = {tag: idx for idx, tag in enumerate(
                {tok_pos for sent_pos in pos for tok_pos in sent_pos})}
        self.y = np.empty((N, T))
        for i, sent_pos in enumerate(pos):
            self.y[i] = [self.pos2idx[tok_pos] for tok_pos in sent_pos]
        assert len(self.toks_bert) == self.x.shape[0] == len(pos), \
            f"{len(self.toks_bert)} == {self.x.shape[0]} == {len(pos)}"
        if verbose:
            print(f"{len(self.toks_bert)} sentences")
            for i in zip(self.toks_bert[0], self.x[0], pos[0], self.y[0],
                         self.input_mask[0], self.real_pos[0]):
                print(i)
            print("\n")

    # ----------------------
    # --- Saving/loading ---

    def save(self, parent_dir='../data/'):
        if not os.path.exists(parent_dir):
            os.mkdir(parent_dir)
        directory = os.path.join(parent_dir, self.name)
        if not os.path.exists(directory):
            os.mkdir(directory)
        # Also works with None args:
        np.savez(os.path.join(directory, "arrays.npz"),
                 x=self.x, y=self.y, input_mask=self.input_mask,
                 pos_mask=self.pos_mask)
        self.save_tsv(self.toks_orig, directory, "toks_orig.tsv")
        self.save_tsv(self.toks_bert, directory, "toks_bert.tsv")
        self.save_tsv(self.pos_orig, directory, "pos_orig.tsv")
        if self.pos2idx:
            with open(os.path.join(directory, "pos2idx.tsv"), 'w',
                      encoding='utf8') as f:
                for pos in self.pos2idx:
                    f.write(f"{pos}\t{self.pos2idx[pos]}\n")

    @staticmethod
    def save_tsv(data, directory, filename):
        if data is None:
            return
        with open(os.path.join(directory, filename), 'w',
                  encoding='utf8') as f:
            for sent in data:
                f.write("\t".join(sent) + '\n')

    def load(self, parent_dir='../data/'):
        directory = os.path.join(parent_dir, self.name)
        npzfile = np.load(os.path.join(directory, "arrays.npz"))
        try:
            self.x = npzfile["x"]
        except ValueError:
            print("Couldn't load 'x'")
            pass
        try:
            self.y = npzfile["z"]
        except ValueError:
            print("Couldn't load 'z'")
            pass
        try:
            self.input_mask = npzfile["input_mask"]
        except ValueError:
            print("Couldn't load 'input_mask'")
            pass
        try:
            self.pos_mask = npzfile["pos_mask"]
        except ValueError:
            print("Couldn't load 'pos_mask'")
            pass
        self.toks_orig = self.tsv2list(directory, "toks_orig.tsv")
        self.toks_bert = self.tsv2list(directory, "toks_bert.tsv")
        self.pos_orig = self.tsv2list(directory, "pos_orig.tsv")
        try:
            with open(os.path.join(directory, "pos2idx.tsv"), 'r',
                      encoding='utf8') as f:
                self.pos2idx = {}
                for line in f:
                    cells = line.strip().split("\t")
                    if cells:
                        self.pos2idx[cells[0]] = int(cells[1])
        except FileNotFoundError:
            print("Couldn't load 'pos2idx.tsv'")

    @staticmethod
    def tsv2list(directory, filename):
        try:
            with open(os.path.join(directory, filename), 'r',
                      encoding='utf8') as f:
                outer = []
                for line in f:
                    cells = line.strip().split("\t")
                    if cells:
                        outer.append(cells)
            return outer
        except FileNotFoundError:
            print(f"Couldn't load '{filename}'")
            return None

    @staticmethod
    def list2str(nested_list):
        if nested_list is not None:
            return f"({len(nested_list)}, ?)"
        return "None"

    @staticmethod
    def tensor2str(tensor):
        if tensor is not None:
            return f"{tuple(tensor.shape)}"
        return "None"

    # ------------------------
    # --- Copying, __str__ ---

    def __str__(self):
        return f"Data(name={self.name}, " \
            f"toks_orig={self.list2str(self.toks_orig)}, " \
            f"pos_orig={self.list2str(self.pos_orig)}, " \
            f"toks_bert={self.list2str(self.toks_bert)}, " \
            f"x={self.tensor2str(self.x)}, " \
            f"y={self.tensor2str(self.y)}, " \
            f"input_mask={self.tensor2str(self.input_mask)}, " \
            f"pos_mask={self.list2str(self.pos_mask)}, " \
            f"pos2idx={'None' if self.pos2idx is None else len(self.pos2idx)})"

    def original_data(self):
        return copy.deepcopy(self.toks_orig), copy.deepcopy(self.pos_orig)
    # ------------------------
