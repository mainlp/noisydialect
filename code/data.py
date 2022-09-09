from collections import Counter
import copy
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer
import unicodedata

# Used for CLS, SEP, and for word-medial/final subtokens
DUMMY_POS = "<DUMMY>"


class PosDataModule(pl.LightningDataModule):
    def __init__(self, config, pos2idx):
        super().__init__()
        self.config = config
        self.pos2idx = pos2idx
        self.tokenizer = None
        if config.tokenizer_name:
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.tokenizer_name)

    def prepare_data(self):
        # Training/validation data: HRL tokens
        if self.config.prepare_input_traindev:
            if self.config.orig_dir_train and self.config.orig_dir_dev:
                train = Data(self.config.name_train,
                             other_dir=self.config.orig_dir_train)
                dev = Data(self.config.name_dev,
                           other_dir=self.config.orig_dir_dev)
            else:
                toks_td, pos_td = read_raw_input(
                    self.config.orig_file_traindev,
                    self.config.max_sents_traindev,
                    self.config.encoding_traindev)
                (toks_orig_train, toks_orig_dev,
                    pos_train, pos_dev) = train_test_split(
                    toks_td, pos_td, test_size=self.config.dev_ratio)
                train = Data(self.config.name_train, toks_orig=toks_orig_train,
                             pos_orig=pos_train, pos2idx=self.pos2idx)
                dev = Data(self.config.name_dev, toks_orig=toks_orig_dev,
                           pos_orig=pos_dev, pos2idx=self.pos2idx)

        # Test data: LRL tokens
        if self.config.prepare_input_test:
            test = Data(self.config.name_test,
                        raw_data_path=self.config.orig_file_test,
                        raw_data_enc=self.config.encoding_test,
                        max_sents=self.config.max_sents_test,
                        pos2idx=self.pos2idx)
            test.prepare_xy(self.tokenizer, self.config.T,
                            self.config.subtoken_rep)
            test.save(self.config.data_parent_dir)
            print(f"Subtoken ratio ({self.config.name_test}): {test.subtok_ratio(return_all=True)}")
            print(f"UNK ratio ({self.config.name_test}): {test.unk_ratio(return_all=True)}")
            print(f"Label distribution ({self.config.name_test}): {test.pos_y_distrib()}")
        else:
            test = Data(self.config.name_test,
                        load_parent_dir=self.config.data_parent_dir)
        alphabet_test = test.alphabet()

        # Prepare input matrices for finetuning
        if self.config.prepare_input_traindev:
            train.add_noise(self.config.noise_type, self.config.noise_lvl_min,
                            self.config.noise_lvl_max, alphabet_test)
            train.prepare_xy(self.tokenizer, self.config.T,
                             self.config.subtoken_rep)
            train.save(self.config.data_parent_dir)
            print(f"Subtoken ratio ({self.config.name_train}): {train.subtok_ratio(return_all=True)}")
            print(f"UNK ratio ({self.config.name_train}): {test.unk_ratio(return_all=True)}")
            print(f"Label distribution ({self.config.name_train}): {train.pos_y_distrib()}")
            dev.add_noise(self.config.noise_type, self.config.noise_lvl_min,
                          self.config.noise_lvl_max, alphabet_test)
            dev.prepare_xy(self.tokenizer, self.config.T,
                           self.config.subtoken_rep)
            dev.save(self.config.data_parent_dir)
            print(f"Subtoken ratio ({self.config.name_dev}): {dev.subtok_ratio(return_all=True)}")
            print(f"UNK ratio ({self.config.name_dev}): {test.unk_ratio(return_all=True)}")
            print(f"Label distribution ({self.config.name_dev}): {dev.pos_y_distrib()}")

    def setup(self, stage):
        if stage == 'fit':
            self.train = Data(self.config.name_train,
                              load_parent_dir=self.config.data_parent_dir)
            self.val = Data(self.config.name_dev,
                            load_parent_dir=self.config.data_parent_dir)
        elif stage in ['test', 'predict']:
            self.test = Data(self.config.name_test,
                             load_parent_dir=self.config.data_parent_dir)

    def print_preview(self, data):
        print(data)
        idx2pos = data.idx2pos()
        for tok, pos_idx in zip(data.toks_bert[0], data.y[0]):
            print(tok, idx2pos[pos_idx])

    def train_dataloader(self):
        self.print_preview(self.train)
        return DataLoader(self.train.tensor_dataset(),
                          batch_size=self.config.batch_size)

    def val_dataloader(self):
        self.print_preview(self.val)
        return DataLoader(self.val.tensor_dataset(),
                          batch_size=self.config.batch_size)

    def test_dataloader(self):
        self.print_preview(self.test)
        return DataLoader(self.test.tensor_dataset(),
                          batch_size=self.config.batch_size)

    def predict_dataloader(self):
        self.print_preview(self.test)
        return DataLoader(self.test.tensor_dataset(),
                          batch_size=self.config.batch_size)


# Preprocessing


def read_raw_input(filename, max_sents=-1, encoding="utf8",
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
            line = line.strip().replace("\xa0", " ")
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
            word, _, word_pos = line.rpartition(" ")
            cur_toks.append(word)
            cur_pos.append(word_pos)
        if cur_toks:
            toks.append(cur_toks)
            pos.append(cur_pos)
    assert len(toks) == len(pos), f"{len(toks)} == {len(pos)}"
    return toks, pos


class Data:
    def __init__(self,
                 name,
                 toks_orig=None, pos_orig=None,
                 toks_bert=None, x=None, y=None, input_mask=None,
                 pos2idx=None,
                 # If loading an existing dataset:
                 load_parent_dir=None,
                 # If initializing from another dataset:
                 other_dir=None,
                 # If initializing from scratch:
                 raw_data_path=None, raw_data_enc="utf8", max_sents=-1,
                 ):
        self.name = name
        self.toks_orig = toks_orig
        self.toks_bert = toks_bert
        self.pos_orig = pos_orig
        self.x = x
        self.y = y
        self.input_mask = input_mask
        self.pos2idx = pos2idx
        if load_parent_dir:
            print(f"Loading {name} from path ({load_parent_dir}/{name})")
            self.load(load_parent_dir)
        elif other_dir:
            print(f"Initializing {name} from other data ({other_dir})")
            self.load_orig(other_dir)
        elif raw_data_path:
            print(f"Initializing {name} from scratch ({raw_data_path})")
            self.read_raw_input(raw_data_path, max_sents, raw_data_enc)
        else:
            print(f"Initializing {name} from args only")
        print(self)

    def subtok_ratio(self, return_all=False, cls_sep_per_sent=2):
        """
        Subtokens per token, *ignoring [SEP]/[CLS]*!
        """
        n_toks_bert = self.n_toks_bert(cls_sep_per_sent)
        n_toks_orig = self.n_toks_orig()
        try:
            ratio = n_toks_bert / n_toks_orig
        except (TypeError, ZeroDivisionError):
            ratio = -1
        if return_all:
            return ratio, n_toks_bert, n_toks_orig
        return ratio

    def unk_ratio(self, return_all=False, cls_sep_per_sent=2):
        n_unks_bert = 0
        for sent in self.toks_bert:
            for tok in sent:
                if tok == "[UNK]":
                    n_unks_bert += 1
        n_toks_bert = self.n_toks_bert(cls_sep_per_sent)
        try:
            ratio = n_unks_bert / n_toks_bert
        except (TypeError, ZeroDivisionError):
            ratio = -1
        if return_all:
            return ratio, n_unks_bert, n_toks_bert
        return ratio

    def n_toks_orig(self):
        try:
            return sum(len(sent_toks) for sent_toks in self.toks_orig)
        except TypeError:
            return None

    def n_toks_bert(self, cls_sep_per_sent=2):
        try:
            return sum(len(sent_toks) - cls_sep_per_sent
                       for sent_toks in self.toks_bert)
        except TypeError:
            return None

    def pos_orig_distrib(self):
        c = Counter([pos for sent in self.pos_orig for pos in sent])
        total = sum(c.values())
        return sorted([(key, cnt / total) for key, cnt in c.items()],
                      key=lambda x: -x[1])

    def idx2pos(self):
        return {self.pos2idx[pos]: pos for pos in self.pos2idx}

    def pos_y_distrib(self):
        if self.y is None:
            return None
        idx2pos = self.idx2pos()
        dummy_idx = self.dummy_idx()
        c = Counter([idx2pos[int(pos)] for sent in self.y
                     for pos in sent if int(pos) != dummy_idx])
        total = sum(c.values())
        return sorted([(key, cnt / total) for key, cnt in c.items()],
                      key=lambda x: -x[1])

    @staticmethod
    def visualize(matrix, name):
        plt.clf()
        plt.pcolormesh(matrix)
        plt.savefig(f"../figs/{name}.png")

    def tensor_dataset(self):
        return TensorDataset(
            torch.Tensor(self.x).to(torch.int64),
            torch.Tensor(self.input_mask).to(torch.int64),
            torch.Tensor(self.y).to(torch.int64))

    def alphabet(self):
        return {c for sent in self.toks_orig for tok in sent for c in tok
                if c != ' '}

    def dummy_idx(self):
        if self.pos2idx:
            return self.pos2idx[DUMMY_POS]
        return None

    # ---- Adding noise ----

    def read_raw_input(self, filename, max_sents, encoding, verbose=True):
        self.toks_orig, self.pos_orig = read_raw_input(
            filename, max_sents, encoding, verbose)

    def add_noise(self, noise_type, noise_lvl_min,
                  noise_lvl_max, target_alphabet):
        if noise_type == 'add_random_noise':
            self.add_random_noise(noise_lvl_min, noise_lvl_max,
                                  target_alphabet)
        elif noise_type == 'add_custom_noise_general':
            self.add_custom_noise_general(noise_lvl_min, noise_lvl_max,
                                          target_alphabet)
        elif noise_type == 'add_custom_noise_gsw':
            self.add_custom_noise_gsw(noise_lvl_min, noise_lvl_max)

    def add_random_noise(self, noise_lvl_min, noise_lvl_max, target_alphabet,
                         noise=["add_char", "delete_char", "replace_char"]):
        """
        Aepli & Sennrich 2022
        """
        toks_noisy = []
        n_changed = 0
        for sent_toks in self.toks_orig:
            percentage_noisy = self.percentage_noisy(noise_lvl_min,
                                                     noise_lvl_max)
            idx_noisy = random.sample(
                range(len(sent_toks)),
                k=round(percentage_noisy * len(sent_toks)))
            sent_toks_noisy = []
            for i, tok in enumerate(sent_toks):
                if i in idx_noisy:
                    sent_toks_noisy.append(
                        getattr(self, random.sample(noise, 1)[0])(
                            tok, target_alphabet))
                    n_changed += 1
                else:
                    sent_toks_noisy.append(tok)
            toks_noisy.append(sent_toks_noisy)
        print(f"Modified {n_changed} tokens.")
        self.toks_orig = toks_noisy

    @staticmethod
    def add_char(word, alphabet):
        idx = random.randrange(-1, len(word))
        if idx == -1:
            return random.sample(alphabet, 1)[0] + word
        return word[:idx + 1] + random.sample(alphabet, 1)[0] + word[idx + 1:]

    @staticmethod
    def delete_char(word, alphabet):
        idx = random.randrange(len(word))
        return word[:idx] + word[idx + 1:]

    @staticmethod
    def replace_char(word, alphabet, idx=-1):
        if idx < 0:
            idx = random.randrange(len(word))
        return word[:idx] + random.sample(alphabet, 1)[0] + word[idx + 1:]

    @staticmethod
    def percentage_noisy(noise_lvl_min, noise_lvl_max):
        if noise_lvl_max - noise_lvl_min < 0.01:
            return noise_lvl_min
        return random.randrange(round(100 * noise_lvl_min),
                                round(100 * noise_lvl_max)) / 100

    def add_custom_noise_general(self, noise_lvl_min, noise_lvl_max,
                                 target_alphabet):
        """
        Aepli & Sennrich 2022 / Aepli, personal correspondence
        """
        vowels, consonants = [], []
        for c in target_alphabet:
            if self.is_vowel(c):
                vowels.append(c)
            elif self.is_consonant(c):
                consonants.append(c)
            # ignore punctuation etc.
        toks_noisy = []
        vowel2umlaut = {"a": "ä", "o": "ö", "u": "ü",
                        "A": "Ä", "O": "Ö", "U": "Ü"}
        voiced2voiceless = {"b": "p", "d": "t", "g": "k",
                            "B": "P", "D": "T", "G": "K"}
        n_changed = 0
        for sent_toks in self.toks_orig:
            percentage_noisy = self.percentage_noisy(noise_lvl_min,
                                                     noise_lvl_max)
            idx_noisy = random.sample(
                range(len(sent_toks)),
                k=round(percentage_noisy * len(sent_toks)))
            sent_toks_noisy = []
            for i, tok in enumerate(sent_toks):
                if i in idx_noisy:
                    idx = random.randrange(len(tok))
                    if self.is_vowel(tok[idx]):
                        if tok[idx] in vowel2umlaut:
                            if random.randrange(2) > 0:
                                # V:Umlaut
                                tok_noisy = tok[:idx] + \
                                    vowel2umlaut[tok[idx]] + tok[idx + 1:]
                            else:
                                # V:V
                                tok_noisy = self.replace_char(
                                    tok, vowels, idx)
                        else:
                            tok_noisy = self.replace_char(tok, vowels, idx)
                        n_changed += 1
                    elif self.is_consonant(tok[idx]):
                        if tok[idx] in voiced2voiceless:
                            if random.randrange(2) > 0:
                                # voiced:voiceless
                                tok_noisy = tok[:idx] + \
                                    voiced2voiceless[tok[idx]] + tok[idx + 1:]
                            else:
                                # C:C
                                tok_noisy = self.replace_char(
                                    tok, consonants, idx)
                        else:
                            tok_noisy = self.replace_char(tok, consonants, idx)
                        n_changed += 1
                    else:
                        tok_noisy = tok
                    sent_toks_noisy.append(tok_noisy)
                else:
                    sent_toks_noisy.append(tok)
            toks_noisy.append(sent_toks_noisy)
        print(f"Modified {n_changed} tokens.")
        self.toks_orig = toks_noisy

    def add_custom_noise_gsw(self, noise_lvl_min, noise_lvl_max):
        """
        Aepli & Sennrich 2022 / Aepli, personal correspondence
        """
        # TODO: k -> ch vs. (c)k -> gg (currenly: ck->gg, k->ch)
        deu2gsw_1 = {"ß": "ss", "gs": "x", "chen": "li", "lein": "li",
                     "ung": "ig"}
        deu2gsw_2 = {"ck": "gg"}
        deu2gsw_3 = {"ah": "aa", "eh": "ee", "ih": "ii", "oh": "oo",
                     "uh": "uu", "äh": "ää", "öh": "öö", "üh": "üü",
                     "ie": "ii", "ei": "ai", "k": "ch"}
        toks_noisy = []
        n_changed = 0
        for sent_toks in self.toks_orig:
            percentage_noisy = self.percentage_noisy(noise_lvl_min,
                                                     noise_lvl_max)
            idx_noisy = random.sample(
                range(len(sent_toks)),
                k=round(percentage_noisy * len(sent_toks)))
            sent_toks_noisy = []
            for i, tok in enumerate(sent_toks):
                if i in idx_noisy:
                    tok_noisy = tok.lower()
                    for deu in deu2gsw_1:
                        tok_noisy = tok_noisy.replace(deu, deu2gsw_1[deu])
                    for deu in deu2gsw_2:
                        tok_noisy = tok_noisy.replace(deu, deu2gsw_2[deu])
                    for deu in deu2gsw_3:
                        tok_noisy = tok_noisy.replace(deu, deu2gsw_3[deu])
                    # swap ä and e:
                    tok_noisy = "".join([c if c not in "eä"
                                         else "ä" if c == "e"
                                         else "e" for c in tok_noisy])
                    # non-circular o:u / eu:oi
                    tok_noisy = tok_noisy.replace("eu", "OI")
                    tok_noisy = tok_noisy.replace("o", "u")
                    tok_noisy = tok_noisy.replace("OI", "oi")
                    if tok[0] == tok[0].upper():
                        if tok == tok.upper():
                            tok_noisy = tok_noisy.upper()
                        else:
                            tok_noisy = tok_noisy[0].upper() + tok_noisy[1:]
                    n_changed += 1
                    sent_toks_noisy.append(tok_noisy)
                else:
                    sent_toks_noisy.append(tok)
            toks_noisy.append(sent_toks_noisy)
        print(f"Modified {n_changed} tokens.")
        self.toks_orig = toks_noisy

    @staticmethod
    def is_vowel(c):
        # remove accents
        return Data.is_vowel_clean(unicodedata.normalize("NFKD", c)[0])

    @staticmethod
    def is_vowel_clean(c):
        return c.lower() in "aeiou"

    @staticmethod
    def is_consonant(c):
        # remove accents
        return unicodedata.normalize("NFKD", c)[0].lower() in \
            "qwrtypsdfghjklzxcvbnm"
        # Note that this selection of consonants is still language-specific
        # (Latin alphabet, "y")

    # ---------------------------
    # --- Matrix preparations ---

    def prepare_xy(self, tokenizer, T,
                   # Which subtoken of a token should be used to represent
                   # the token during the evaluation?
                   # 'first', 'last' or 'all'
                   subtoken_rep,
                   verbose=True):
        assert T >= 2
        N = len(self.toks_orig)
        self.x = np.zeros((N, T), dtype=np.float64)
        self.toks_bert, pos = [], []
        self.input_mask = np.zeros((N, T))
        cur_toks, cur_pos = [], []
        for i, (sent_toks, sent_pos) in enumerate(
                zip(self.toks_orig, self.pos_orig)):
            cur_toks = ["[CLS]"]
            cur_pos = [DUMMY_POS]  # CLS
            for token, pos_tag in zip(sent_toks, sent_pos):
                subtoks = tokenizer.tokenize(token)
                cur_toks += subtoks
                if subtoken_rep == 'first':
                    cur_pos += [pos_tag]
                    cur_pos += [DUMMY_POS for _ in range(1, len(subtoks))]
                elif subtoken_rep == 'last':
                    cur_pos += [DUMMY_POS for _ in range(1, len(subtoks))]
                    cur_pos += [pos_tag]
                else:
                    cur_pos += [pos_tag for _ in range(len(subtoks))]
            cur_toks = cur_toks[:T - 1] + ["[SEP]"]
            self.toks_bert.append(cur_toks)
            self.input_mask[i][:len(cur_toks)] = len(cur_toks) * [1]
            self.x[i][:len(cur_toks)] = tokenizer.convert_tokens_to_ids(
                cur_toks)
            cur_pos = (cur_pos[:T - 1]
                       + [DUMMY_POS]  # SEP
                       + (T - len(cur_pos) - 1) * [DUMMY_POS]  # padding
                       )
            pos.append(cur_pos)
            if verbose and i % 1000 == 0:
                print(i)
        if not self.pos2idx:
            # BERT doesn't want labels that are already onehot-encoded
            self.pos2idx = {tag: idx for idx, tag in enumerate(
                {tok_pos for sent_pos in pos for tok_pos in sent_pos})}
        self.y = np.empty((N, T))
        for i, sent_pos in enumerate(pos):
            try:
                self.y[i] = [self.pos2idx[tok_pos] for tok_pos in sent_pos]
            except KeyError as e:
                print("Encounted unknown POS tag:", e)
                print([(tok, pos) for tok, pos in zip(
                    self.toks_bert[i], sent_pos)])
                sys.exit(1)
        assert len(self.toks_bert) == self.x.shape[0] == len(pos), \
            f"{len(self.toks_bert)} == {self.x.shape[0]} == {len(pos)}"
        if verbose:
            print(f"{len(self.toks_bert)} sentences")
            print(f"{' '.join(self.toks_orig[0])}")
            print(f"{' '.join(self.toks_bert[0])}")
            # for i in zip(self.toks_bert[0], self.x[0], pos[0], self.y[0],
            #              self.input_mask[0], self.real_pos[0]):
            #     print(i)
            if N > 1:
                print(f"{' '.join(self.toks_orig[1])}")
                print(f"{' '.join(self.toks_bert[1])}")
                if N > 2:
                    print(f"{' '.join(self.toks_orig[2])}")
                    print(f"{' '.join(self.toks_bert[2])}")

    # ----------------------
    # --- Saving/loading ---

    def save(self, parent_dir='../data/', save_orig=True):
        if not os.path.exists(parent_dir):
            os.mkdir(parent_dir)
        directory = os.path.join(parent_dir, self.name)
        if not os.path.exists(directory):
            os.mkdir(directory)
        print(f"Saving {self.name} to {directory}")
        # Also works with None args:
        np.savez(os.path.join(directory, "arrays.npz"),
                 x=self.x, y=self.y, input_mask=self.input_mask)
        if save_orig:
            self.save_tsv(self.toks_orig, directory, "toks_orig.tsv")
            self.save_tsv(self.pos_orig, directory, "pos_orig.tsv")
        self.save_tsv(self.toks_bert, directory, "toks_bert.tsv")
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
        try:
            self.y = npzfile["y"]
        except ValueError:
            print("Couldn't load 'y'")
        try:
            self.input_mask = npzfile["input_mask"]
        except ValueError:
            print("Couldn't load 'input_mask'")
        self.toks_bert = self.tsv2list(directory, "toks_bert.tsv")
        self.load_orig(directory)

    def load_orig(self, other_dir):
        self.toks_orig = self.tsv2list(other_dir, "toks_orig.tsv")
        self.pos_orig = self.tsv2list(other_dir, "pos_orig.tsv")
        self.load_pos2idx(os.path.join(other_dir, "pos2idx.tsv"))

    def load_pos2idx(self, path):
        try:
            with open(path, encoding='utf8') as f:
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
            f"pos2idx={'None' if self.pos2idx is None else len(self.pos2idx)})"

    def copy_toks_orig(self):
        return copy.deepcopy(self.toks_orig)

    def copy_pos_orig(self):
        return copy.deepcopy(self.pos_orig)
    # ------------------------
