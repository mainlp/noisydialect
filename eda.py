import cust_logger

from argparse import ArgumentParser
import collections
import os
import sys

from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DUMMY_POS = "<DUMMY>"

STTS_TIGER = {"ADJA", "ADJD", "ADV", "APPR", "APPRART", "APPO", "APZR", "ART",
              "CARD", "FM", "ITJ", "KOUI", "KOUS", "KON", "KOKOM", "NN", "NE",
              "PDS", "PDAT", "PIS", "PIAT", "PPER", "PPOSS", "PPOSAT", "PRELS",
              "PRELAT", "PRF", "PWS", "PWAT", "PWAV", "PROAV", "PTKZU",
              "PTKNEG", "PTKVZ", "PTKANT", "PTKA", "TRUNC", "VVFIN", "VVIMP",
              "VVINF", "VVIZU", "VVPP", "VAFIN", "VAIMP", "VAINF", "VAPP",
              "VMFIN", "VMINF", "VMPP", "XY", "$,", "$.", "$("}


def read_input_data(filename, tokenizer, verbose=True, encoding='utf8'):
    if verbose:
        print("Reading data from " + filename)
    with open(filename, encoding=encoding) as f_in:
        toks, pos = [], []  # n_sents * sent_len
        cur_toks, cur_pos = [], []
        counter = 0
        for line in f_in:
            line = line.strip()
            if not line:
                if cur_toks:
                    toks.append(cur_toks)
                    pos.append(cur_pos)
                    cur_toks, cur_pos, cur_starts = [], [], []
                    counter += 1
                    if verbose and counter % 5000 == 0:
                        print(counter)
                continue
            *words, word_pos = line.split()
            word_toks = []
            for word in words:
                word_toks += tokenizer.tokenize(word)
            cur_toks += word_toks
            cur_pos.append(word_pos)
            cur_pos += [DUMMY_POS for _ in range(1, len(word_toks))]
            cur_starts.append(1)
            cur_starts += [0 for _ in range(1, len(word_toks))]
        if cur_toks:
            toks.append(cur_toks)
            pos.append(cur_pos)
    assert len(toks) == len(pos), f"{len(toks)} == {len(pos)}"
    if verbose:
        print(f"{len(toks)} sentences")
        for i in zip(toks[0], pos[0]):
            print(i)
        print("\n")
    return toks, pos


def print_toks(tokenizer, sent):
    print(sent)
    num_toks, num_words = 0, 0
    for word in sent.split():
        num_words += 1
        toks = tokenizer.tokenize(word)
        num_toks += len(toks)
        print(toks)
    print(f"{num_toks} / {num_words} = {num_toks / num_words:.2f}")


def pos_stats(nested_labels, max_len=None):
    counter = collections.Counter()
    total = 0
    for sent_pos in nested_labels:
        if max_len:
            sent_pos = sent_pos[:max_len]
        counter.update(sent_pos)
        total += len([p for p in sent_pos if p != DUMMY_POS])
    total_with_dummy = total + counter[DUMMY_POS]
    print(f"----\tTOTAL_D\t1.00\t{total_with_dummy:>7}")
    print(f"1.00\tTOTAL\t{total/total_with_dummy:.2f}\t{total:>7}")
    pos_rel = dict()
    for pos, abs in counter.most_common():
        print(("----" if pos == DUMMY_POS else f"{abs/total:.2f}")
              + f"\t{pos}\t{abs/total_with_dummy:.2f}\t{abs:>7}")
        if pos != DUMMY_POS:
            pos_rel[pos] = abs / total
    pos_set = set(counter.keys())
    print(f"Tags not in STTS-TIGER: {pos_set - STTS_TIGER - {DUMMY_POS}}")
    print(f"STTS-TIGER tags not in data: {STTS_TIGER - pos_set}")
    return pos_set, pos_rel


def length_stats(tokens, other_threshold=None):
    sent_lens = np.asarray([len(sent) for sent in tokens])
    print(f"Min: {np.amin(sent_lens)}")
    print(f"Max: {np.amax(sent_lens)}")
    print(f"Mean: {np.mean(sent_lens)}")
    print(f"Std: {np.std(sent_lens)}")
    threshold = round(np.mean(sent_lens) + 2 * np.std(sent_lens))
    print(f"Mean + 2 std: {threshold} ({sent_lens[np.where(sent_lens <= threshold)].size / sent_lens.size:.2f})")
    if other_threshold:
        print(f"{other_threshold} ({sent_lens[np.where(sent_lens <= other_threshold)].size / sent_lens.size:.2f})")
    return sent_lens, threshold


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-r", "--train", dest="train_file",
                        help="training data file",
                        default="data/hamburg-dependency-treebank/"
                                "train_DHT_STTS.txt")
    parser.add_argument("-e", "--test", dest="test_file",
                        help="test data file",
                        default="data/NOAH-corpus/test_GSW_STTS.txt")

    args = parser.parse_args()
    sys.stdout = cust_logger.Logger("eda")
    for arg in vars(args):
        print(arg, getattr(args, arg))

    tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-german-cased")
    train_toks, train_pos = read_input_data(args.train_file, tokenizer)
    test_toks, test_pos = read_input_data(args.test_file, tokenizer)

    print()
    print_toks(tokenizer, "Mit der Eroberung vom Aargau durch die alte "
                          "Eidgenossenschaft 1415 ist Baden der Sitz des "
                          "Landvogts der Grafschaft Baden geworden und "
                          "viele Tagsatzungen haben hier auch stattgefunden .")
    print_toks(tokenizer, "Mit der Eroberung vom Aargau durch die alte "
                          "Eidgenossenschaft 1415 wurde Baden der Sitz des "
                          "Landvogts der Grafschaft Baden und "
                          "viele Tagsatzungen fanden hier auch statt .")
    print_toks(tokenizer, "Mit de Eroberig vom Aargau durch di alti "
                          "Eidgnosseschaft im 1415i isch Bade de Sitz vom "
                          "Landvogt vo de Grafschaft Bade worde und "
                          "au vili Tagsatzige hei hiir schtattgfunde .")
    print()
    print_toks(tokenizer, "Die Gemeindeversammlung von Neuenhof hat , wie "
                          "auch der Einwohnerrat von Baden am 30. März 2010 "
                          "der geplanten Fusion mit der Stadt Baden zugestimmt"
                          " , welche zum 1. Januar 2012 realisiert werden "
                          "sollte .")
    print_toks(tokenizer, "d Gmeindsversammlig vo Noiehof het , wi au de "
                          "Iiwohnerrot vo Bade am 30. März 2010 de plante "
                          "Fusion mit de Schtadt Bade zugschtimmt , wo uf de "
                          "1. Jänner 2012 het soll realisiirt werde .")
    print()

    print("TRAINING")
    train_pos_set, train_pos_ratio = pos_stats(train_pos)
    print("\nTEST")
    test_pos_set, test_pos_ratio = pos_stats(test_pos)

    print(f"Tags only in train, not test: {train_pos_set - test_pos_set}")

    assert test_pos_set.issubset(train_pos_set), \
        f"Only in the test set: {test_pos_set - train_pos_set}"

    df = pd.DataFrame([train_pos_ratio, test_pos_ratio],
                      index=["Training (DHT)", "Test (NOAH)"])
    df = df.transpose()

    plt.rcParams["figure.figsize"] = (20, 5)
    df.plot(kind='bar')
    plt.tight_layout()
    # plt.show()
    if not os.path.exists("figs"):
        os.mkdir("figs")
    plt.savefig("figs/pos_all.png")
    plt.clf()

    print()
    print("TRAINING")
    train_lens, train_threshold = length_stats(train_toks)
    print("\nTEST")
    test_lens, _ = length_stats(test_toks, train_threshold)
    print()

    bins = np.linspace(0, 150, 31)
    plt.hist(train_lens, bins, density=True, alpha=0.5,
             label="Training (DHT)")
    plt.hist(test_lens, bins, density=True, alpha=0.5,
             label="Test (NOAH)")

    plt.grid(True)
    plt.legend()
    # plt.show()
    plt.savefig("figs/sentence_lengths.png")
    plt.clf()

    print(f"TRAINING (max len {train_threshold})")
    train_pos_set_thr, train_pos_ratio_thr = pos_stats(
        train_pos, train_threshold)
    print(f"\nTEST (max len {train_threshold})")
    test_pos_set_thr, test_pos_ratio_thr = pos_stats(test_pos, train_threshold)
    print(f"Tags only in train, not test: {train_pos_set_thr - test_pos_set_thr}")
    assert test_pos_set_thr.issubset(train_pos_set_thr), \
        f"Only in the test set: {test_pos_set_thr - train_pos_set_thr}"

    df = pd.DataFrame([train_pos_ratio_thr, test_pos_ratio_thr],
                      index=[f"Training (DHT, max len: {train_threshold})",
                             f"Test (NOAH, max len: {train_threshold})"])
    df = df.transpose()
    plt.rcParams["figure.figsize"] = (20, 5)
    df.plot(kind='bar')
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"figs/pos_maxlen{train_threshold}.png")
    plt.clf()
