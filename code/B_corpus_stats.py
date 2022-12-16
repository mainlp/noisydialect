from data import Data

import sys

from transformers import AutoTokenizer


def corpus_stats(train, devs, out, tokenizer_name, access_mode_out):
    with open(out, access_mode_out, encoding="utf8") as f_out:
        if access_mode_out[0] == "w":
            f_out.write("\t".join(("DATASET", "TOKENIZER", "NOISE_TYPE",
                                   "NOISE_LVL", "N_SENTS", "MIN_TOKS_PER_SENT",
                                   "MAX_TPS", "MEAN_TPS", "STD_TPS",
                                   "MEAN+STD", "MEAN+1.5STD", "MEAN+2STD",
                                   "SAMPLE_SENTENCE")))
            f_out.write("\n")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        alphabet = {}
        n_sents = -1
        for infile in [train] + devs.split(","):
            print("Analyzing " + infile)
            for noise_lvl in ("-", 0.15, 0.35, 0.55, 0.75, 0.95):
                print("Noise: " + str(noise_lvl))
                d = Data("dummy_name", raw_data_path=infile)
                if noise_lvl == "-":
                    noise_type = "-"
                    if infile == train:
                        alphabet = d.alphabet()
                        n_sents = len(d.toks_orig)
                else:
                    d.add_random_noise(noise_lvl, alphabet)
                    noise_type = "add_random_noise"
                min_len, max_len, mean, std = d.tokenization_info(
                    tokenizer, verbose=False)
                sample_sentence = " ".join(
                    d.tokenize_sample_sentence(tokenizer)).replace('"', '\\"')
                f_out.write("\t".join((str(x) for x in (
                    infile, tokenizer_name, noise_type, noise_lvl, n_sents,
                    min_len, max_len, mean, std, mean + std, mean + 1.5 * std,
                    mean + 2 * std, sample_sentence))))
                f_out.write("\n")


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("USAGE:")
        print("python B_corpus_stats.py TRAIN_FILE DEV_FILE(s)* OUT_FILE "
              "TOKENIZER_NAME [OUT_FILE_ACCESSMODE]")
        print("(*Comma-separted dev files)")
    try:
        access_mode_out = sys.argv[5]
    except IndexError:
        access_mode_out = "w+"
    corpus_stats(*sys.argv[1:5], access_mode_out)
