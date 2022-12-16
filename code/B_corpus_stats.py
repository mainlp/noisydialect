from data import Data
import sys


def corpus_stats(train, dev, out, tokenizer, access_mode_out):
    with open(out, access_mode_out, encoding="utf8") as f_out:
        if access_mode_out[0] == "w":
            f_out.write("\t".join("DATASET", "TOKENIZER" "NOISE_TYPE",
                                  "NOISE_LVL", "N_SENTS", "MIN_TOKS_PER_SENT",
                                  "MAX_TPS", "MEAN_TPS", "STD_TPS",
                                  "MEAN+STD", "MEAN+1.5STD", "MEAN+2STD"))
            f_out.write("\n")
        alphabet = {}
        n_sents = -1
        for infile in (train, dev):
            d = Data("dummy_name", infile)
            if infile == train:
                alphabet = d.alphabet()
                n_sents = len(d.toks_orig)
            min_len, max_len, mean, std = d.tokeinzation_info(tokenizer,
                                                              verbose=False)
            f_out.write("\t".join(infile, tokenizer, "-", "-", n_sents,
                                  min_len, max_len, mean, std, mean + std,
                                  mean + 1.5 * std, mean + 2 * std))
            f_out.write("\n")
            for noise_lvl in (0.15, 0.35, 0.55, 0.75, 0.95):
                d = Data("dummy_name", infile)
                d.add_random_noise(noise_lvl, alphabet)
                min_len, max_len, mean, std = d.tokeinzation_info(
                    tokenizer, verbose=False)
                f_out.write("\t".join(infile, tokenizer, "-", "-", n_sents,
                                      min_len, max_len, mean, std, mean + std,
                                      mean + 1.5 * std, mean + 2 * std))
                f_out.write("\n")


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("USAGE:")
        print("python B_corpus_stats.py TRAIN_FILE DEV_FILE OUT_FILE "
              "TOKENIZER_NAME [OUT_FILE_ACCESSMODE]")
    try:
        access_mode = sys.argv[5]
    except IndexError:
        append_mode = "w+"
    corpus_stats(*sys.argv[1:5], access_mode_out)
