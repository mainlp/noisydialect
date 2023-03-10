from C_data import Data

import sys

from transformers import AutoTokenizer


def corpus_stats(files, out, tokenizer_name, access_mode_out):
    with open(out, access_mode_out, encoding="utf8") as f_out:
        if access_mode_out[0] == "w":
            f_out.write("\t".join(("DATASET", "TOKENIZER", "N_SENTS",
                                   "MIN_TOKS_PER_SENT", "MAX_TPS", "MEAN_TPS",
                                   "STD_TPS", "MEAN+STD", "MEAN+1.5STD",
                                   "MEAN+2STD")))
            f_out.write("\n")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        n_sents = -1
        for infile in files.split(","):
            d = Data("dummy_name", raw_data_path=infile)
            n_sents = len(d.toks_orig)
            min_len, max_len, mean, std = d.tokenization_info(
                tokenizer, verbose=False)
            f_out.write("\t".join((str(x) for x in (
                infile, tokenizer_name, n_sents,
                min_len, max_len, mean, std, mean + std, mean + 1.5 * std,
                mean + 2 * std))))
            f_out.write("\n")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("USAGE:")
        print("python B_corpus_stats.py DATA_FILES* OUT_FILE "
              "TOKENIZER_NAME [OUT_FILE_ACCESSMODE]")
        print("(*Comma-separted file names)")
    try:
        access_mode_out = sys.argv[4]
    except IndexError:
        access_mode_out = "w+"
    corpus_stats(*sys.argv[1:4], access_mode_out)
