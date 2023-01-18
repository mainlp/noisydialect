from glob import glob
import sys

from transformers import AutoTokenizer

from config import Config
from data import Data


def print_line(f_out, setup_name, seed, name_train, target_name,
               train_stats, target_stats, f1, acc):
    f_out.write("\t".join((setup_name, str(seed), name_train, target_name)))
    f_out.write("\t")
    for i in range(len(train_stats)):
        f_out.write("\t".join((
            str(train_stats[i]), str(target_stats[i]),
            str(target_stats[i] - train_stats[i]))))
        f_out.write("\t")
    f_out.write("\t".join((str(target_stats[-i]) for i in range(4, 0, -1))))
    f_out.write("\t")
    f_out.write("\t".join((f1, acc)))
    f_out.write("\n")


def data_stats(input_pattern, out_file):
    saved_data_stats = {}
    with open(out_file, "w+", encoding="utf8") as f_out:
        f_out.write("\t".join((
            "SETUP_NAME", "SEED", "TRAIN_SET", "TARGET_SET",
            "SUBTOKEN_RATIO_TRAIN", "SUBTOKEN_RATIO_TARGET",
            "SUBTOKEN_RATIO_DIFF",
            "UNK_RATIO_TRAIN", "UNK_RATIO_TARGET", "UNK_RATIO_DIFF",
            "TTR_TRAIN", "TTR_TARGET", "TTR_DIFF",
            "SPLIT_TOKEN_RATIO_TRAIN", "SPLIT_TOKEN_RATIO_TARGET",
            "SPLIT_TOKEN_RATIO_DIFF",
            "TARGET_SUBTOKS_IN_TRAIN", "TARGET_SUBTOK_TYPES_IN_TRAIN",
            "TARGET_WORD_TOKENS_IN_TRAIN", "TARGET_WORD_TYPES_IN_TRAIN",
            "F1_MACRO", "ACCURACY",
        )))
        f_out.write("\n")
        folders = sorted(glob(input_pattern))
        for folder in folders:
            setup_name = folder.rsplit("/", 1)[1]
            config = Config()
            config.load(folder + "/" + setup_name + ".cfg")
            tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
            train_stats = None
            train_tok_counter = None
            train_word_counter = None
            if config.noise_type is None:
                try:
                    train_data = Data(name=config.name_train,
                                      load_parent_dir="../data/",
                                      verbose=False)
                except FileNotFoundError:
                    print("!! Could not load " + config.name_train)
                    continue
                train_data.calculate_toks_orig_cutoff(tokenizer, config.T)
                train_stats = (train_data.subtok_ratio(),
                               train_data.unk_ratio(),
                               train_data.type_token_ratio(),
                               train_data.split_token_ratio(
                                   subtoken_rep=config.subtoken_rep))
                train_tok_counter = train_data.subtok_counter()
                train_word_counter = train_data.word_counter()

            max_epoch = str(config.n_epochs)
            scores = {}
            for seed in config.random_seeds:
                seed_scores = {}
                with open(folder + "/results_"
                          + str(config.random_seeds[0]) + ".tsv") as f_in:
                    for line in f_in:
                        line = line.strip()
                        if not line:
                            continue
                        cells = line.split("\t")
                        target_name, metric, ep = cells[0].rsplit("_", 2)
                        if not (ep[-1] == "x" or ep.endswith(max_epoch)):
                            continue
                        try:
                            seed_scores[target_name][metric] = cells[1]
                        except KeyError:
                            seed_scores[target_name] = {metric: cells[1]}
                scores[seed] = seed_scores

            for target_name in scores[config.random_seeds[0]]:
                # Get the target data
                target_data = Data(name=target_name,
                                   load_parent_dir="../data/",
                                   verbose=False)
                target_data.calculate_toks_orig_cutoff(tokenizer, config.T)
                if target_name in saved_data_stats:
                    target_stats = saved_data_stats[target_name]
                else:
                    target_stats = [target_data.subtok_ratio(),
                                    target_data.unk_ratio(),
                                    target_data.type_token_ratio(),
                                    target_data.split_token_ratio(
                                        subtoken_rep=config.subtoken_rep)]
                    saved_data_stats[target_name] = target_stats

                if train_stats:
                    # The training data don't depend on the seed
                    cur_target_stats = []
                    for stat in target_stats:
                        cur_target_stats.append(stat)
                    subtoks_in_train, subtok_types_in_train =\
                        target_data.subtoks_present_in_other(train_tok_counter)
                    cur_target_stats.append(subtoks_in_train)
                    cur_target_stats.append(subtok_types_in_train)
                    words_in_train, word_types_in_train =\
                        target_data.words_present_in_other(train_word_counter)
                    cur_target_stats.append(words_in_train)
                    cur_target_stats.append(word_types_in_train)
                    for seed in config.random_seeds:
                        cur_scores = scores[seed][target_name]
                        print_line(f_out, setup_name, seed, config.name_train,
                                   target_name, train_stats, cur_target_stats,
                                   cur_scores["f1"], cur_scores["acc"])
                else:
                    # The training data depend on the seed
                    for seed in config.random_seeds:
                        train_data = Data(
                            name=config.name_train + "_" + str(seed),
                            load_parent_dir="../data/", verbose=False)
                        train_data.calculate_toks_orig_cutoff(tokenizer,
                                                              config.T)
                        train_stats = (train_data.subtok_ratio(),
                                       train_data.unk_ratio(),
                                       train_data.type_token_ratio(),
                                       train_data.split_token_ratio(
                                           subtoken_rep=config.subtoken_rep))
                        train_tok_counter = train_data.subtok_counter()
                        train_word_counter = train_data.word_counter()
                        cur_target_stats = []
                        for stat in target_stats:
                            cur_target_stats.append(stat)
                        subtoks_in_train, subtok_types_in_train =\
                            target_data.subtoks_present_in_other(
                                train_tok_counter)
                        cur_target_stats.append(subtoks_in_train)
                        cur_target_stats.append(subtok_types_in_train)
                        words_in_train, word_types_in_train =\
                            target_data.words_present_in_other(
                                train_word_counter)
                        cur_target_stats.append(words_in_train)
                        cur_target_stats.append(word_types_in_train)
                        cur_scores = scores[seed][target_name]
                        print_line(f_out, setup_name, seed, config.name_train,
                                   target_name, train_stats, cur_target_stats,
                                   cur_scores["f1"], cur_scores["acc"])
        print("-- Finished " + setup_name)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python data_stats.py INPUT_GLOB_PATTERN OUT_FILE")
        print("e.g. python data_stats.py ../results/C_nno* "
              "../results/stats-nno.tsv")
        sys.exit(1)

    data_stats(sys.argv[1], sys.argv[2])
