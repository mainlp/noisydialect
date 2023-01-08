from glob import glob
import sys

from config import Config
from data import Data


def data_stats(input_pattern, out_file):
    saved_data_stats = {}
    with open(out_file, "w+", encoding="utf8") as f_out:
        f_out.write("\t".join((
            "SETUP_NAME", "TRAIN_SET", "DEV_SET",
            "SUBTOKEN_RATIO_TRAIN", "SUBTOKEN_RATIO_DEV",
            "SUBTOKEN_RATIO_DIFF",
            "UNK_RATIO_TRAIN", "UNK_RATIO_DEV", "UNK_RATIO_DIFF",
            "TTR_TRAIN", "TTR_DEV", "TTR_DIFF",
            "SPLIT_TOKEN_RATIO_TRAIN", "SPLIT_TOKEN_RATIO_DEV",
            "SPLIT_TOKEN_RATIO_DIFF",
            "F1_MACRO_AVG_DEV", "F1_MACRO_STD_DEV",
            "ACCURACY_AVG_DEV", "ACCURACY_STD_DEV",
        )))
        f_out.write("\n")
        folders = sorted(glob(input_pattern))
        for folder in folders:
            setup_name = folder.rsplit("/", 1)[1]
            config = Config()
            config.load(folder + "/" + setup_name + ".cfg")
            if config.noise_type is None:
                try:
                    train_data = Data(name=config.name_train,
                                      load_parent_dir="../data/",
                                      verbose=False)
                except FileNotFoundError:
                    print("!! Could not load " + config.name_train)
                    continue
                train_stats = (train_data.subtok_ratio(),
                               train_data.unk_ratio(),
                               train_data.type_token_ratio(),
                               train_data.split_token_ratio(
                                   subtoken_rep=config.subtoken_rep))
            else:
                # Get the averages of the different noise initializations
                train_stats = [0.0, 0.0, 0.0, 0.0]
                n_runs = 0
                success = True
                for seed in config.random_seeds:
                    try:
                        train_data = Data(
                            name=config.name_train + "_" + str(seed),
                            load_parent_dir="../data/", verbose=False)
                        train_stats[0] += train_data.subtok_ratio()
                        train_stats[1] += train_data.unk_ratio()
                        train_stats[2] += train_data.type_token_ratio()
                        train_stats[3] += train_data.split_token_ratio(
                            subtoken_rep=config.subtoken_rep)
                        n_runs += 1
                    except FileNotFoundError:
                        success = False
                        print("!! Could not load " + config.name_train)
                        break
                if not success:
                    continue
                for i in range(len(train_stats)):
                    train_stats[i] = train_stats[i] / n_runs
            dev_stats = []
            dev_names = config.name_dev.split(",")
            for cur_dev_name in dev_names:
                if cur_dev_name in saved_data_stats:
                    dev_stats.append(saved_data_stats[cur_dev_name])
                else:
                    cur_dev_data = Data(name=cur_dev_name,
                                        load_parent_dir="../data/",
                                        verbose=False)
                    cur_dev_stats = (cur_dev_data.subtok_ratio(),
                                     cur_dev_data.unk_ratio(),
                                     cur_dev_data.type_token_ratio(),
                                     cur_dev_data.split_token_ratio(
                                         subtoken_rep=config.subtoken_rep))
                    dev_stats.append(cur_dev_stats)
                    saved_data_stats[cur_dev_name] = cur_dev_stats
            scores = {}
            with open(folder + "/results_AVG.tsv") as f_in:
                for line in f_in:
                    line = line.strip()
                    if not line:
                        continue
                    cells = line.split("\t")
                    run_and_metric = cells[0]
                    if run_and_metric.endswith(str(config.n_epochs)):
                        for dev_name in dev_names:
                            if run_and_metric.startswith(dev_name):
                                metric = run_and_metric.rsplit("_", 2)[1]
                                try:
                                    scores[dev_name][metric] = (
                                        cells[1], cells[2])
                                except KeyError:
                                    scores[dev_name] = {
                                        metric: (cells[1], cells[2])}
                                break
            for cur_dev_name, cur_dev_stats in zip(dev_names, dev_stats):
                try:
                    cur_scores = scores[cur_dev_name]
                except KeyError:
                    print("!! No (reformatted) scores for " + cur_dev_name)
                    continue
                f_out.write("\t".join((setup_name, config.name_train,
                                       cur_dev_name)))
                f_out.write("\t")
                for i in range(len(train_stats)):
                    f_out.write("\t".join((
                        str(train_stats[i]), str(cur_dev_stats[i]),
                        str(cur_dev_stats[i] - train_stats[i]))))
                    f_out.write("\t")
                f_out.write("\t".join((*cur_scores["f1"], *cur_scores["acc"])))
                f_out.write("\n")
            print("-- Finished " + setup_name)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python data_stats.py INPUT_GLOB_PATTERN OUT_FILE")
        print("e.g. python data_stats.py ../results/C_nno* "
              "../results/stats-nno.tsv")
        sys.exit(1)

    data_stats(sys.argv[1], sys.argv[2])
