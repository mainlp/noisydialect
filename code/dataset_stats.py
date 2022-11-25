from glob import glob

import numpy as np
import pandas as pd
import seaborn
from scipy.stats import entropy

from config import Config
from data import Data, DUMMY_POS


def write_dataset_stats(tagset, out_file, data_folder="../data/"):
    with open(out_file, "w+", encoding="utf8") as f_out:
        f_out.write("DATASET\tN_SENTS\tMAX_SENT_TOKS\tSUBTOKS_PER_TOK\t"
                    "UNKS_PER_SUBTOK\tLABEL_DISTRIBUTION\n")
        for path in glob(data_folder + "*" + tagset):
            data = Data(name=path.split("/")[-1], load_parent_dir=data_folder)
            infos = [data.name, *data.x.shape, data.subtok_ratio(),
                     data.unk_ratio(), data.pos_y_distrib()]
            print(infos)
            f_out.write("\t".join([str(info) for info in infos]))
            f_out.write("\n")
        setup2instantiations = {}
        for path in glob(data_folder + "*" + tagset + "_[0-9]*[0-9]"):
            print(path)
            _, _, setup_and_inst = path.rpartition("/")
            setup, _, inst = setup_and_inst.rpartition("_")
            data = Data(name=path.split("/")[-1], load_parent_dir=data_folder)
            if setup not in setup2instantiations:
                setup2instantiations[setup] = [data]
            else:
                setup2instantiations[setup].append(data)
        for setup in setup2instantiations:
            instantiations = setup2instantiations[setup]
            n_inst = len(instantiations)
            (n_sents, sent_len) = instantiations[0].x.shape
            subtok_ratio = sum(
                [inst.subtok_ratio() for inst in instantiations]) / n_inst
            unk_ratio = sum(
                [inst.unk_ratio() for inst in instantiations]) / n_inst
            pos_y_distrib = instantiations[0].pos_y_distrib()
            infos = [setup, n_sents, sent_len, subtok_ratio, unk_ratio,
                     pos_y_distrib]
            print(infos)
            f_out.write("\t".join([str(info) for info in infos]))
            f_out.write("\n")


def tagset_order(tagset_file):
    tagset_order = []
    with open(tagset_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tagset_order.append(line)
    tagset_order.remove(DUMMY_POS)
    return tagset_order


def read_dataset_stats(stats_file, name2subtoks, name2unks,
                       name2label_distrib, tagset_order):
    with open(stats_file) as f_in:
        header_skipped = False
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            if not header_skipped:
                header_skipped = True
                continue
            cells = line.split("\t")
            name = cells[0]
            subtoks_per_tok = float(cells[3])
            unks_per_subtok = float(cells[4])
            name2subtoks[name] = subtoks_per_tok
            name2unks[name] = unks_per_subtok
            label_distrib = {}
            for e in cells[5][2:-2].split("), ("):
                key_val = e[1:].replace("'", "").split(", ")
                label_distrib[key_val[0]] = float(key_val[1])
            name2label_distrib[name] = [
                label_distrib.get(tag, 0.0) for tag in tagset_order]
    return name2subtoks, name2unks, name2label_distrib


def average_scores(directory, data2metric2scores):
    # Average scores across initializations *and* epochs
    if directory.endswith("/"):
        directory = directory[:-1]
    _, dir_name = directory.rsplit("/", 1)
    config = Config()
    config.load(directory + "/" + dir_name + ".cfg")
    for res_file in glob(f"{directory}/results*.tsv"):
        if res_file.endswith("AVG.tsv"):
            continue
        with open(res_file) as f:
            for line in f:
                line = line.strip()
                if (line.startswith("test") or line.startswith("dev")
                    or line.startswith("val")) \
                        and not line.endswith("loss"):
                    data_metric_epoch, score = line.split("\t")
                    data, metric, _ = data_metric_epoch.rsplit("_", 2)
                    data = config.name_train + "+" + data
                    if data not in data2metric2scores:
                        data2metric2scores[data] = {}
                    scores = data2metric2scores[data].get(metric, [])
                    scores.append(float(score))
                    data2metric2scores[data][metric] = scores
    return data2metric2scores


def data2col(target_data):
    if "hdt" in target_data:
        return 0
    if "alpino" in target_data:
        return 1
    if "noah" in target_data or "uzh" in target_data:
        return 2
    if "lsdc" in target_data:
        return 3
    return 4


if __name__ == "__main__":
    write_dataset_stats("stts", "../results/data_statistics_stts.tsv")
    write_dataset_stats("upos", "../results/data_statistics_upos.tsv")

    upos_order = tagset_order("../datasets/tagset_upos.txt")
    stts_order = tagset_order("../datasets/tagset_stts.txt")
    name2subtoks, name2unks, name2label_distrib = read_dataset_stats(
        "../results/data_statistics_stts.tsv", {}, {}, {}, stts_order)
    name2subtoks, name2unks, name2label_distrib = read_dataset_stats(
        "../results/data_statistics_upos.tsv", name2subtoks, name2unks,
        name2label_distrib, upos_order)

    data2metric2scores = {}
    for d in glob("../results/*upos"):
        data2metric2scores = average_scores(d, data2metric2scores)
    for d in glob("../results/*stts"):
        data2metric2scores = average_scores(d, data2metric2scores)

    data2metric2avg = {}
    for data in data2metric2scores:
        data2metric2avg[data] = {}
        for metric in data2metric2scores[data]:
            scores = data2metric2scores[data][metric]
            data2metric2avg[data][metric] = sum(scores) / len(scores)

    setups = [data for data in data2metric2avg]
    target_data = []
    source_data = []
    target_type = []
    target_corpus = []
    tagset = []
    for setup in setups:
        src, tgt = setup.split("+")
        source_data.append(src)
        target_data.append(tgt)
        tgt_splits = tgt.split(".")
        target_type.append(tgt_splits[0])
        target_corpus.append(tgt_splits[1])
        tagset.append(tgt_splits[-1])
    # TODO where do these NaNs come from
    subtoks_per_tok = [name2subtoks[tgt] if tgt in name2subtoks
                       else np.nan for tgt in target_data]
    unks_per_subtok = [name2unks[tgt] if tgt in name2subtoks
                       else np.nan for tgt in target_data]
    acc = [data2metric2avg[setup]["acc"] if setup in data2metric2avg
           else np.nan for setup in setups]
    f1_macro = [data2metric2avg[setup]["f1"] if setup in data2metric2avg
                else np.nan for setup in setups]
    run_df = pd.DataFrame(
        {"source_data": source_data,
         "target_data": target_data, "target_corpus": target_corpus,
         "target_type": target_type, "tagset": tagset,
         "subtoks_per_tok": subtoks_per_tok,
         "unks_per_subtok": unks_per_subtok,
         "acc": acc, "f1_macro": f1_macro,
         },
        index=setups)

    datasets = list({src for src in source_data}.union(
        {tgt for tgt in target_data}))
    data_df = pd.DataFrame(
        {"noise": [d.split(".", 4)[3] for d in datasets],
         "subtoks_per_tok": [name2subtoks[d] if d in name2subtoks
                             else np.nan for d in datasets],
         "unks_per_subtok": [name2unks[d] if d in name2unks
                             else np.nan for d in datasets],
         "label_distrib": [name2label_distrib[d] if d in name2label_distrib
                           else None for d in datasets],
         },
        index=datasets)

    run_df["source_noise"] = run_df.apply(
        lambda row: data_df.loc[row["source_data"]]["noise"],
        axis=1)

    run_df["dev"] = run_df.apply(
        lambda row: None if row["target_type"] == "dev"
        else row["source_data"] + "+" + row["source_data"].replace(
            "train", "dev"),
        axis=1)

    for indep_var in ("subtoks_per_tok", "unks_per_subtok"):
        run_df[indep_var + "_diff"] = run_df.apply(
            lambda row:
            row[indep_var] - data_df.loc[row["source_data"]][indep_var],
            axis=1)

    for metric in ("acc", "f1_macro"):
        run_df[metric + "_diff"] = run_df.apply(
            lambda row:
            row[metric] - run_df.loc[row["dev"]][metric]
            if row["dev"] in run_df.index else np.nan,
            axis=1)

    run_df["kullback_leibner"] = run_df.apply(
        lambda row:
        entropy(data_df.loc[row["source_data"]]["label_distrib"],
                data_df.loc[row["target_data"]]["label_distrib"]), axis=1)

    seaborn.scatterplot(run_df, x="subtoks_per_tok_diff", y="acc_diff",
                        hue="source_noise", style="target_corpus")
