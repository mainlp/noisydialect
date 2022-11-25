from glob import glob

import matplotlib.pyplot as plt

from config import Config
from data import Data


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


def read_dataset_stats(stats_file, name2subtoks, name2unks):
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
    return name2subtoks, name2unks


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


def plot(metric, x_axis_dict, data2metric2avg, x_label, diff=False):
    x, y, c = [], [], []
    for data in data2metric2avg:
        # TODO randnoise runs
        src_data, target_data = data.split("+")
        try:
            if diff:
                x_i = x_axis_dict[target_data] - x_axis_dict[src_data]
            else:
                x_i = x_axis_dict[target_data]
            y_i = data2metric2avg[data][metric]
            x.append(x_i)
            y.append(y_i)
            c.append(data2col(target_data))
        except KeyError:
            print("Skipping " + data)

    scatter = plt.scatter(x, y, c=c, alpha=0.8)
    plt.legend(handles=scatter.legend_elements()[0],
               labels=["HDT (dev)", "Alpino (dev)", "NOAH/UZH", "LSDC"])
    if diff:
        x_label = "Difference in " + x_label[0].lower() + x_label[1:] +\
                  " (target - src)"
    plt.xlabel(x_label)
    plt.ylabel("Accuracy" if metric == "acc" else "F1 macro")
    plt.show()


if __name__ == "__main__":
    write_dataset_stats("stts", "../results/data_statistics_stts.tsv")
    write_dataset_stats("upos", "../results/data_statistics_upos.tsv")

    name2subtoks, name2unks = read_dataset_stats(
        "../results/data_statistics_stts.tsv", {}, {})
    name2subtoks, name2unks = read_dataset_stats(
        "../results/data_statistics_upos.tsv", name2subtoks, name2unks)

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

    for metric in ("acc", "f1"):
        for x_label in ("Subtokens per token", "UNKs per subtoken"):
            for diff in (True, False):
                x_axis_dict = name2subtoks if x_label[0] == "S" else name2unks
                plot(metric, x_axis_dict, data2metric2avg, x_label, diff)