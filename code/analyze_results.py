import glob
import numpy as np
import re
import sys

pattern = re.compile("(val|test)_(f1|acc)_epoch[0-9]+$")


def average_scores(directory):
    # Average scores across initializations
    scores_all = {}
    for res_file in glob.glob(f"{directory}/results*.tsv"):
        if res_file.endswith("AVG.tsv"):
            continue
        with open(res_file) as f:
            for line in f:
                line = line.strip()
                if (line.startswith("test") or line.startswith("val")) \
                        and not line.endswith("loss"):
                    metric, score = line.split("\t")
                    scores_for_metric = scores_all.get(metric, [])
                    scores_for_metric.append(float(score))
                    scores_all[metric] = scores_for_metric
    summary_file = f"{directory}/results_AVG.tsv"
    print("Writing average scores to " + summary_file)

    table_compatible = True
    for key in scores_all:
        if not pattern.match(key):
            table_compatible = False
            break

    if table_compatible:
        epoch2data2metric = {}
        max_epoch = -1
        for key in scores_all:
            data, metric, epochstr = key.split("_")
            epoch = int(epochstr[5:])
            if epoch > max_epoch:
                max_epoch = epoch
            if epoch not in epoch2data2metric:
                epoch2data2metric[epoch] = {}
            if data not in epoch2data2metric[epoch]:
                epoch2data2metric[epoch][data] = {}
            if metric not in epoch2data2metric[epoch][data]:
                epoch2data2metric[epoch][data][metric] = {}
            epoch2data2metric[epoch][data][metric] = scores_all[key]
        with open(summary_file, "w") as f_out:
            f_out.write("EPOCH\tF1 (TEST)\tSTDEV (F1 TEST)\t"
                        "ACC (TEST)\tSTDEV (ACC TEST)\t"
                        "F1 (VAL)\tSTDEV (F1 VAL)\t"
                        "ACC (VAL)\tSTDEV (ACC VAL)\n")
            for epoch in range(1, 1 + max_epoch):
                f_out.write(f"{epoch}")
                for data in ("test", "val"):
                    for metric in ("f1", "acc"):
                        scores = epoch2data2metric[epoch][data][metric]
                        n_runs = len(scores)
                        avg = sum(scores) / n_runs
                        stdev = np.std(scores)
                        f_out.write(f"\t{avg}\t{stdev}")
                f_out.write("\n")

    else:
        with open(summary_file, "w") as f_out:
            f_out.write("METRIC\tAVERAGE\tSTD_DEV\tN_RUNS\n")
            for metric in scores_all:
                scores = scores_all[metric]
                n_runs = len(scores)
                avg = sum(scores) / n_runs
                stdev = np.std(scores)
                f_out.write(f"{metric}\t{avg}\t{stdev}\t{n_runs}\n")
                print(metric, avg, stdev, str(n_runs) + " run(s)")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Usage: analyze.py FILENAME")
        sys.exit(1)
    average_scores(sys.argv[1])
