import glob
import os
import re
import sys

import numpy as np

from model import score


pattern = re.compile("(dev|val|test).*_(f1|acc)_epoch[0-9]+$")


def average_scores(directory):
    # Average scores across initializations
    scores_all = {}
    for res_file in glob.glob(f"{directory}/results*.tsv"):
        if res_file.endswith("AVG.tsv"):
            continue
        with open(res_file) as f:
            for line in f:
                line = line.strip()
                if (line.startswith("test") or line.startswith("dev")
                    or line.startswith("val")) \
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
        testset2epoch2data2metric = {}
        max_epoch = -1
        for key in scores_all:
            cells = key.split("_")
            data = cells[0]
            testset = "-"
            if "." in data:
                data_details = data.split(".")
                data = data_details[0]
                testset = data_details[1]
            metric = cells[-2]
            epochstr = cells[-1]
            # TODO multi-UPOS scenarios
            if data[:3] in ("dev", "val"):
                data = "val"
            elif data[:4] == "test":
                data = "test"
            epoch = int(epochstr[5:])
            if epoch > max_epoch:
                max_epoch = epoch
            if testset not in testset2epoch2data2metric:
                testset2epoch2data2metric[testset] = {}
            if epoch not in testset2epoch2data2metric[testset]:
                testset2epoch2data2metric[testset][epoch] = {}
            if data not in testset2epoch2data2metric[testset][epoch]:
                testset2epoch2data2metric[testset][epoch][data] = {}
            if metric not in testset2epoch2data2metric[testset][epoch][data]:
                testset2epoch2data2metric[testset][epoch][data][metric] = {}
            testset2epoch2data2metric[testset][epoch][data][metric] =\
                scores_all[key]
        with open(summary_file, "w") as f_out:
            f_out.write("N_RUNS\tEPOCH\tF1 (TEST)\tSTDEV (F1 TEST)\t"
                        "ACC (TEST)\tSTDEV (ACC TEST)\t"
                        "F1 (VAL)\tSTDEV (F1 VAL)\t"
                        "ACC (VAL)\tSTDEV (ACC VAL)\n")
            for testset in testset2epoch2data2metric:
                if testset != "-":
                    f_out.write(testset)
                    f_out.write("\n---------------------\n")
                for epoch in range(1, 1 + max_epoch):
                    try:
                        n_runs = len(testset2epoch2data2metric[testset][epoch]["test"]["f1"])
                    except KeyError:
                        n_runs = len(testset2epoch2data2metric[testset][epoch]["val"]["f1"])
                    f_out.write(f"{n_runs}\t{epoch}")
                    for data in ("test", "val"):
                        for metric in ("f1", "acc"):
                            try:
                                scores = testset2epoch2data2metric[testset][epoch][data][metric]
                                n_runs = len(scores)
                                avg = sum(scores) / n_runs
                                stdev = np.std(scores)
                                f_out.write(f"\t{avg}\t{stdev}")
                            except KeyError:
                                f_out.write("\t-\t-")
                    f_out.write("\n")
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


# Calculate the scores again for the given predictions (in case the
# old F1 implementation was used, which didn't properly exclude the
# dummy class from the averaging step).
def rescore_predictions(predictions_file, dummy_idx=0):
    preds, golds = [], []
    header_found = False
    with open(predictions_file) as in_file:
        for line in in_file:
            if not header_found:
                header_found = True
                continue
            line = line.strip()
            if not line:
                continue
            cells = line.split("\t")
            preds.append(int(cells[0]))
            golds.append(int(cells[1]))
    acc, f1_macro = score(np.asarray(preds), np.asarray(golds), dummy_idx)
    return acc, f1_macro


def rescore_dir(directory):
    seed2scores = {}
    for filename in glob.glob(directory + "/predictions*"):
        try:
            run_details = filename.split("/")[-1].split("_")
            setup = run_details[1]
            seed = run_details[2]
            epoch = run_details[3][2:-4]
            acc, f1_macro = rescore_predictions(filename)
            scores_for_seed = seed2scores.get(seed, set())
            scores_for_seed.add((f"{setup}_acc_epoch{epoch}", acc))
            scores_for_seed.add((f"{setup}_f1_epoch{epoch}", f1_macro))
            seed2scores[seed] = scores_for_seed
        except IndexError:
            print("Skipping " + directory)
            return False
    for seed in seed2scores:
        out_file = f"{directory}/results_{seed}.tsv"
        print("- " + out_file)
        with open(out_file, "w+") as f:
            for scenario, result in seed2scores[seed]:
                f.write(f"{scenario}\t{result}\n")
    print("Re-scored " + directory)
    return True


if __name__ == "__main__":
    if len(sys.argv) > 1:
        path_glob = sys.argv[1]
    else:
        path_glob = "../results/*"
    for path in glob.glob(path_glob):
        if os.path.isfile(path):
            continue
        # Remove dummy predictions:
        for filename in glob.glob(path + "/*ep0.tsv"):
            os.remove(filename)
        # Fix the old F1 scores:
        rescored = rescore_dir(path)
        # Use the new score summary script:
        if rescored:
            average_scores(path)
