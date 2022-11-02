import glob
import numpy as np
import re
import sys

from data import DUMMY_POS
import matplotlib.pyplot as plt
from model import filter_predictions

from sklearn.metrics import ConfusionMatrixDisplay, classification_report,\
    confusion_matrix

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


def tag_distributions_confusion_matrix(tagset_file, predictions_file,
                                       pretty_confusion_matrix=False):
    dummy_idx = None
    idx2pos = []
    with open(tagset_file) as in_file:
        for line in in_file:
            line = line.strip()
            if line:
                if line == DUMMY_POS:
                    dummy_idx = len(idx2pos)
                idx2pos.append(line)

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
            golds.append(int(cells[1]))
            preds.append(int(cells[0]))
    gold_filtered, pred_filtered = filter_predictions(preds, golds, dummy_idx)
    print(classification_report(gold_filtered, pred_filtered))
    print("With DUMMY tag (ignored while training):")
    print(confusion_matrix(golds, preds,
                           labels=[i for i in range(len(idx2pos))]))
    print("\nWithout DUMMY tag:")
    cm = confusion_matrix(gold_filtered, pred_filtered,
                          labels=[i for i in range(len(idx2pos))])
    print(cm)
    if pretty_confusion_matrix:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=idx2pos)
        disp.plot()
        plt.xticks(rotation=90)
        plt.show()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Usage: analyze.py FILENAME")
        sys.exit(1)
    average_scores(sys.argv[1])
