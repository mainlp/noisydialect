from collections import Counter
import sys

from C_data import DUMMY_POS
from C_model import filter_predictions

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, classification_report,\
    confusion_matrix, accuracy_score, f1_score


def tag_distributions_confusion_matrix(predictions_file, tagset_file,
                                       pretty_confusion_matrix=True,
                                       ignore_dummies_in_cm=True):
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
            preds.append(int(cells[0]))
            golds.append(int(cells[1]))

    counter_g = Counter(golds)
    counter_p = Counter(preds)
    total = sum(counter_g.values())
    print("TAG_IDX\tTAG\tGOLD_ABS\tGOLD_REL\tPRED_ABS\tPRED_REL")
    for i in range(len(idx2pos)):
        print(f"{i}\t{idx2pos[i]}\t{counter_g[i]}\t{counter_g[i] / total:.2f}\t{counter_p[i]}\t{counter_p[i] / total:.2f}")
    golds_filtered, preds_filtered = filter_predictions(
        np.asarray(preds), np.asarray(golds), dummy_idx)
    print(len(golds), len(golds_filtered))
    print(len(preds), len(preds_filtered))
    acc = accuracy_score(golds_filtered, preds_filtered)
    f1 = f1_score(golds_filtered, preds_filtered, average="macro",
                  zero_division=0)
    print(f"\nAccuracy: {acc:.4f}")
    print(f"F1_macro: {f1:.4f}\n")

    print(classification_report(golds_filtered, preds_filtered))

    if ignore_dummies_in_cm:
        print("\nWithout DUMMY tag:")
        cm = confusion_matrix(golds_filtered, preds_filtered,
                              labels=[i for i in range(len(idx2pos))])
    else:
        print("With DUMMY tag (ignored while training):")
        cm = confusion_matrix(golds, preds,
                              labels=[i for i in range(len(idx2pos))])
    print(cm)
    if pretty_confusion_matrix:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=idx2pos)
        disp.plot()
        plt.xticks(rotation=90)
        plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: D_analyze_results.py PRED_FILE TAGSET_FILE [--nop] [--incl]")
        print("(--nop = don't visualize the confusion matrix)")
        print("(--incl = include DUMMY_POS in the confusion matrix)")
        sys.exit(1)

    pretty_confusion_matrix = True
    ignore_dummies_in_cm = True
    if len(sys.argv) > 3:
        if sys.argv[3] == "--nop":
            pretty_confusion_matrix = False
        elif sys.argv[3] == "--incl":
            ignore_dummies_in_cm = False
        if len(sys.argv) > 4:
            if sys.argv[4] == "--nop":
                pretty_confusion_matrix = False
            elif sys.argv[4] == "--incl":
                ignore_dummies_in_cm = False
    tag_distributions_confusion_matrix(sys.argv[1], sys.argv[2],
                                       pretty_confusion_matrix,
                                       ignore_dummies_in_cm)
