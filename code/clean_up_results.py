import glob
import os
import sys

import numpy as np

from model import score


def average_scores(directory):
    # Average scores across initializations
    scores_all = {}
    for res_file in glob.glob(f"{directory}/results_[0-9]*.tsv"):
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

    with open(summary_file, "w") as f_out:
        f_out.write("METRIC\tAVERAGE\tSTD_DEV\tN_RUNS\n")
        for metric in scores_all:
            scores = scores_all[metric]
            n_runs = len(scores)
            avg = sum(scores) / n_runs
            stdev = np.std(scores)
            f_out.write(f"{metric}\t{avg}\t{stdev}\t{n_runs}\n")
            # print(metric, avg, stdev, str(n_runs) + " run(s)")


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
            run_details = filename.split("/")[-1].rsplit("_", 2)
            setup = run_details[0].split("_", 1)[1]
            seed = run_details[1]
            epoch = run_details[2].replace("epoch", "").replace("ep", "")[:-4]
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
