import glob
import sys


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
    with open(summary_file, "w") as f_out:
        for metric in scores_all:
            n_runs = len(scores_all[metric])
            f_out.write(f"{metric}\t{sum(scores_all[metric]) / n_runs}\n")
            print(metric, sum(scores_all[metric]) / n_runs,
                  str(n_runs) + " run(s)")
        f_out.write(f"n_runs\t{n_runs}\n")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Usage: analyze.py FILENAME")
        sys.exit(1)
    average_scores(sys.argv[1])
