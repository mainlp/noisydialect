import glob
import os

from analyze_results import average_scores


if __name__ == "__main__":
    for directory in glob.glob("../results/*"):
        # Remove dummy predictions:
        for filename in glob.glob(directory + "/*ep0.tsv"):
            os.remove(filename)
        # Use the new score summary script:
        average_scores(directory)
