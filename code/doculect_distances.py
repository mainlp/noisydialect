import lang2vec.lang2vec as l2v
from scipy.spatial.distance import pdist
import numpy as np
from functools import partial


def complete_coverage(dist_measure):
    return dist_measure == "fam" or dist_measure.endswith("_knn")


def distance(lang1, lang2, dist_measure):
    if lang1 == lang2:
        return 0, 0, 0
    # Only two langs at a time so we only skip a minimal
    # number of (partially) missing features
    missing = set()
    if dist_measure == "learned":
        if lang1 not in l2v.LEARNED_LETTER_CODES \
                or lang2 not in l2v.LEARNED_LETTER_CODES:
            return -1, 0, 512
        # Issues with pickle setting.
        # https://stackoverflow.com/a/58586450
        np_load_old = partial(np.load)
        np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
    if complete_coverage(dist_measure):
        vectors = [v for v in l2v.get_features(
            [lang1, lang2], dist_measure).values()]
    else:
        vectors = []
        for vector in l2v.get_features([lang1, lang2], dist_measure).values():
            vectors.append(vector)
            for i, val in enumerate(vector):
                if val == "--":
                    missing.add(i)
        if missing:
            vectors = [[val for i, val in
                        enumerate(vector) if i not in missing]
                       for vector in vectors]
    if dist_measure == "learned":
        np.load = np_load_old
    num_skipped = len(missing)
    num_compared = len(vectors[0])
    if num_compared == 0:
        return -1, 0, num_skipped
    return pdist(vectors, "cosine")[0], num_compared, num_skipped


if __name__ == "__main__":
    lects1 = ["deu", "nld", "eng"]
    lects2 = ["deu", "bar", "gsw", "nld", "nds", "fry", "yid"]

    for feat in sorted(l2v.FEATURE_SETS):
        print(feat)
        print("===")
        for lect1 in lects1:
            for lect2 in lects2:
                if lect1 == lect2:
                    continue
                print(lect1, lect2, distance(lect1, lect2, feat))
        print()
