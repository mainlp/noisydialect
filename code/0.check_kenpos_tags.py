from collections import Counter


def get_counts(filename):
    tag2words = {}
    word2tag2count = {}
    tag2files = {}
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cells = line.split("\t")
            word = cells[0].strip().lower()
            tag = cells[1].strip()
            try:
                tag2words[tag].append(word)
            except KeyError:
                tag2words[tag] = [word]
            try:
                word2tag2count[word][tag] = word2tag2count[word][tag] + 1
            except KeyError:
                try:
                    word2tag2count[word][tag] = 1
                except KeyError:
                    word2tag2count[word] = {tag: 1}
            try:
                src_file = cells[2].strip()
            except IndexError:
                src_file = "UNK_FILE"
            try:
                tag2files[tag].add(src_file)
            except KeyError:
                tag2files[tag] = {src_file}
    tag_counts = [(tag, len(tag2words[tag])) for tag in tag2words]
    tag_counts.sort(key=lambda x: -x[1])
    words_with_rare_tags = []
    print(filename)
    print("\t".join(("POS TAG", "# TOKENS", "# TYPES",
                     "TOP 5 TYPES", "# FILES", "5 OF THE FILES")))
    for tag, count in tag_counts:
        counter = Counter(tag2words[tag])
        top5 = [x[0] for x in counter.most_common(5)]
        if count < 10:
            words_with_rare_tags += top5
        print("\t".join((tag,
                         str(count),
                         str(len(counter)),
                         " ".join(top5),
                         str(len(tag2files[tag])),
                         " ".join(list(tag2files[tag])[:5]))))
    print()
    for word in words_with_rare_tags:
        tag_counts = [(tag, word2tag2count[word][tag])
                      for tag in word2tag2count[word]]
        sorted(tag_counts, key=lambda x: -x[1])
        print(word + "\t" + "  ".join(
              (t + " " + str(c) for t, c in tag_counts)))
    print()
    return word2tag2count


if __name__ == "__main__":
    lang2word2tag2count = {}
    for filename in ("../datasets/dev_kenpos-bxk_upos.tsv",
                     "../datasets/test_kenpos-lri_upos.tsv",
                     "../datasets/test_kenpos-rag_upos.tsv"):
        word2tag2count = get_counts(filename)
        lang2word2tag2count[
            filename.split("-")[-1].split("_")[0]] = word2tag2count

    # print(lang2word2tag2count["bxk"]["ewe"])
