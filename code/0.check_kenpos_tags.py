from collections import Counter


if __name__ == "__main__":
    filenames = ("../datasets/dev_kenpos-bxk_upos.tsv",
                 "../datasets/test_kenpos-lri_upos.tsv",
                 "../datasets/test_kenpos-rag_upos.tsv")

    for filename in filenames:
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
        print(filename)
        print("\t".join(("POS TAG", "# TOKENS", "# TYPES",
                         "TOP 5 TYPES", "# FILES", "5 OF THE FILES")))
        for tag, count in tag_counts:
            counter = Counter(tag2words[tag])
            print("\t".join((tag,
                             str(count),
                             str(len(counter)),
                             " ".join([x[0] for x in counter.most_common(5)]),
                             str(len(tag2files[tag])),
                             " ".join(list(tag2files[tag])[:5]))))
        print()
