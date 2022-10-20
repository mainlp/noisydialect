use the `-recursive` flag while cloning (submodules!)

1. Convert the corpora into a common format:

```
# Hamburg Dependency Treebank (DEU) -- train/dev -- TIGERized STTS tags
python3 0.corpus_prep.py --type ud --dir ../datasets/UD_German-HDT/ --files de_hdt-ud-train-a-1.conllu,de_hdt-ud-train-a-2.conllu,de_hdt-ud-train-b-1.conllu,de_hdt-ud-train-b-2.conllu --out ../datasets/train_HDT_STTS.tsv --xpos
python3 0.corpus_prep.py --type ud --dir ../datasets/UD_German-HDT/ --files de_hdt-ud-dev.conllu --out ../datasets/dev_HDT_STTS.tsv --xpos
# Hamburg Dependency Treebank (DEU) -- UPOS tags
python3 0.corpus_prep.py --type ud --dir ../datasets/UD_German-HDT/ --files de_hdt-ud-train-a-1.conllu,de_hdt-ud-train-a-2.conllu,de_hdt-ud-train-b-1.conllu,de_hdt-ud-train-b-2.conllu --out ../datasets/train_HDT_UPOS.tsv
python3 0.corpus_prep.py --type ud --dir ../datasets/UD_German-HDT/ --files de_hdt-ud-dev.conllu --out ../datasets/dev_HDT_UPOS.tsv
# NOAH (GSW) -- test -- TIGERized STTS tags
python3 0.corpus_prep.py --type noah --dir ../datasets/NOAH-corpus/ --files test_GSW_STTS.txt --out ../datasets/test_NOAH_STTS.tsv
# NOAH (GSW) -- test -- UPOS tags
python3 0.corpus_prep.py --type noah --dir ../datasets/NOAH-corpus/ --files test_GSW_UPOS.txt --out ../datasets/test_NOAH_UPOS.tsv
```

2. Extract feature matrices for those experiments where the input representations aren't modified:

