Use the `-recursive` flag while cloning (the `datasets` folder contains submodules):

```
git clone git@github.com:mainlp/noisydialect.git -recursive
```

1. Retrieve the datasets that weren't included in git submodules.

```
wget https://zenodo.org/record/2536041/files/Corpus_Release2_090119.zip
unzip -d datasets/Restaure_Alsatian/ Corpus_Release2_090119.zip
rm Corpus_Release2_090119.zip
```

2. Convert the corpora into a common format: (This creates files named `{train,dev,test}_CORPUS-NAME_TAG-TYPE.tsv` in the `datasets` folder.)

```
cd code

# Hamburg Dependency Treebank (DEU) -- TIGERized STTS tags
python 0.corpus_prep.py --type ud --dir ../datasets/UD_German-HDT/ --files de_hdt-ud-train-a-1.conllu,de_hdt-ud-train-a-2.conllu,de_hdt-ud-train-b-1.conllu,de_hdt-ud-train-b-2.conllu --out ../datasets/train_HDT_STTS.tsv --xpos --tigerize
python 0.corpus_prep.py --type ud --dir ../datasets/UD_German-HDT/ --files de_hdt-ud-dev.conllu --out ../datasets/dev_HDT_STTS.tsv --xpos --tigerize
python 0.corpus_prep.py --type ud --dir ../datasets/UD_German-HDT/ --files de_hdt-ud-test.conllu --out ../datasets/test_HDT_STTS.tsv --xpos --tigerize

# Hamburg Dependency Treebank (DEU) -- UPOS tags
python 0.corpus_prep.py --type ud --dir ../datasets/UD_German-HDT/ --files de_hdt-ud-train-a-1.conllu,de_hdt-ud-train-a-2.conllu,de_hdt-ud-train-b-1.conllu,de_hdt-ud-train-b-2.conllu --out ../datasets/train_HDT_UPOS.tsv
python 0.corpus_prep.py --type ud --dir ../datasets/UD_German-HDT/ --files de_hdt-ud-dev.conllu --out ../datasets/dev_HDT_UPOS.tsv
python 0.corpus_prep.py --type ud --dir ../datasets/UD_German-HDT/ --files de_hdt-ud-test.conllu --out ../datasets/test_HDT_UPOS.tsv

# Alpino (NLD) -- UPOS tags
python 0.corpus_prep.py --type ud --dir ../datasets/UD_Dutch-Alpino/ --files nl_alpino-ud-train.conllu --out ../datasets/train_Alpino_UPOS.tsv
python 0.corpus_prep.py --type ud --dir ../datasets/UD_Dutch-Alpino/ --files nl_alpino-ud-dev.conllu --out ../datasets/dev_Alpino_UPOS.tsv
python 0.corpus_prep.py --type ud --dir ../datasets/UD_Dutch-Alpino/ --files nl_alpino-ud-test.conllu --out ../datasets/test_Alpino_UPOS.tsv

# NOAH (GSW) -- TIGERized STTS tags, UPOS tags
python 0.corpus_prep.py --type noah --dir ../datasets/NOAH-corpus/ --files test_GSW_STTS.txt --out ../datasets/test_NOAH_STTS.tsv
python 0.corpus_prep.py --type noah --dir ../datasets/NOAH-corpus/ --files test_GSW_UPOS.txt --out ../datasets/test_NOAH_UPOS.tsv

# UD_Swiss_German-UZH (GSW) -- UPOS tags
python 0.corpus_prep.py --type ud --dir ../datasets/UD_Swiss_German-UZH/ --files gsw_uzh-ud-test.conllu --out ../datasets/test_UZH_UPOS.tsv

# Restaure_Alsatian (GSW) -- UPOS tags
python 0.corpus_prep.py --type ud --glob "../datasets/Restaure_Alsatian/ud/*" --out ../datasets/test_RA_UPOS.tsv

# UD_Low_Saxon-LSDC (NDS) -- UPOS tags
python 0.corpus_prep.py --type ud --dir ../datasets/UD_Low_Saxon-LSDC/ --files nds_lsdc-ud-test.conllu --out ../datasets/test_LSDC_UPOS.tsv

# UD_Frisian-Frysk (FRY) -- UPOS tags
python 0.corpus_prep.py --type ud --dir ../datasets/UD_Frisian-Frysk/ --files fy-frysk-ud-all.conllu --out ../datasets/test_Frysk_UPOS.tsv
```

3. Extract feature matrices for those experiments where the input representations aren't modified: (This creates subfolders in `data`, containing the input representations.)

```
# Vanilla STTS data:
python 0.data-matrix_prep.py ../configs/0.hdt-noah.dbmdz-cased_first.orig.60.stts.cfg
python 0.data-matrix_prep.py ../configs/0.hdt-noah.dbmdz-cased.orig.60.stts.cfg
python 0.data-matrix_prep.py ../configs/0.hdt-noah.dbmdz-uncased.orig.60.stts.cfg
python 0.data-matrix_prep.py ../configs/0.hdt-noah.gbert-base.orig.60.stts.cfg
python 0.data-matrix_prep.py ../configs/0.hdt-noah.gbert-large.orig.60.stts.cfg
python 0.data-matrix_prep.py ../configs/0.hdt-noah.mbert-cased.orig.60.stts.cfg
python 0.data-matrix_prep.py ../configs/0.hdt-noah.europeana-deu.orig.60.stts.cfg
python 0.data-matrix_prep.py ../configs/0.hdt-noah.xlm-roberta.orig.60.stts.cfg

# Vanilla UPOS data:
python 0.data-matrix_prep.py ../configs/0.hdt-noah_lsdc_uzh.dbmdz-cased.orig.60.upos.cfg
python 0.data-matrix_prep.py ../configs/0.hdt-noah_lsdc_uzh.dbmdz-uncased.orig.60.upos.cfg
python 0.data-matrix_prep.py ../configs/0.hdt-noah_lsdc_uzh.gbert-base.orig.60.upos.cfg
python 0.data-matrix_prep.py ../configs/0.hdt-noah_lsdc_uzh.gbert-large.orig.60.upos.cfg
python 0.data-matrix_prep.py ../configs/0.hdt-noah_lsdc_uzh.mbert-cased.orig.60.upos.cfg
python 0.data-matrix_prep.py ../configs/0.hdt-noah_lsdc_uzh.europeana-deu.orig.60.upos.cfg
python 0.data-matrix_prep.py ../configs/0.hdt-noah_lsdc_uzh.xlm-roberta.orig.60.upos.cfg
python 0.data-matrix_prep.py ../configs/0.hdt-noah_lsdc_uzh.bertje.orig.60.upos.cfg
python 0.data-matrix_prep.py ../configs/0.hdt-noah_lsdc_uzh.bert-base-cased.orig.60.upos.cfg
python 0.data-matrix_prep.py ../configs/0.hdt-noah_lsdc_uzh.bert-base-uncased.orig.60.upos.cfg
python 0.data-matrix_prep.py ../configs/0.hdt-noah_lsdc_uzh.finnish-bert-cased.orig.60.upos.cfg

python 0.data-matrix_prep.py ../configs/0.hdt_12k_f-noah_lsdc_uzh.dbmdz-cased.orig.60.upos.cfg

python 0.data-matrix_prep.py ../configs/0.alpino-noah_lsdc_uzh.dbmdz-cased.orig.60.upos.cfg
python 0.data-matrix_prep.py ../configs/0.alpino-noah_lsdc_uzh.dbmdz-uncased.orig.60.upos.cfg
python 0.data-matrix_prep.py ../configs/0.alpino-noah_lsdc_uzh.bertje.orig.60.upos.cfg
python 0.data-matrix_prep.py ../configs/0.alpino-noah_lsdc_uzh.mbert-cased.orig.60.upos.cfg
python 0.data-matrix_prep.py ../configs/0.alpino-noah_lsdc_uzh.gbert-base.orig.60.upos.cfg
python 0.data-matrix_prep.py ../configs/0.alpino-noah_lsdc_uzh.xlm-roberta.orig.60.upos.cfg
python 0.data-matrix_prep.py ../configs/0.alpino-noah_lsdc_uzh.bert-base-cased.orig.60.upos.cfg
python 0.data-matrix_prep.py ../configs/0.alpino-noah_lsdc_uzh.bert-base-uncased.orig.60.upos.cfg
python 0.data-matrix_prep.py ../configs/0.alpino-noah_lsdc_uzh.finnish-bert-cased.orig.60.upos.cfg
```

4. Run baselines: (The results are saved in `results`, in folders named after the configs.)

```
# Vanilla STTS data:
python run.py -c ../configs/hdt-noah.dbmdz-cased.orig.60_first.stts.cfg --test_per_epoch
python run.py -c ../configs/hdt-noah.dbmdz-cased.orig.60.stts.cfg --test_per_epoch
python run.py -c ../configs/hdt-noah.dbmdz-uncased.orig.60.stts.cfg --test_per_epoch
python run.py -c ../configs/hdt-noah.gbert-base.orig.60.stts.cfg --test_per_epoch
python run.py -c ../configs/hdt-noah.gbert-large.orig.60.stts.cfg --test_per_epoch
python run.py -c ../configs/hdt-noah.mbert-cased.orig.60.stts.cfg --test_per_epoch
python run.py -c ../configs/hdt-noah.europeana-deu.orig.60.stts.cfg --test_per_epoch
python run.py -c ../configs/hdt-noah.xlm-roberta.orig.60.stts.cfg --test_per_epoch

# dbmdz-cased with random noise (STTS):
python run.py -c ../configs/hdt-noah.dbmdz-cased.randnoise10-15.60.stts.cfg --test_per_epoch
python run.py -c ../configs/hdt-noah.dbmdz-cased.randnoise25-30.60.stts.cfg --test_per_epoch
python run.py -c ../configs/hdt-noah.dbmdz-cased.randnoise40-45.60.stts.cfg --test_per_epoch
python run.py -c ../configs/hdt-noah.dbmdz-cased.randnoise55-60.60.stts.cfg --test_per_epoch
python run.py -c ../configs/hdt-noah.dbmdz-cased.randnoise70-75.60.stts.cfg --test_per_epoch
python run.py -c ../configs/hdt-noah.dbmdz-cased.randnoise85-90.60.stts.cfg --test_per_epoch
python run.py -c ../configs/hdt-noah.dbmdz-cased.randnoise100.60.stts.cfg --test_per_epoch
python run.py -c ../configs/hdt-noah.dbmdz-cased.randnoise0-100.60.stts.cfg --test_per_epoch


# Vanilla UPOS data:
python run.py -c ../configs/hdt-noah_lsdc_uzh.dbmdz-cased.orig.60.upos.cfg --test_per_epoch
python run.py -c ../configs/hdt-noah_lsdc_uzh.dbmdz-uncased.orig.60.upos.cfg --test_per_epoch
python run.py -c ../configs/hdt-noah_lsdc_uzh.gbert-base.orig.60.upos.cfg --test_per_epoch
python run.py -c ../configs/hdt-noah_lsdc_uzh.gbert-large.orig.60.upos.cfg --test_per_epoch
python run.py -c ../configs/hdt-noah_lsdc_uzh.mbert-cased.orig.60.upos.cfg --test_per_epoch
python run.py -c ../configs/hdt-noah_lsdc_uzh.europeana-deu.orig.60.upos.cfg --test_per_epoch
python run.py -c ../configs/hdt-noah_lsdc_uzh.xlm-roberta.orig.60.upos.cfg --test_per_epoch
python run.py -c ../configs/hdt-noah_lsdc_uzh.bertje.orig.60.upos.cfg --test_per_epoch
python run.py -c ../configs/hdt-noah_lsdc_uzh.bert-base-cased.orig.60.upos.cfg --test_per_epoch
python run.py -c ../configs/hdt-noah_lsdc_uzh.bert-base-uncased.orig.60.upos.cfg --test_per_epoch
python run.py -c ../configs/hdt-noah_lsdc_uzh.finnish-bert-cased.orig.60.upos.cfg --test_per_epoch

python run.py -c ../configs/hdt_12k_f-noah_lsdc_uzh.dbmdz-cased.orig.60.upos.cfg --test_per_epoch
python run.py -c ../configs/hdt_12k_r-noah_lsdc_uzh.dbmdz-cased.orig.60.upos.cfg --test_per_epoch

python run.py -c ../configs/alpino-noah_lsdc_uzh.dbmdz-cased.orig.60.upos.cfg --test_per_epoch
python run.py -c ../configs/alpino-noah_lsdc_uzh.dbmdz-uncased.orig.60.upos.cfg --test_per_epoch
python run.py -c ../configs/alpino-noah_lsdc_uzh.bertje.orig.60.upos.cfg --test_per_epoch
python run.py -c ../configs/alpino-noah_lsdc_uzh.mbert-cased.orig.60.upos.cfg --test_per_epoch
python run.py -c ../configs/alpino-noah_lsdc_uzh.gbert-base.orig.60.upos.cfg --test_per_epoch
python run.py -c ../configs/alpino-noah_lsdc_uzh.xlm-roberta.orig.60.upos.cfg --test_per_epoch
python run.py -c ../configs/alpino-noah_lsdc_uzh.bert-base-cased.orig.60.upos.cfg --test_per_epoch
python run.py -c ../configs/alpino-noah_lsdc_uzh.bert-base-uncased.orig.60.upos.cfg --test_per_epoch
python run.py -c ../configs/alpino-noah_lsdc_uzh.finnish-bert-cased.orig.60.upos.cfg --test_per_epoch

# dbmdz-cased with random noise (UPOS):
python run.py -c ../configs/hdt-noah_lsdc_uzh.dbmdz-cased.randnoise10-15.60.upos.cfg --test_per_epoch
python run.py -c ../configs/hdt-noah_lsdc_uzh.dbmdz-cased.randnoise25-30.60.upos.cfg --test_per_epoch
python run.py -c ../configs/hdt-noah_lsdc_uzh.dbmdz-cased.randnoise40-45.60.upos.cfg --test_per_epoch
python run.py -c ../configs/hdt-noah_lsdc_uzh.dbmdz-cased.randnoise55-60.60.upos.cfg --test_per_epoch
python run.py -c ../configs/hdt-noah_lsdc_uzh.dbmdz-cased.randnoise70-75.60.upos.cfg --test_per_epoch
python run.py -c ../configs/hdt-noah_lsdc_uzh.dbmdz-cased.randnoise85-90.60.upos.cfg --test_per_epoch
python run.py -c ../configs/hdt-noah_lsdc_uzh.dbmdz-cased.randnoise100.60.upos.cfg --test_per_epoch
python run.py -c ../configs/hdt-noah_lsdc_uzh.dbmdz-cased.randnoise0-100.60.upos.cfg --test_per_epoch
```

5. Reformat results files:
```
python clean_up_results.py
```
