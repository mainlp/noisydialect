Use the `-recursive` flag while cloning (the `datasets` folder contains git submodules):

```
git clone git@github.com:mainlp/noisydialect.git -recursive
```

1. Retrieve the datasets that weren't included in any submodules.

```
# NArabizi
wget https://parsiti.github.io/NArabizi/NArabizi_Treebank.tar.gz
tar -xzf NArabizi_Treebank.tar.gz -C datasets/
rm NArabizi_Treebank.tar.gz 

# RESTAURE Alsatian
wget https://zenodo.org/record/2536041/files/Corpus_Release2_090119.zip
unzip -d datasets/Restaure_Alsatian/ Corpus_Release2_090119.zip
rm Corpus_Release2_090119.zip

# RESTAURE Occitan
wget https://zenodo.org/record/1182949/files/CorpusRestaureOccitan.zip
unzip -d datasets/ CorpusRestaureOccitan.zip
mv datasets/CorpusRestaureOccitan datasets/Restaure_Occitan
rm CorpusRestaureOccitan.zip
rm -r datasets/__MACOSX

# RESTAURE Picard
wget https://zenodo.org/record/1485988/files/corpus_picard_restaure.zip
unzip -d datasets/ corpus_picard_restaure.zip
mv datasets/corpus_picard_restaure datasets/Restaure_Picard
rm corpus_picard_restaure.zip

# LA-Murre
wget https://korp.csc.fi/download/la-murre/vrt/la-murre-vrt.zip
unzip -d datasets/ la-murre-vrt.zip
rm la-murre-vrt.zip

# UD_Norwegian-NynorskLIA_dialect
cd datasets/UD_Norwegian-NynorskLIA_dialect
./run.sh
cd ../..
```

The RESTAURE corpora are released under the [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) license.

2. Convert the corpora into a common format: (This creates files named `{train,dev,test}_CORPUS_TAGSET.tsv` in the `datasets` folder.)

```
cd code

################
# Arabic lects #
################

# UD_Arabic-PADT (original and transliterated)
python3 0.corpus_prep.py --type ud --dir ../datasets/UD_Arabic-PADT/ --files ar_padt-ud-train.conllu --out ../datasets/train_PADT_UPOS.tsv
python3 0.corpus_prep.py --type ud --dir ../datasets/UD_Arabic-PADT/ --files ar_padt-ud-train.conllu --out ../datasets/train_PADT-translit_UPOS.tsv --translit

# UD_Maltese-MUDT
python3 0.corpus_prep.py --type ud --dir ../datasets/UD_Maltese-MUDT/ --files mt_mudt-ud-train.conllu --out ../datasets/train_MUDT_UPOS.tsv

# NArabizi treebank
python3 0.corpus_prep.py --type narabizi --dir ../datasets/NArabizi_Treebank/Release_ACL2020/Gold_annotation --files test.NArabizi_treebank.conllu --out ../datasets/NArabizi_Treebank/test.NArabizi_treebank_cleaned.conllu --tagset ../datasets/tagset_upos.txt
python3 0.corpus_prep.py --type ud --dir ../datasets/NArabizi_Treebank/ --files test.NArabizi_treebank_cleaned.conllu --out ../datasets/test_NArabizi_UPOS.tsv

# dialectal_arabic_resources 
# Optional preliminary check (see logs/arabic_preprocessing.log):
python3 0.check_arabic_segmentation.py ../datasets/dialectal_arabic_resources/seg_plus_pos_egy.txt ../datasets/dialectal_arabic_resources/seg_plus_pos_lev.txt ../datasets/dialectal_arabic_resources/seg_plus_pos_glf.txt ../datasets/dialectal_arabic_resources/seg_plus_pos_mgr.txt
# The actual data conversion:
python3 0.corpus_prep.py --type ara --dir ../datasets/dialectal_arabic_resources/ --files seg_plus_pos_egy.txt --out ../datasets/dev_dar-egy.tsv
python3 0.corpus_prep.py --type ara --dir ../datasets/dialectal_arabic_resources/ --files seg_plus_pos_glf.txt --out ../datasets/test_dar-glf.tsv
python3 0.corpus_prep.py --type ara --dir ../datasets/dialectal_arabic_resources/ --files seg_plus_pos_lev.txt --out ../datasets/test_dar-lev.tsv
python3 0.corpus_prep.py --type ara --dir ../datasets/dialectal_arabic_resources/ --files seg_plus_pos_mgr.txt --out ../datasets/test_dar-mgr.tsv
# Optional checks:
python3 0.validate_input_file.py ../datasets/dev_dar-egy.tsv ../datasets/tagset_upos.txt
python3 0.validate_input_file.py ../datasets/test_dar-glf.tsv ../datasets/tagset_upos.txt
python3 0.validate_input_file.py ../datasets/test_dar-lev.tsv ../datasets/tagset_upos.txt
python3 0.validate_input_file.py ../datasets/test_dar-mgr.tsv ../datasets/tagset_upos.txt


#######################
# West Germanic lects #
#######################

# UD_German-HDT
python3 0.corpus_prep.py --type ud --dir ../datasets/UD_German-HDT/ --files de_hdt-ud-train-a-1.conllu,de_hdt-ud-train-a-2.conllu,de_hdt-ud-train-b-1.conllu,de_hdt-ud-train-b-2.conllu --out ../datasets/train_HDT_UPOS.tsv
# python3 0.corpus_prep.py --type ud --dir ../datasets/UD_German-HDT/ --files de_hdt-ud-dev.conllu --out ../datasets/dev_HDT_STTS.tsv --xpos --tigerize
# python3 0.corpus_prep.py --type ud --dir ../datasets/UD_German-HDT/ --files de_hdt-ud-test.conllu --out ../datasets/test_HDT_STTS.tsv --xpos --tigerize

# UD_Dutch-Alpino
python3 0.corpus_prep.py --type ud --dir ../datasets/UD_Dutch-Alpino/ --files nl_alpino-ud-train.conllu --out ../datasets/train_Alpino_UPOS.tsv
# python3 0.corpus_prep.py --type ud --dir ../datasets/UD_Dutch-Alpino/ --files nl_alpino-ud-dev.conllu --out ../datasets/dev_Alpino_UPOS.tsv
# python3 0.corpus_prep.py --type ud --dir ../datasets/UD_Dutch-Alpino/ --files nl_alpino-ud-test.conllu --out ../datasets/test_Alpino_UPOS.tsv

# NOAH
python3 0.corpus_prep.py --type noah --dir ../datasets/NOAH-corpus/ --files test_GSW_UPOS.txt --out ../datasets/test_NOAH_UPOS.tsv --excl ../datasets/test_UZH_UPOS.tsv

# Restaure_Alsatian
python3 0.corpus_prep.py --type ud --glob "../datasets/Restaure_Alsatian/ud/*" --out ../datasets/test_RA_UPOS.tsv

# UD_Low_Saxon-LSDC
python3 0.corpus_prep.py --type ud --dir ../datasets/UD_Low_Saxon-LSDC/ --files nds_lsdc-ud-test.conllu --out ../datasets/test_LSDC_UPOS.tsv


######################
# Norwegian dialects #
######################

# UD_Norwegian-Bokmaal
python3 0.corpus_prep.py --type ud --dir ../datasets/UD_Norwegian-Bokmaal/ --files no_bokmaal-ud-train.conllu --out ../datasets/train_NDT-NOB_UPOS.tsv

# UD_Norwegian-Nynorsk
python3 0.corpus_prep.py --type ud --dir ../datasets/UD_Norwegian-Nynorsk/ --files no_nynorsk-ud-train.conllu --out ../datasets/train_NDT-NNO_UPOS.tsv

# UD_Norwegian-NynorskLIA_dialect
python3 0.prep_lia.py


#################
# Romance lects #
#################

# UD_Spanish-AnCora
python3 0.corpus_prep.py --type ud --dir ../datasets/UD_Spanish-AnCora/ --files es_ancora-ud-train.conllu --out ../datasets/train_AnCora-SPA_UPOS.tsv

# UD_French-GSD
python3 0.corpus_prep.py --type ud --dir ../datasets/UD_French-GSD/ --files fr_gsd-ud-train.conllu --out ../datasets/train_GSD_UPOS.tsv

# UD_Italian-ISDT
python3 0.corpus_prep.py --type ud --dir ../datasets/UD_Italian-ISDT/ --files it_isdt-ud-train.conllu --out ../datasets/train_ISDT_UPOS.tsv

# UD_Old_French-SRCMF
python3 0.corpus_prep.py --type ud --dir ../datasets/UD_Old_French-SRCMF/ --files fro_srcmf-ud-train.conllu --out ../datasets/train_SRCMF_UPOS.tsv

# UD_Catalan-AnCora
python3 0.corpus_prep.py --type ud --dir ../datasets/UD_Catalan-AnCora/ --files ca_ancora-ud-dev.conllu --out ../datasets/dev_AnCora-CAT_UPOS.tsv

# Restaure_Occitan
python3 0.corpus_prep.py --type ud --glob "../datasets/Restaure_Occitan/*" --out ../datasets/test_RO_UPOS.tsv

# Restaure_Picard
python3 0.corpus_prep.py --type ud --glob "../datasets/Restaure_Picard/picud/*" --out ../datasets/test_RP_UPOS.tsv --glosscomp

####################
# Finnish dialects #
####################

# UD_Finnish-TDT
python3 0.corpus_prep.py --type ud --dir ../datasets/UD_Finnish-TDT/ --files fi_tdt-ud-train.conllu --out ../datasets/train_TDT_UPOS.tsv

# UD_Estonian-EDT
python3 0.corpus_prep.py --type ud --dir ../datasets/UD_Estonian-EDT/ --files et_edt-ud-train.conllu --out ../datasets/train_EDT_UPOS.tsv

# Lauseopin arkiston murrekorpus
python3 0.prep_lamurre.py
```

3. Extract feature matrices for those experiments where the input representations aren't modified: (This creates subfolders in `data`, containing the input representations.)

```
# Vanilla UPOS data:
python3 0.data-matrix_prep.py ../configs/0.hdt-noah_lsdc_uzh.dbmdz-cased.orig.60.upos.cfg
python3 0.data-matrix_prep.py ../configs/0.hdt-noah_lsdc_uzh.dbmdz-uncased.orig.60.upos.cfg
python3 0.data-matrix_prep.py ../configs/0.hdt-noah_lsdc_uzh.gbert-base.orig.60.upos.cfg
python3 0.data-matrix_prep.py ../configs/0.hdt-noah_lsdc_uzh.gbert-large.orig.60.upos.cfg
python3 0.data-matrix_prep.py ../configs/0.hdt-noah_lsdc_uzh.mbert-cased.orig.60.upos.cfg
python3 0.data-matrix_prep.py ../configs/0.hdt-noah_lsdc_uzh.europeana-deu.orig.60.upos.cfg
python3 0.data-matrix_prep.py ../configs/0.hdt-noah_lsdc_uzh.xlm-roberta.orig.60.upos.cfg
python3 0.data-matrix_prep.py ../configs/0.hdt-noah_lsdc_uzh.bertje.orig.60.upos.cfg
python3 0.data-matrix_prep.py ../configs/0.hdt-noah_lsdc_uzh.bert-base-cased.orig.60.upos.cfg
python3 0.data-matrix_prep.py ../configs/0.hdt-noah_lsdc_uzh.bert-base-uncased.orig.60.upos.cfg
python3 0.data-matrix_prep.py ../configs/0.hdt-noah_lsdc_uzh.finnish-bert-cased.orig.60.upos.cfg

python3 0.data-matrix_prep.py ../configs/0.hdt_6k_f-noah_lsdc_uzh.dbmdz-cased.orig.60.upos.cfg
python3 0.data-matrix_prep.py ../configs/0.hdt_12k_f-noah_lsdc_uzh.dbmdz-cased.orig.60.upos.cfg
python3 0.data-matrix_prep.py ../configs/0.hdt_12k_l-noah_lsdc_uzh.dbmdz-cased.orig.60.upos.cfg
python3 0.data-matrix_prep.py ../configs/0.hdt_24k_f-noah_lsdc_uzh.dbmdz-cased.orig.60.upos.cfg
python3 0.data-matrix_prep.py ../configs/0.hdt_48k_f-noah_lsdc_uzh.dbmdz-cased.orig.60.upos.cfg

python3 0.data-matrix_prep.py ../configs/0.hdt_6k_f-noah_lsdc_uzh.bert-base-cased.orig.60.upos.cfg
python3 0.data-matrix_prep.py ../configs/0.hdt_12k_f-noah_lsdc_uzh.bert-base-cased.orig.60.upos.cfg
python3 0.data-matrix_prep.py ../configs/0.hdt_24k_f-noah_lsdc_uzh.bert-base-cased.orig.60.upos.cfg
python3 0.data-matrix_prep.py ../configs/0.hdt_48k_f-noah_lsdc_uzh.bert-base-cased.orig.60.upos.cfg

python3 0.data-matrix_prep.py ../configs/0.alpino-noah_lsdc_uzh.dbmdz-cased.orig.60.upos.cfg
python3 0.data-matrix_prep.py ../configs/0.alpino-noah_lsdc_uzh.dbmdz-uncased.orig.60.upos.cfg
python3 0.data-matrix_prep.py ../configs/0.alpino-noah_lsdc_uzh.bertje.orig.60.upos.cfg
python3 0.data-matrix_prep.py ../configs/0.alpino-noah_lsdc_uzh.mbert-cased.orig.60.upos.cfg
python3 0.data-matrix_prep.py ../configs/0.alpino-noah_lsdc_uzh.gbert-base.orig.60.upos.cfg
python3 0.data-matrix_prep.py ../configs/0.alpino-noah_lsdc_uzh.xlm-roberta.orig.60.upos.cfg
python3 0.data-matrix_prep.py ../configs/0.alpino-noah_lsdc_uzh.bert-base-cased.orig.60.upos.cfg
python3 0.data-matrix_prep.py ../configs/0.alpino-noah_lsdc_uzh.bert-base-uncased.orig.60.upos.cfg
python3 0.data-matrix_prep.py ../configs/0.alpino-noah_lsdc_uzh.finnish-bert-cased.orig.60.upos.cfg
```

4. Run baselines: (The results are saved in `results`, in folders named after the configs.)

```
# Vanilla STTS data:
python3 run.py -c ../configs/hdt-noah.dbmdz-cased.orig.60_first.stts.cfg --test_per_epoch
python3 run.py -c ../configs/hdt-noah.dbmdz-cased.orig.60.stts.cfg --test_per_epoch
python3 run.py -c ../configs/hdt-noah.dbmdz-uncased.orig.60.stts.cfg --test_per_epoch
python3 run.py -c ../configs/hdt-noah.gbert-base.orig.60.stts.cfg --test_per_epoch
python3 run.py -c ../configs/hdt-noah.gbert-large.orig.60.stts.cfg --test_per_epoch
python3 run.py -c ../configs/hdt-noah.mbert-cased.orig.60.stts.cfg --test_per_epoch
python3 run.py -c ../configs/hdt-noah.europeana-deu.orig.60.stts.cfg --test_per_epoch
python3 run.py -c ../configs/hdt-noah.xlm-roberta.orig.60.stts.cfg --test_per_epoch

# dbmdz-cased with random noise (STTS):
python3 run.py -c ../configs/hdt-noah.dbmdz-cased.randnoise10-15.60.stts.cfg --test_per_epoch
python3 run.py -c ../configs/hdt-noah.dbmdz-cased.randnoise25-30.60.stts.cfg --test_per_epoch
python3 run.py -c ../configs/hdt-noah.dbmdz-cased.randnoise40-45.60.stts.cfg --test_per_epoch
python3 run.py -c ../configs/hdt-noah.dbmdz-cased.randnoise55-60.60.stts.cfg --test_per_epoch
python3 run.py -c ../configs/hdt-noah.dbmdz-cased.randnoise70-75.60.stts.cfg --test_per_epoch
python3 run.py -c ../configs/hdt-noah.dbmdz-cased.randnoise85-90.60.stts.cfg --test_per_epoch
python3 run.py -c ../configs/hdt-noah.dbmdz-cased.randnoise100.60.stts.cfg --test_per_epoch
python3 run.py -c ../configs/hdt-noah.dbmdz-cased.randnoise0-100.60.stts.cfg --test_per_epoch


# Vanilla UPOS data:
python3 run.py -c ../configs/hdt-noah_lsdc_uzh.dbmdz-cased.orig.60.upos.cfg --test_per_epoch
python3 run.py -c ../configs/hdt-noah_lsdc_uzh.dbmdz-uncased.orig.60.upos.cfg --test_per_epoch
python3 run.py -c ../configs/hdt-noah_lsdc_uzh.gbert-base.orig.60.upos.cfg --test_per_epoch
python3 run.py -c ../configs/hdt-noah_lsdc_uzh.gbert-large.orig.60.upos.cfg --test_per_epoch
python3 run.py -c ../configs/hdt-noah_lsdc_uzh.mbert-cased.orig.60.upos.cfg --test_per_epoch
python3 run.py -c ../configs/hdt-noah_lsdc_uzh.europeana-deu.orig.60.upos.cfg --test_per_epoch
python3 run.py -c ../configs/hdt-noah_lsdc_uzh.xlm-roberta.orig.60.upos.cfg --test_per_epoch
python3 run.py -c ../configs/hdt-noah_lsdc_uzh.bertje.orig.60.upos.cfg --test_per_epoch
python3 run.py -c ../configs/hdt-noah_lsdc_uzh.bert-base-cased.orig.60.upos.cfg --test_per_epoch
python3 run.py -c ../configs/hdt-noah_lsdc_uzh.bert-base-uncased.orig.60.upos.cfg --test_per_epoch
python3 run.py -c ../configs/hdt-noah_lsdc_uzh.finnish-bert-cased.orig.60.upos.cfg --test_per_epoch

python3 run.py -c ../configs/hdt_6k_f-noah_lsdc_uzh.dbmdz-cased.orig.60.upos.cfg --test_per_epoch
python3 run.py -c ../configs/hdt_6r_f-noah_lsdc_uzh.dbmdz-cased.orig.60.upos.cfg --test_per_epoch
python3 run.py -c ../configs/hdt_12k_f-noah_lsdc_uzh.dbmdz-cased.orig.60.upos.cfg --test_per_epoch
python3 run.py -c ../configs/hdt_12k_l-noah_lsdc_uzh.dbmdz-cased.orig.60.upos.cfg --test_per_epoch
python3 run.py -c ../configs/hdt_12k_r-noah_lsdc_uzh.dbmdz-cased.orig.60.upos.cfg --test_per_epoch
python3 run.py -c ../configs/hdt_24k_f-noah_lsdc_uzh.dbmdz-cased.orig.60.upos.cfg --test_per_epoch
python3 run.py -c ../configs/hdt_24k_r-noah_lsdc_uzh.dbmdz-cased.orig.60.upos.cfg --test_per_epoch
python3 run.py -c ../configs/hdt_48k_f-noah_lsdc_uzh.dbmdz-cased.orig.60.upos.cfg --test_per_epoch
python3 run.py -c ../configs/hdt_48k_r-noah_lsdc_uzh.dbmdz-cased.orig.60.upos.cfg --test_per_epoch

python3 run.py -c ../configs/hdt_6k_f-noah_lsdc_uzh.bert-base-cased.orig.60.upos.cfg --test_per_epoch
python3 run.py -c ../configs/hdt_6r_f-noah_lsdc_uzh.bert-base-cased.orig.60.upos.cfg --test_per_epoch
python3 run.py -c ../configs/hdt_12k_f-noah_lsdc_uzh.bert-base-cased.orig.60.upos.cfg --test_per_epoch
python3 run.py -c ../configs/hdt_12k_l-noah_lsdc_uzh.bert-base-cased.orig.60.upos.cfg --test_per_epoch
python3 run.py -c ../configs/hdt_12k_r-noah_lsdc_uzh.bert-base-cased.orig.60.upos.cfg --test_per_epoch
python3 run.py -c ../configs/hdt_24k_f-noah_lsdc_uzh.bert-base-cased.orig.60.upos.cfg --test_per_epoch
python3 run.py -c ../configs/hdt_24k_r-noah_lsdc_uzh.bert-base-cased.orig.60.upos.cfg --test_per_epoch
python3 run.py -c ../configs/hdt_48k_f-noah_lsdc_uzh.bert-base-cased.orig.60.upos.cfg --test_per_epoch
python3 run.py -c ../configs/hdt_48k_r-noah_lsdc_uzh.bert-base-cased.orig.60.upos.cfg --test_per_epoch

python3 run.py -c ../configs/alpino-noah_lsdc_uzh.dbmdz-cased.orig.60.upos.cfg --test_per_epoch
python3 run.py -c ../configs/alpino-noah_lsdc_uzh.dbmdz-uncased.orig.60.upos.cfg --test_per_epoch
python3 run.py -c ../configs/alpino-noah_lsdc_uzh.bertje.orig.60.upos.cfg --test_per_epoch
python3 run.py -c ../configs/alpino-noah_lsdc_uzh.mbert-cased.orig.60.upos.cfg --test_per_epoch
python3 run.py -c ../configs/alpino-noah_lsdc_uzh.gbert-base.orig.60.upos.cfg --test_per_epoch
python3 run.py -c ../configs/alpino-noah_lsdc_uzh.xlm-roberta.orig.60.upos.cfg --test_per_epoch
python3 run.py -c ../configs/alpino-noah_lsdc_uzh.bert-base-cased.orig.60.upos.cfg --test_per_epoch
python3 run.py -c ../configs/alpino-noah_lsdc_uzh.bert-base-uncased.orig.60.upos.cfg --test_per_epoch
python3 run.py -c ../configs/alpino-noah_lsdc_uzh.finnish-bert-cased.orig.60.upos.cfg --test_per_epoch

# dbmdz-cased with random noise (UPOS):
python3 run.py -c ../configs/hdt-noah_lsdc_uzh.dbmdz-cased.randnoise10-15.60.upos.cfg --test_per_epoch
python3 run.py -c ../configs/hdt-noah_lsdc_uzh.dbmdz-cased.randnoise25-30.60.upos.cfg --test_per_epoch
python3 run.py -c ../configs/hdt-noah_lsdc_uzh.dbmdz-cased.randnoise40-45.60.upos.cfg --test_per_epoch
python3 run.py -c ../configs/hdt-noah_lsdc_uzh.dbmdz-cased.randnoise55-60.60.upos.cfg --test_per_epoch
python3 run.py -c ../configs/hdt-noah_lsdc_uzh.dbmdz-cased.randnoise70-75.60.upos.cfg --test_per_epoch
python3 run.py -c ../configs/hdt-noah_lsdc_uzh.dbmdz-cased.randnoise85-90.60.upos.cfg --test_per_epoch
python3 run.py -c ../configs/hdt-noah_lsdc_uzh.dbmdz-cased.randnoise100.60.upos.cfg --test_per_epoch
python3 run.py -c ../configs/hdt-noah_lsdc_uzh.dbmdz-cased.randnoise0-100.60.upos.cfg --test_per_epoch
```

5. Reformat results files:
```
python3 clean_up_results.py
```

6.
```
python3 dataset_stats.py
```
