0. Use the `--recursive` flag while cloning (the `datasets` folder contains git submodules):

```
git clone git@github.com:mainlp/noisydialect.git --recursive
```

Install the requirements:
```
virtualenv env
source env/bin activate
python -m pip install -r requirements.txt
pip install torch==1.12.1+cu116 -f https://download.pytorch.org/whl/torch_stable.html
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
The LA-murre corpus is released under the [CC BY-ND 4.0](https://creativecommons.org/licenses/by-nd/4.0/) license.

2. Convert the corpora into a common format: (This creates files named `{train,dev,test}_CORPUS_TAGSET.tsv` in the `datasets` folder.)

```
cd code

################
# Arabic lects #
################

# UD_Arabic-PADT (original and transliterated) & UD_Maltese-MUDT
for split in "train" "dev" "test"
do 
  python3 A_corpus_prep.py --type ud --dir ../datasets/UD_Arabic-PADT/ --files ar_padt-ud-${split}.conllu --out ../datasets/${split}_PADT_UPOS.tsv
  python3 A_corpus_prep.py --type ud --dir ../datasets/UD_Arabic-PADT/ --files ar_padt-ud-${split}.conllu --out ../datasets/${split}_PADT-translit_UPOS.tsv --translit
  python3 A_corpus_prep.py --type ud --dir ../datasets/UD_Maltese-MUDT/ --files mt_mudt-ud-${split}.conllu --out ../datasets/${split}_MUDT_UPOS.tsv
done

# NArabizi treebank
python3 A_corpus_prep.py --type narabizi --dir ../datasets/NArabizi_Treebank/Release_ACL2020/Gold_annotation --files test.NArabizi_treebank.conllu --out ../datasets/NArabizi_Treebank/test.NArabizi_treebank_cleaned.conllu --tagset ../datasets/tagset_upos.txt
python3 A_corpus_prep.py --type ud --dir ../datasets/NArabizi_Treebank/ --files test.NArabizi_treebank_cleaned.conllu --out ../datasets/test_NArabizi_UPOS.tsv

# dialectal_arabic_resources 
# Optional preliminary check:
python3 A_check_arabic_segmentation.py ../datasets/dialectal_arabic_resources/seg_plus_pos_egy.txt ../datasets/dialectal_arabic_resources/seg_plus_pos_lev.txt ../datasets/dialectal_arabic_resources/seg_plus_pos_glf.txt ../datasets/dialectal_arabic_resources/seg_plus_pos_mgr.txt  > ../logs/arabic_preprocessing.log 
# The actual data conversion:
python3 A_corpus_prep.py --type ara --dir ../datasets/dialectal_arabic_resources/ --files seg_plus_pos_egy.txt --out ../datasets/dev_dar-egy_UPOS.tsv
python3 A_corpus_prep.py --type ara --dir ../datasets/dialectal_arabic_resources/ --files seg_plus_pos_glf.txt --out ../datasets/test_dar-glf_UPOS.tsv
python3 A_corpus_prep.py --type ara --dir ../datasets/dialectal_arabic_resources/ --files seg_plus_pos_lev.txt --out ../datasets/test_dar-lev_UPOS.tsv
python3 A_corpus_prep.py --type ara --dir ../datasets/dialectal_arabic_resources/ --files seg_plus_pos_mgr.txt --out ../datasets/test_dar-mgr_UPOS.tsv
# Optional checks:
python3 A_validate_input_file.py ../datasets/dev_dar-egy.tsv ../datasets/tagset_upos.txt
python3 A_validate_input_file.py ../datasets/test_dar-glf.tsv ../datasets/tagset_upos.txt
python3 A_validate_input_file.py ../datasets/test_dar-lev.tsv ../datasets/tagset_upos.txt
python3 A_validate_input_file.py ../datasets/test_dar-mgr.tsv ../datasets/tagset_upos.txt


#######################
# West Germanic lects #
#######################

# UD_German-HDT
python3 A_corpus_prep.py --type ud --dir ../datasets/UD_German-HDT/ --files de_hdt-ud-train-a-1.conllu,de_hdt-ud-train-a-2.conllu,de_hdt-ud-train-b-1.conllu,de_hdt-ud-train-b-2.conllu --out ../datasets/train_HDT_UPOS.tsv
python3 A_corpus_prep.py --type ud --dir ../datasets/UD_German-HDT/ --files de_hdt-ud-dev.conllu --out ../datasets/dev_HDT_UPOS.tsv
python3 A_corpus_prep.py --type ud --dir ../datasets/UD_German-HDT/ --files de_hdt-ud-test.conllu --out ../datasets/test_HDT_UPOS.tsv

# UD_Dutch-Alpino
for split in "train" "dev" "test"
do
  python3 A_corpus_prep.py --type ud --dir ../datasets/UD_Dutch-Alpino/ --files nl_alpino-ud-${split}.conllu --out ../datasets/${split}_Alpino_UPOS.tsv
done

# NOAH
python3 A_corpus_prep.py --type noah --dir ../datasets/NOAH-corpus/ --files test_GSW_UPOS.txt --out ../datasets/dev_NOAH_UPOS.tsv

# Restaure_Alsatian
python3 A_corpus_prep.py --type ud --glob "../datasets/Restaure_Alsatian/ud/*" --out ../datasets/test_RAls_UPOS.tsv

# UD_Low_Saxon-LSDC
python3 A_corpus_prep.py --type ud --dir ../datasets/UD_Low_Saxon-LSDC/ --files nds_lsdc-ud-test.conllu --out ../datasets/test_LSDC_UPOS.tsv


######################
# Norwegian dialects #
######################

# UD_Norwegian-Bokmaal & UD_Norwegian-Nynorsk
for split in "train" "dev" "test"
do
  python3 A_corpus_prep.py --type ud --dir ../datasets/UD_Norwegian-Bokmaal/ --files no_bokmaal-ud-${split}.conllu --out ../datasets/${split}_NDT-NOB_UPOS.tsv
  python3 A_corpus_prep.py --type ud --dir ../datasets/UD_Norwegian-Nynorsk/ --files no_nynorsk-ud-${split}.conllu --out ../datasets/${split}_NDT-NNO_UPOS.tsv
done

# UD_Norwegian-NynorskLIA_dialect
python3 A_prep_lia.py


#################
# Romance lects #
#################

# UD_Spanish-AnCora, UD_French-GSD, & UD_Catalan-AnCora
for split in "train" "dev" "test"
do
  python3 A_corpus_prep.py --type ud --dir ../datasets/UD_Spanish-AnCora/ --files es_ancora-ud-${split}.conllu --out ../datasets/${split}_AnCora-SPA_UPOS.tsv
  python3 A_corpus_prep.py --type ud --dir ../datasets/UD_French-GSD/ --files fr_gsd-ud-${split}.conllu --out ../datasets/${split}_GSD_UPOS.tsv
  python3 A_corpus_prep.py --type ud --dir ../datasets/UD_Catalan-AnCora/ --files ca_ancora-ud-${split}.conllu --out ../datasets/${split}_AnCora-CAT_UPOS.tsv
done

# Restaure_Picard
python3 A_corpus_prep.py --type ud --glob "../datasets/Restaure_Picard/picud/*" --out ../datasets/dev_RPic_UPOS.tsv --glosscomp

# Restaure_Occitan
python3 A_corpus_prep.py --type ud --glob "../datasets/Restaure_Occitan/*" --out ../datasets/test_ROci_UPOS.tsv


####################
# Finnish dialects #
####################

# UD_Finnish-TDT & UD_Estonian-EDT
for split in "train" "dev" "test"
do
  python3 A_corpus_prep.py --type ud --dir ../datasets/UD_Finnish-TDT/ --files fi_tdt-ud-${split}.conllu --out ../datasets/${split}_TDT_UPOS.tsv
  python3 A_corpus_prep.py --type ud --dir ../datasets/UD_Estonian-EDT/ --files et_edt-ud-${split}.conllu --out ../datasets/${split}_EDT_UPOS.tsv
done

# Lauseopin arkiston murrekorpus
python3 A_prep_lamurre.py
```

3. Figure out which sentence lengths to use: (The results are used for the configs later on.)
```
python3 B_corpus_stats.py ../datasets/train_PADT_UPOS.tsv,../datasets/dev_PADT_UPOS.tsv,../datasets/dev_dar-egy_UPOS.tsv ../logs/sentence_lengths_ara.tsv aubmindlab/bert-base-arabertv2 w+
python3 B_corpus_stats.py ../datasets/train_PADT_UPOS.tsv,../datasets/dev_PADT_UPOS.tsv,../datasets/dev_dar-egy_UPOS.tsv ../logs/sentence_lengths_ara.tsv bert-base-multilingual-cased a
python3 B_corpus_stats.py ../datasets/train_PADT_UPOS.tsv,../datasets/dev_PADT_UPOS.tsv,../datasets/dev_dar-egy_UPOS.tsv ../logs/sentence_lengths_ara.tsv xlm-roberta-base a

python3 B_corpus_stats.py ../datasets/train_HDT_UPOS.tsv,../datasets/dev_HDT_UPOS.tsv,../datasets/dev_NOAH_UPOS.tsv ../logs/sentence_lengths_wger.tsv deepset/gbert-base w+
python3 B_corpus_stats.py ../datasets/train_Alpino_UPOS.tsv,../datasets/dev_Alpino_UPOS.tsv,../datasets/dev_NOAH_UPOS.tsv ../logs/sentence_lengths_wger.tsv GroNLP/bert-base-dutch-cased a
python3 B_corpus_stats.py ../datasets/train_HDT_UPOS.tsv,../datasets/dev_HDT_UPOS.tsv,../datasets/dev_NOAH_UPOS.tsv,../datasets/train_Alpino_UPOS.tsv,../datasets/dev_Alpino_UPOS.tsv ../logs/sentence_lengths_wger.tsv bert-base-multilingual-cased a
python3 B_corpus_stats.py ../datasets/train_HDT_UPOS.tsv,../datasets/dev_HDT_UPOS.tsv,../datasets/dev_NOAH_UPOS.tsv,../datasets/train_Alpino_UPOS.tsv,../datasets/dev_Alpino_UPOS.tsv ../logs/sentence_lengths_wger.tsv xlm-roberta-base a

python3 B_corpus_stats.py ../datasets/train_NDT-NOB_UPOS.tsv,../datasets/dev_NDT-NOB_UPOS.tsv,../datasets/dev_LIA-west_UPOS.tsv../datasets/train_NDT-NNO_UPOS.tsv,../datasets/dev_NDT-NNO_UPOS.tsv ../logs/sentence_lengths_nor.tsv ltgoslo/norbert2 w+
python3 B_corpus_stats.py ../datasets/train_NDT-NOB_UPOS.tsv,../datasets/dev_NDT-NOB_UPOS.tsv,../datasets/dev_LIA-west_UPOS.tsv,../datasets/train_NDT-NNO_UPOS.tsv,../datasets/dev_NDT-NNO_UPOS.tsv ../logs/sentence_lengths_nor.tsv bert-base-multilingual-cased a
python3 B_corpus_stats.py ../datasets/train_NDT-NOB_UPOS.tsv,../datasets/dev_NDT-NOB_UPOS.tsv,../datasets/dev_LIA-west_UPOS.tsv,../datasets/train_NDT-NNO_UPOS.tsv,../datasets/dev_NDT-NNO_UPOS.tsv ../logs/sentence_lengths_nor.tsv xlm-roberta-base a

python3 B_corpus_stats.py ../datasets/train_GSD_UPOS.tsv,../datasets/dev_GSD_UPOS.tsv,../datasets/dev_RPic_UPOS.tsv ../logs/sentence_lengths_rom.tsv camembert-base a
python3 B_corpus_stats.py ../datasets/train_AnCora-SPA_UPOS.tsv,../datasets/dev_AnCora-SPA_UPOS.tsv,../datasets/dev_RPic_UPOS.tsv,../datasets/train_AnCora-CAT_UPOS.tsv,../datasets/dev_AnCora-CAT_UPOS.tsv ../logs/sentence_lengths_rom.tsv dccuchile/bert-base-spanish-wwm-uncased a
python3 B_corpus_stats.py ../datasets/train_GSD_UPOS.tsv,../datasets/dev_GSD_UPOS.tsv,../datasets/dev_RPic_UPOS.tsv,../datasets/train_AnCora-SPA_UPOS.tsv,../datasets/dev_AnCora-SPA_UPOS.tsv,../datasets/train_AnCora-CAT_UPOS.tsv,../datasets/dev_AnCora-CAT_UPOS.tsv ../logs/sentence_lengths_rom.tsv bert-base-multilingual-cased a
python3 B_corpus_stats.py ../datasets/train_GSD_UPOS.tsv,../datasets/dev_GSD_UPOS.tsv,../datasets/dev_RPic_UPOS.tsv,../datasets/train_AnCora-SPA_UPOS.tsv,../datasets/dev_AnCora-SPA_UPOS.tsv,../datasets/train_AnCora-CAT_UPOS.tsv,../datasets/dev_AnCora-CAT_UPOS.tsv ../logs/sentence_lengths_rom.tsv xlm-roberta-base a

python3 B_corpus_stats.py ../datasets/train_TDT_UPOS.tsv,../datasets/dev_TDT_UPOS.tsv,../datasets/dev_murre-SAV_UPOS.tsv,../datasets/train_EDT_UPOS.tsv,../datasets/dev_EDT_UPOS.tsv ../logs/sentence_lengths_fin.tsv TurkuNLP/bert-base-finnish-cased-v1 a
python3 B_corpus_stats.py ../datasets/train_TDT_UPOS.tsv,../datasets/dev_TDT_UPOS.tsv,../datasets/dev_murre-SAV_UPOS.tsv,../datasets/train_EDT_UPOS.tsv,../datasets/dev_EDT_UPOS.tsv ../logs/sentence_lengths_fin.tsv bert-base-multilingual-cased a
python3 B_corpus_stats.py ../datasets/train_TDT_UPOS.tsv,../datasets/dev_TDT_UPOS.tsv,../datasets/dev_murre-SAV_UPOS.tsv,../datasets/train_EDT_UPOS.tsv,../datasets/dev_EDT_UPOS.tsv ../logs/sentence_lengths_fin.tsv xlm-roberta-base a
```

4. A simple grid search for hyperparameters:
```
# Data prep (the created folders will be reused later)
python3 B_data-matrix_prep.py ../configs/B_padt-full_padt-egy_arabert_orig.cfg
python3 B_data-matrix_prep.py ../configs/B_tdt-full_tdt-sav_finbert_orig.cfg
python3 B_data-matrix_prep.py ../configs/B_hdt-full_hdt-noah_gbert_orig.cfg
python3 B_data-matrix_prep.py ../configs/B_padt-full_padt-egy_xlmr_orig.cfg
python3 B_data-matrix_prep.py ../configs/B_tdt-full_tdt-sav_xlmr_orig.cfg
python3 B_data-matrix_prep.py ../configs/B_hdt-full_hdt-noah_xlmr_orig.cfg

python3 B_create_gridsearch_configs.py --hyperparams

for sfx in "2e-05_16" "3e-05_16" "2e-05_32" "3e-05_32"
do
  python3 run.py -c ../configs/B_hyperparams_tdt-full_tdt-sav_finbert_orig_${sfx}.cfg --test_per_epoch
  python3 run.py -c ../configs/B_hyperparams_tdt-full_tdt-sav_xlmr_orig_${sfx}.cfg --test_per_epoch
  python3 run.py -c ../configs/B_hyperparams_hdt-full_hdt-noah_gbert_orig_${sfx}.cfg --test_per_epoch
  python3 run.py -c ../configs/B_hyperparams_hdt-full_hdt-noah_xlmr_orig_${sfx}.cfg --test_per_epoch
  python3 run.py -c ../configs/B_hyperparams_padt-full_padt-egy_arabert_orig_${sfx}.cfg --test_per_epoch
  python3 run.py -c ../configs/B_hyperparams_padt-full_padt-egy_xlmr_orig_${sfx}.cfg --test_per_epoch
done
```

5. Dev set experiments with different noise levels:
```
# Data prep
python3 B_data-matrix_prep.py ../configs/B_padt-full_padt-egy_mbert_orig.cfg
python3 B_data-matrix_prep.py ../configs/B_tdt-full_tdt-sav_mbert_orig.cfg
python3 B_data-matrix_prep.py ../configs/B_hdt-full_hdt-noah_mbert_orig.cfg
python3 B_data-matrix_prep.py ../configs/B_gsd-full_gsd-rpic_camembert_orig.cfg
python3 B_data-matrix_prep.py ../configs/B_gsd-full_gsd-rpic_mbert_orig.cfg
python3 B_data-matrix_prep.py ../configs/B_gsd-full_gsd-rpic_xlmr_orig.cfg
python3 B_data-matrix_prep.py ../configs/B_nob-full_nob-west_norbert_orig.cfg
python3 B_data-matrix_prep.py ../configs/B_nob-full_nob-west_mbert_orig.cfg
python3 B_data-matrix_prep.py ../configs/B_nob-full_nob-west_xlmr_orig.cfg

python3 B_create_gridsearch_configs.py --noise


for noise in "orig" "rand15" "rand35" "rand55" "rand75" "rand95"
do
  python3 run.py -c ../configs/C_hdt-full_hdt-noah_gbert_${noise}.cfg --test_per_epoch --save_model
  python3 run.py -c ../configs/C_hdt-full_hdt-noah_mbert_${noise}.cfg --tes_per_epoch --save_model
  python3 run.py -c ../configs/C_hdt-full_hdt-noah_xlmr_${noise}.cfg --test_per_epoch --save_model
  
  python3 run.py -c ../configs/C_gsd-full_gsd-rpic_camembert_${noise}.cfg --test_per_epoch --save_model
  python3 run.py -c ../configs/C_gsd-full_gsd-rpic_mbert_${noise}.cfg --test_per_epoch --save_model
  python3 run.py -c ../configs/C_gsd-full_gsd-rpic_xlmr_${noise}.cfg --test_per_epoch --save_model

  python3 run.py -c ../configs/C_nob-full_nob-west_norbert_${noise}.cfg --test_per_epoch --save_model
  python3 run.py -c ../configs/C_nob-full_nob-west_mbert_${noise}.cfg --test_per_epoch --save_model
  python3 run.py -c ../configs/C_nob-full_nob-west_xlmr_${noise}.cfg --test_per_epoch --save_model

  python3 run.py -c ../configs/C_padt-full_padt-egy_arabert_${noise}.cfg --test_per_epoch --save_model
  python3 run.py -c ../configs/C_padt-full_padt-egy_mbert_${noise}.cfg --test_per_epoch --save_model
  python3 run.py -c ../configs/C_padt-full_padt-egy_xlmr_${noise}.cfg --test_per_epoch --save_model

  python3 run.py -c ../configs/C_tdt-full_tdt-sav_finbert_${noise}.cfg --test_per_epoch --save_model
  python3 run.py -c ../configs/C_tdt-full_tdt-sav_mbert_${noise}.cfg --test_per_epoch --save_model
  python3 run.py -c ../configs/C_tdt-full_tdt-sav_xlmr_${noise}.cfg --test_per_epoch --save_model
done
```



3. Extract feature matrices for those experiments where the input representations aren't modified: (This creates subfolders in `data`, containing the input representations.)

```
# Vanilla UPOS data:
python3 B_data-matrix_prep.py ../configs/B_hdt-noah_lsdc_uzh.dbmdz-cased.orig.60.upos.cfg
python3 B_data-matrix_prep.py ../configs/B_hdt-noah_lsdc_uzh.dbmdz-uncased.orig.60.upos.cfg
python3 B_data-matrix_prep.py ../configs/B_hdt-noah_lsdc_uzh.gbert-base.orig.60.upos.cfg
python3 B_data-matrix_prep.py ../configs/B_hdt-noah_lsdc_uzh.gbert-large.orig.60.upos.cfg
python3 B_data-matrix_prep.py ../configs/B_hdt-noah_lsdc_uzh.mbert-cased.orig.60.upos.cfg
python3 B_data-matrix_prep.py ../configs/B_hdt-noah_lsdc_uzh.europeana-deu.orig.60.upos.cfg
python3 B_data-matrix_prep.py ../configs/B_hdt-noah_lsdc_uzh.xlm-roberta.orig.60.upos.cfg
python3 B_data-matrix_prep.py ../configs/B_hdt-noah_lsdc_uzh.bertje.orig.60.upos.cfg
python3 B_data-matrix_prep.py ../configs/B_hdt-noah_lsdc_uzh.bert-base-cased.orig.60.upos.cfg
python3 B_data-matrix_prep.py ../configs/B_hdt-noah_lsdc_uzh.bert-base-uncased.orig.60.upos.cfg
python3 B_data-matrix_prep.py ../configs/B_hdt-noah_lsdc_uzh.finnish-bert-cased.orig.60.upos.cfg

python3 B_data-matrix_prep.py ../configs/B_hdt_6k_f-noah_lsdc_uzh.dbmdz-cased.orig.60.upos.cfg
python3 B_data-matrix_prep.py ../configs/B_hdt_12k_f-noah_lsdc_uzh.dbmdz-cased.orig.60.upos.cfg
python3 B_data-matrix_prep.py ../configs/B_hdt_12k_l-noah_lsdc_uzh.dbmdz-cased.orig.60.upos.cfg
python3 B_data-matrix_prep.py ../configs/B_hdt_24k_f-noah_lsdc_uzh.dbmdz-cased.orig.60.upos.cfg
python3 B_data-matrix_prep.py ../configs/B_hdt_48k_f-noah_lsdc_uzh.dbmdz-cased.orig.60.upos.cfg

python3 B_data-matrix_prep.py ../configs/B_hdt_6k_f-noah_lsdc_uzh.bert-base-cased.orig.60.upos.cfg
python3 B_data-matrix_prep.py ../configs/B_hdt_12k_f-noah_lsdc_uzh.bert-base-cased.orig.60.upos.cfg
python3 B_data-matrix_prep.py ../configs/B_hdt_24k_f-noah_lsdc_uzh.bert-base-cased.orig.60.upos.cfg
python3 B_data-matrix_prep.py ../configs/B_hdt_48k_f-noah_lsdc_uzh.bert-base-cased.orig.60.upos.cfg

python3 B_data-matrix_prep.py ../configs/B_alpino-noah_lsdc_uzh.dbmdz-cased.orig.60.upos.cfg
python3 B_data-matrix_prep.py ../configs/B_alpino-noah_lsdc_uzh.dbmdz-uncased.orig.60.upos.cfg
python3 B_data-matrix_prep.py ../configs/B_alpino-noah_lsdc_uzh.bertje.orig.60.upos.cfg
python3 B_data-matrix_prep.py ../configs/B_alpino-noah_lsdc_uzh.mbert-cased.orig.60.upos.cfg
python3 B_data-matrix_prep.py ../configs/B_alpino-noah_lsdc_uzh.gbert-base.orig.60.upos.cfg
python3 B_data-matrix_prep.py ../configs/B_alpino-noah_lsdc_uzh.xlm-roberta.orig.60.upos.cfg
python3 B_data-matrix_prep.py ../configs/B_alpino-noah_lsdc_uzh.bert-base-cased.orig.60.upos.cfg
python3 B_data-matrix_prep.py ../configs/B_alpino-noah_lsdc_uzh.bert-base-uncased.orig.60.upos.cfg
python3 B_data-matrix_prep.py ../configs/B_alpino-noah_lsdc_uzh.finnish-bert-cased.orig.60.upos.cfg
```

4. Run baselines: (The results are saved in `results`, in folders named after the configs.)

```
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
```

5. Reformat results files:
```
python3 clean_up_results.py
```

6.
```
python3 dataset_stats.py
```
