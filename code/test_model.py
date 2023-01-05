from config import Config
from datamodule import PosDataModule
from model import Classifier

import sys

import numpy as np
import pytorch_lightning as pl
import torch


def load_config_and_data(config_name, results_folder="../results/"):
    config = Config()
    config.load(results_folder + config_name + "/" + config_name + ".cfg")
    pos2idx = {}
    with open(config.tagset_path, encoding="utf8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line:
                pos2idx[line] = i
    dm = PosDataModule(config, pos2idx)
    dm.setup("test")
    return config, pos2idx, dm


def load_model(config, pos2idx, datamodule, seed,
               results_folder="../results/"):
    dir_name = results_folder + config.config_name
    subtok2weight = None
    model = Classifier(pretrained_model_name_or_path=config.bert_name,
                       plm_type=config.plm_type,
                       pos2idx=pos2idx,
                       classifier_dropout=config.classifier_dropout,
                       learning_rate=config.learning_rate,
                       use_sca_embeddings=config.use_sca_tokenizer,
                       subtok2weight=subtok2weight,
                       test_data_names=datamodule.test_names,)
    weights_path = dir_name + "/model_" + seed + ".pt"
    weights = torch.load(weights_path)
    model.finetuning_model.load_state_dict(weights)
    return model


def test_model(model, datamodule, seed, out_dir):
    trainer = pl.Trainer()
    trainer.test(model=model, datamodule=datamodule, verbose=True)
    for name, (preds, gold) in zip(datamodule.test_names,
                                   model.get_test_predictions()):
        with open(out_dir + f"/predictions_{name}_{seed}_epochx.tsv",
                  "w", encoding="utf8") as f:
            f.write("PREDICTED\tGOLD\n")
            for p, g in zip(preds, gold):
                f.write(f"{p}\t{g}\n")
    return model.test_results()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("USAGE: python test_model.py CONFIG_NAME")
        sys.exit(1)

    # Load config + datamodule
    results_folder = "../results/"
    config_name = sys.argv[1]
    config, pos2idx, dm = load_config_and_data(config_name, results_folder)
    # Get predictions for each of the model initializations:
    acc = [[] for _ in dm.test_names]
    f1 = [[] for _ in dm.test_names]
    seeds = ("12345", "23456", "34567", "45678", "56789")
    out_dir = results_folder + config_name
    for seed in seeds:
        print("Seed", seed)
        model = load_model(config, pos2idx, dm, seed)
        print("Loaded model")
        results = test_model(model, dm, seed, out_dir)
        print(results)
        for i, (acc_seed, f1_seed) in enumerate(results):
            acc[i].append(acc_seed)
            f1[i].append(f1_seed)
            n_runs = len(seeds)
    with open(out_dir + "/results_test_AVG.tsv", "w", encoding="utf8") as f:
        f.write("METRIC\tAVERAGE\tSTD_DEV\tN_RUNS\n")
        for name, a, f in zip(dm.test_names, acc, f1):
            acc_avg = sum(a) / n_runs
            acc_std = np.std(a)
            f1_avg = sum(f) / n_runs
            f1_std = np.std(f)
            print(name, acc_avg, acc_std, f1_avg, f1_std)
            f.write(f"{name}_acc\t{acc_avg}\t{acc_std}\t{n_runs}\n")
            f.write(f"{name}_f1\t{f1_avg}\t{f1_std}\t{n_runs}\n")
