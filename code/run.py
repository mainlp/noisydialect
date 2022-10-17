from config import Config
from data import PosDataModule
from model import Classifier

from argparse import ArgumentParser
import glob
from pathlib import Path
import sys

import pytorch_lightning as pl
import torch
from transformers import BertTokenizer


def main(config_path, gpus=[0], dryrun=False, seeds=[],
         results_dir="../results", save_model=False, save_predictions=False,
         test_per_epoch=False):
    print(config_path)
    config = Config()
    out_dir = ""
    try:
        config.load(config_path)
        out_dir = f"{results_dir}/{config.config_name}/"
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        config.save(f"{out_dir}/config.cfg")
    except FileNotFoundError:
        print("Couldn't find config (quitting)")
        sys.exit(1)
    print(config)

    with open(config.tagset_path, encoding="utf8") as f:
        pos2idx = {}
        for i, line in enumerate(f):
            line = line.strip()
            if line:
                pos2idx[line] = i

    if config.use_sca_tokenizer and config.sca_sibling_weighting == 'relative':
        orig_tokenizer = BertTokenizer.from_pretrained(config.tokenizer_name)

    if seeds:
        gen = enumerate(seeds)
    else:
        gen = enumerate(["unkseed"])
    for i, seed in gen:
        print(f"Run {i} of {len(seeds)} (seed {seed})")
        if seeds:
            deterministic = True
            pl.seed_everything(seed, workers=True)
        else:
            deterministic = None

        dm = PosDataModule(config, pos2idx)
        subtok2weight = None
        if config.use_sca_tokenizer and \
                config.sca_sibling_weighting == 'relative':
            dm.prepare_data()
            # Don't reload and re-prepare the input data when the classifier
            # is trained/evaluated (especially since this could result in a
            # new train--dev split):
            dm.config.prepare_input_traindev = False
            dm.config.prepare_input_test = False
            dm.setup("fit")
            subtok2weight = dm.train.get_subtoken_sibling_distribs(
                dm.tokenizer, orig_tokenizer)

        model = Classifier(config.bert_name, pos2idx,
                           config.classifier_dropout,
                           config.learning_rate, config.use_sca_tokenizer,
                           subtok2weight)

        if dryrun:
            # just checking if the code works
            dummy_trainer = pl.Trainer(accelerator='gpu', devices=gpus,
                                       fast_dev_run=True,)
            dummy_trainer.fit(model, datamodule=dm)
            dummy_trainer.validate(datamodule=dm)
            dummy_trainer.test(datamodule=dm)
            return

        # --- Continued pre-training ---
        # TODO

        # --- Finetuning ---
        if save_model:
            default_root_dir = f"{out_dir}/checkpoints/"
            Path(default_root_dir).mkdir(parents=True, exist_ok=True)
        else:
            default_root_dir = None
        trainer = pl.Trainer(accelerator='gpu', devices=gpus,
                             max_epochs=config.n_epochs,
                             deterministic=deterministic,
                             default_root_dir=default_root_dir)
        if test_per_epoch:
            dm.prepare_data()
            dm.setup("fit")
            dm.setup("test")
            trainer.fit(model, train_dataloaders=dm.train_dataloader(),
                        val_dataloaders=[dm.val_dataloader(),
                                         dm.test_dataloader()])
        else:
            trainer.fit(model, datamodule=dm)
        # Training and validation scores:
        scores = {key: trainer.logged_metrics[key].item()
                  for key in trainer.logged_metrics}
        if save_model:
            torch.save(model.finetuning_model.state_dict(),
                       f"{out_dir}/model_{seed}.pt")

        if not test_per_epoch:
            trainer.test(datamodule=dm)
            # trainer.logged_metrics got re-initialized during trainer.test
            scores.update({key: trainer.logged_metrics[key].item()
                           for key in trainer.logged_metrics})

        with open(f"{out_dir}/results_{seed}.tsv", "w") as f:
            for metric in scores:
                if "_batch" not in metric:
                    f.write(f"{metric}\t{scores[metric]}\n")

        if save_predictions:
            predictions = trainer.predict(datamodule=dm)
            torch.save(predictions, f"{out_dir}/predictions_{seed}.pickle")

    # Average scores across initializations
    scores_all = {}
    for res_file in glob.glob(f"{out_dir}/results*.tsv"):
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
    with open(f"{out_dir}/results_AVG.tsv", "w") as f:
        for metric in scores_all:
            n_runs = len(scores_all[metric])
            f.write(f"{metric}\t{sum(scores_all[metric]) / n_runs}\n")
            print(metric, sum(scores_all[metric]) / n_runs,
                  str(n_runs) + " run(s)")
        f.write(f"n_runs\t{n_runs}\n")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-c", dest="config_path",
                        help="path to the configuration file",
                        default="")
    parser.add_argument("-g", dest="gpus",
                        help="GPU IDs", nargs="+", type=int,
                        default=[0])
    parser.add_argument("-d", "--dryrun", action="store_true", dest="dryrun",
                        default=False)
    parser.add_argument("-s", dest="seeds", nargs="+", type=int,
                        default=[12345, 23456, 34567, 45678, 56789],
                        help="(ignored if dryrun == True)")
    parser.add_argument("--test_per_epoch", action="store_true", default=False,
                        help="compute test set performance after each epoch")
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--dont_save_model', dest='save_model',
                        action='store_false')
    parser.add_argument('--save_predictions', action='store_true')
    parser.add_argument('--dont_save_predictions', dest='save_predictions',
                        action='store_false')
    parser.add_argument("-r", dest="results_dir", default="../results",
                        help="the parent directory in which the directory "
                        "with the results should be saved")
    parser.set_defaults(save_model=False, save_predictions=False)
    args = parser.parse_args()
    main(args.config_path, args.gpus, args.dryrun, args.seeds,
         args.results_dir, args.save_model, args.save_predictions,
         args.test_per_epoch)
