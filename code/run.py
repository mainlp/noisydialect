from clean_up_results import average_scores
from config import Config
from datamodule import PosDataModule
from model import Classifier

from argparse import ArgumentParser
from pathlib import Path
import sys

import pytorch_lightning as pl
import torch
from transformers import BertTokenizer


def main(config_path, gpus=[0], dryrun=False,
         results_dir="../results", save_model=False, save_predictions=False,
         test_per_epoch=False):
    print(config_path)
    config = Config()
    out_dir = ""
    try:
        config.load(config_path)
        out_dir = f"{results_dir}/{config.config_name}/"
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        config.save(f"{out_dir}/{config.config_name}.cfg")
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

    if config.random_seeds:
        gen = enumerate(config.random_seeds)
    else:
        gen = enumerate(["unkseed"])
    for i, seed in gen:
        print(f"Run {i} of {len(config.random_seeds)} (seed {seed})")
        if config.random_seeds:
            deterministic = True
            pl.seed_everything(seed, workers=True)
        else:
            deterministic = None

        train_sfx, dev_sfx, test_sfx = "", "", ""
        if config.reinit_train_each_seed:
            train_sfx = "_" + str(seed)
        if config.reinit_dev_each_seed:
            dev_sfx = "_" + str(seed)
        if config.reinit_test_each_seed:
            test_sfx = "_" + str(seed)
        dm = PosDataModule(config, pos2idx,
                           train_sfx=train_sfx, dev_sfx=dev_sfx,
                           test_sfx=test_sfx)
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

        if config.name_dev:
            val_data_names = config.name_dev.split(",")
        else:
            val_data_names = []
        if test_per_epoch and config.name_test:
            val_data_names += [name for name in config.name_test.split(",")]

        model = Classifier(config.bert_name, config.plm_type, pos2idx,
                           config.classifier_dropout,
                           config.learning_rate, config.use_sca_tokenizer,
                           subtok2weight, val_data_names)

        if dryrun:
            # just checking if the code works
            dummy_trainer = pl.Trainer(accelerator='gpu', devices=gpus,
                                       fast_dev_run=True,)
            dummy_trainer.fit(model, datamodule=dm)
            dummy_trainer.validate(datamodule=dm, ckpt_path="last")
            dummy_trainer.test(datamodule=dm, ckpt_path="last")
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
            # test_dataloader() returns a list of dataloaders
            val_dataloaders = dm.val_dataloader() + dm.test_dataloader()
            trainer.fit(model, train_dataloaders=dm.train_dataloader(),
                        val_dataloaders=val_dataloaders)
        else:
            trainer.fit(model, datamodule=dm)
        # Training and validation scores:
        scores = {key: trainer.logged_metrics[key].item()
                  for key in trainer.logged_metrics}
        if save_model:
            torch.save(model.finetuning_model.state_dict(),
                       f"{out_dir}/model_{seed}.pt")

        if test_per_epoch:
            for i, val_name in enumerate(val_data_names):
                for epoch, (epoch_preds, epoch_gold) in enumerate(
                        zip(model.val_preds[i], model.val_gold[i])):
                    with open(
                            out_dir
                            + f"/predictions_{val_name}_{seed}_ep{epoch}.tsv",
                            "w", encoding="utf8") as f:
                        f.write("PREDICTED\tGOLD\n")
                        for p, g in zip(epoch_preds, epoch_gold):
                            f.write(f"{p}\t{g}\n")
                            # If there is an unexpected 0th epoch, it is
                            # due to model validation before training starts
        else:
            trainer.test(datamodule=dm, ckpt_path='last')
            # trainer.logged_metrics got re-initialized during trainer.test()
            scores.update({key: trainer.logged_metrics[key].item()
                           for key in trainer.logged_metrics})
            with open(f"{out_dir}/predictions_test_{seed}.tsv",
                      "w", encoding="utf8") as f:
                f.write("PREDICTED\tGOLD\n")
                for p, g in zip(model.test_preds, model.test_gold):
                    f.write(f"{p}\t{g}\n")

        with open(f"{out_dir}/results_{seed}.tsv", "w") as f:
            for metric in scores:
                if "_batch" not in metric:
                    f.write(f"{metric}\t{scores[metric]}\n")

        if save_predictions:
            predictions = trainer.predict(datamodule=dm)
            torch.save(predictions, f"{out_dir}/predictions_{seed}.pickle")

    # Average scores across initializations
    average_scores(out_dir)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-c", dest="config_path",
                        help="path to the configuration file",
                        default="")
    parser.add_argument("-g", dest="gpus",
                        help="GPU IDs", nargs="+", type=int, default=[0])
    parser.add_argument("-d", "--dryrun", action="store_true", dest="dryrun",
                        default=False)
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
    main(args.config_path, args.gpus, args.dryrun,
         args.results_dir, args.save_model, args.save_predictions,
         args.test_per_epoch)
