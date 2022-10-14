from config import Config
from data import PosDataModule
from model import Classifier

from argparse import ArgumentParser
from pathlib import Path
import sys

import pytorch_lightning as pl
import torch
from transformers import BertTokenizer


def main(config_path, gpus=[0], dryrun=False, n_runs=1, save_model=False):
    print(config_path)
    config = Config()
    try:
        config.load(config_path)
        Path("../results/{config.config_name}/").mkdir(parents=True,
                                                       exist_ok=True)
        config.save("../results/{config.config_name}/config.cfg")
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

    dm = PosDataModule(config, pos2idx)
    subtok2weight = None
    if config.use_sca_tokenizer and config.sca_sibling_weighting == 'relative':
        dm.prepare_data()
        # Don't reload and re-prepare the input data when the classifier
        # is trained/evaluated (especially since this could result in a
        # new train--dev split):
        dm.config.prepare_input_traindev = False
        dm.config.prepare_input_test = False
        dm.setup("fit")
        orig_tokenizer = BertTokenizer.from_pretrained(config.tokenizer_name)
        subtok2weight = dm.train.get_subtoken_sibling_distribs(dm.tokenizer,
                                                               orig_tokenizer)

    for i in range(n_runs):
        print("Run {i} of {n_runs}")
        model = Classifier(config.bert_name, pos2idx,
                           config.classifier_dropout,
                           config.learning_rate, config.use_sca_tokenizer,
                           subtok2weight)

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
        trainer = pl.Trainer(accelerator='gpu', devices=gpus,
                             max_epochs=config.n_epochs)
        trainer.fit(model, datamodule=dm)
        if save_model:
            torch.save(model.finetuning_model.state_dict(),
                       "../results/{config.config_name}/model_{i}.pt")
        trainer.validate(datamodule=dm)
        trainer.test(datamodule=dm)
        predictions = trainer.predict(datamodule=dm)
        torch.save(predictions,
                   "../results/{config.config_name}/predictions_{i}.pickle")


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
    parser.add_argument("-n", dest="n_runs",
                        help="Number of model initializations "
                             "(1 if dryrun==True)",
                        type=int, default=5)
    parser.add_argument('--save_model', default=None,
                        help="File where model should be saved (no extension)")
    args = parser.parse_args()
    main(args.config_path, args.gpus, args.dryrun, args.n_runs,
         args.save_model)
