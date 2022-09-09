import cust_logger
from config import Config
from data import PosDataModule
from model import Classifier

from argparse import ArgumentParser
import sys

import pytorch_lightning as pl


def main(config_path, dryrun=False):
    print(config_path)
    config = Config()
    try:
        config.load(config_path)
        config.save(config_path)
    except FileNotFoundError:
        print("Couldn't find config (using standard config)")
    sys.stdout = cust_logger.Logger("run_" + config.name_train,
                                    include_timestamp=True)
    print(config)

    # TODO put path into config
    with open("tagset_stts.txt", encoding="utf8") as f:
        pos2idx = {}
        for i, line in enumerate(f):
            line = line.strip()
            if line:
                pos2idx[line] = i

    dm = PosDataModule(config, pos2idx)
    model = Classifier(config.bert_name, pos2idx, config.classifier_dropout,
                       config.learning_rate)

    if dryrun:
        # just checking if the code works
        dummy_trainer = pl.Trainer(accelerator='gpu', gpus=[1],
                                   fast_dev_run=True,)
        dummy_trainer.fit(model, datamodule=dm)
        return

    # TODO gpu id
    trainer = pl.Trainer(accelerator='gpu', gpus=[6],
                         max_epochs=config.n_epochs)
    trainer.fit(model, datamodule=dm)
    trainer.validate(datamodule=dm)
    trainer.test(datamodule=dm)

    # --- Continued pre-training ---
    # TODO

    # --- Finetuning ---

    dm = PosDataModule(config, pos2idx)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-c", dest="config_path",
                        help="path to the configuration file",
                        default="")
    # parser.add_argument("-q", "--quiet", action="store_false", dest="verbose",
    #                     default=True, help="no messages to stdout")
    args = parser.parse_args()
    main(args.config_path)
