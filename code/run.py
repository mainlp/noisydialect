import cust_logger
from config import Config
from data import PosDataModule
from model import Classifier

from argparse import ArgumentParser
import sys

import pytorch_lightning as pl


def main(config_path, gpus=[0], dryrun=False):
    print(config_path)
    config = Config()
    try:
        config.load(config_path)
        config.save(config_path)
    except FileNotFoundError:
        print("Couldn't find config (quitting)")
        sys.exit(1)
    sys.stdout = cust_logger.Logger("run_" + config.name_train,
                                    include_timestamp=True)
    print(config)

    # TODO put path into config
    with open(config.tagset_path, encoding="utf8") as f:
        pos2idx = {}
        for i, line in enumerate(f):
            line = line.strip()
            if line:
                pos2idx[line] = i

    dm = PosDataModule(config, pos2idx)
    model = Classifier(config.bert_name, pos2idx, config.classifier_dropout,
                       config.learning_rate, config.use_sca_tokenizer)

    if dryrun:
        # just checking if the code works
        dummy_trainer = pl.Trainer(accelerator='gpu', devices=gpus,
                                   fast_dev_run=True,)
        dummy_trainer.fit(model, datamodule=dm)
        dummy_trainer.validate(datamodule=dm, ckpt_path="last")
        dummy_trainer.test(datamodule=dm, ckpt_path="last")
        return

    trainer = pl.Trainer(accelerator='gpu', devices=gpus,
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
    parser.add_argument("-g", dest="gpus",
                        help="GPU IDs", nargs="+", type=int,
                        default=[0])
    parser.add_argument("-d", "--dryrun", action="store_true", dest="dryrun",
                        default=False)
    args = parser.parse_args()
    main(args.config_path, args.gpus, args.dryrun)
