import cust_logger
from config import Config
from data import Data, read_raw_input
from model import Model

from argparse import ArgumentParser
import sys

from sklearn.model_selection import train_test_split
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler
from transformers import AutoTokenizer, get_linear_schedule_with_warmup


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-c", dest="config_path",
                        help="path to the configuration file",
                        default="")
    # parser.add_argument("-q", "--quiet", action="store_false", dest="verbose",
    #                     default=True, help="no messages to stdout")

    args = parser.parse_args()
    config = Config()
    try:
        config.load(args.config_path)
        config.save(args.config_path)
    except FileNotFoundError:
        print("Couldn't find config (using standard config)")
    sys.stdout = cust_logger.Logger("run_" + config.name_train,
                                    include_timestamp=True)
    print(args.config_path)
    print(config)

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

    # --- Continued pre-training ---
    # TODO

    # --- Finetuning ---

    # Training/validation data: HRL tokens
    if not config.prepare_input_traindev:
        train = Data(config.name_train, load_parent_dir=config.data_parent_dir)
        dev = Data(config.name_dev, load_parent_dir=config.data_parent_dir)
    elif config.orig_dir_train is not None:
        train = Data(config.name_train, other_dir=config.orig_dir_train)
        dev = Data(config.name_dev, other_dir=config.orig_dir_dev)
    else:
        toks_td, pos_td = read_raw_input(config.orig_file_traindev,
                                         config.max_sents_traindev)
        toks_orig_train, toks_orig_dev, pos_train, pos_dev = train_test_split(
            toks_td, pos_td, test_size=config.dev_ratio)
        train = Data(config.name_train, toks_orig=toks_orig_train,
                     pos_orig=pos_train)
        dev = Data(config.name_dev, toks_orig=toks_orig_dev,
                   pos_orig=pos_dev, pos2idx=train.pos2idx)

    # Test data: LRL tokens
    if config.orig_file_test:
        test = Data(config.name_test, raw_data_path=config.orig_file_test,
                    max_sents=config.max_sents_test,
                    pos2idx=train.pos2idx)
        test.prepare_xy(tokenizer, config.T, config.subtoken_rep)
        test.save(config.data_parent_dir)
        print(f"Subtoken ratio ({config.name_test}): {test.subtok_ratio(return_all=True)}\n")
    else:
        test = Data(config.name_test, load_parent_dir=config.data_parent_dir)
    alphabet_test = test.alphabet()

    # Prepare input matrices for finetuning
    if config.prepare_input_traindev:
        train.add_noise(config.noise_type, config.noise_lvl_min,
                        config.noise_lvl_max, alphabet_test)
        train.prepare_xy(tokenizer, config.T, config.subtoken_rep)
        train.save(config.data_parent_dir)
        print(f"Subtoken ratio ({config.name_train}): {train.subtok_ratio(return_all=True)}\n")
        dev.add_noise(config.noise_type, config.noise_lvl_min,
                      config.noise_lvl_max, alphabet_test)
        dev.prepare_xy(tokenizer, config.T, config.subtoken_rep)
        dev.save(config.data_parent_dir)
        print(f"Subtoken ratio ({config.name_dev}): {dev.subtok_ratio(return_all=True)}\n")

    # visualize(x_test, f"x_test_{args.n_sents_test}")

    model = Model(config.bert_name, pos2idx=train.pos2idx,
                  classifier_dropout=config.classifier_dropout)

    # TODO continued pretraining

    # Finetuning
    if torch.cuda.is_available():
        model.finetuning_model.cuda()
        device = 'cuda'
    else:
        device = 'cpu'
    print("Device", device)
    optimizer = AdamW(model.finetuning_model.parameters())

    dataset_train = train.tensor_dataset()
    iter_train = DataLoader(dataset_train,
                            sampler=RandomSampler(dataset_train),
                            batch_size=config.batch_size)
    dataset_dev = dev.tensor_dataset()
    iter_dev = DataLoader(dataset_dev,
                          sampler=RandomSampler(dataset_dev),
                          batch_size=config.batch_size)
    dataset_test = test.tensor_dataset()
    iter_test = DataLoader(dataset_test,
                           sampler=RandomSampler(dataset_test),
                           batch_size=config.batch_size)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0,
        num_training_steps=len(iter_train) * config.n_epochs)

    model.finetune(device, iter_train, iter_dev, iter_test,
                   optimizer, scheduler, config.n_epochs, tokenizer,
                   train.dummy_idx())
