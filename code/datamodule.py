from data import Data, read_raw_input
from tokenizer import SCATokenizer

import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


class PosDataModule(pl.LightningDataModule):
    def __init__(self, config, pos2idx, traindev_sfx="", test_sfx=""):
        super().__init__()
        self.config = config
        self.pos2idx = pos2idx
        self.train_name = config.name_train + traindev_sfx
        self.dev_name = config.name_dev + traindev_sfx
        self.test_name = config.name_test + test_sfx
        self.test_sfx = test_sfx
        self.tokenizer = None
        self.use_sca_tokenizer = config.use_sca_tokenizer
        if self.use_sca_tokenizer:
            self.tokenizer = SCATokenizer(config.tokenizer_name)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.tokenizer_name)

    def prepare_data(self):
        # Training/validation data: HRL tokens
        if self.config.prepare_input_traindev:
            if self.config.orig_dir_train and self.config.orig_dir_dev:
                print("Constructing new data dirs based on existing dirs")
                train = Data(self.train_name,
                             other_dir=self.config.orig_dir_train)
                dev = Data(self.dev_name,
                           other_dir=self.config.orig_dir_dev)
            else:
                if self.config.orig_file_traindev:
                    print("Extracting data from traindev corpus and splitting "
                          "them into train vs. dev")
                    toks_td, pos_td = read_raw_input(
                        self.config.orig_file_traindev,
                        self.config.max_sents_traindev)
                    (toks_orig_train, toks_orig_dev,
                        pos_train, pos_dev) = train_test_split(
                        toks_td, pos_td, test_size=self.config.dev_ratio)
                    train = Data(self.train_name, toks_orig=toks_orig_train,
                                 pos_orig=pos_train, pos2idx=self.pos2idx)
                    dev = Data(self.dev_name, toks_orig=toks_orig_dev,
                               pos_orig=pos_dev, pos2idx=self.pos2idx)
                else:
                    print("Extracting data from train and dev corpora")
                    train = Data(self.train_name,
                                 raw_data_path=self.config.orig_file_train)
                    dev = Data(self.dev_name,
                               raw_data_path=self.config.orig_file_dev)
            # Prepare input matrices for finetuning
            alphabet = train.alphabet()
            train.add_noise(self.config.noise_type, self.config.noise_lvl_min,
                            self.config.noise_lvl_max, alphabet)
            train.prepare_xy(self.tokenizer, self.config.T,
                             self.config.subtoken_rep,
                             alias_tokenizer=self.use_sca_tokenizer)
            train.save(self.config.data_parent_dir)
            print(f"Subtoken ratio ({self.config.name_train}): {train.subtok_ratio(return_all=True)}")
            print(f"UNK ratio ({self.config.name_train}): {train.unk_ratio(return_all=True)}")
            print(f"Label distribution ({self.config.name_train}): {train.pos_y_distrib()}")
            dev.add_noise(self.config.noise_type, self.config.noise_lvl_min,
                          self.config.noise_lvl_max, alphabet)
            dev.prepare_xy(self.tokenizer, self.config.T,
                           self.config.subtoken_rep,
                           alias_tokenizer=self.use_sca_tokenizer)
            dev.save(self.config.data_parent_dir)
            print(f"Subtoken ratio ({self.config.name_dev}): {dev.subtok_ratio(return_all=True)}")
            print(f"UNK ratio ({self.config.name_dev}): {dev.unk_ratio(return_all=True)}")
            print(f"Label distribution ({self.config.name_dev}): {dev.pos_y_distrib()}")

        # Test data: LRL tokens
        self.multi_test = "," in self.config.test_name
        if self.config.prepare_input_test:
            # TODO multi-test
            test = Data(self.test_name,
                        raw_data_path=self.config.orig_file_test,
                        max_sents=self.config.max_sents_test,
                        pos2idx=self.pos2idx)
            test.prepare_xy(self.tokenizer, self.config.T,
                            self.config.subtoken_rep,
                            alias_tokenizer=self.use_sca_tokenizer)
            test.save(self.config.data_parent_dir)
            print(f"Subtoken ratio ({self.config.name_test}): {test.subtok_ratio(return_all=True)}")
            print(f"UNK ratio ({self.config.name_test}): {test.unk_ratio(return_all=True)}")
            print(f"Label distribution ({self.config.name_test}): {test.pos_y_distrib()}")
        else:
            # TODO multi-test
            test = Data(self.test_name,
                        load_parent_dir=self.config.data_parent_dir)


    def setup(self, stage):
        if stage == 'fit':
            self.train = Data(self.train_name,
                              load_parent_dir=self.config.data_parent_dir)
            self.val = Data(self.dev_name,
                            load_parent_dir=self.config.data_parent_dir)
        elif stage in ['test', 'predict']:
            self.test = Data(self.test_name,
                             load_parent_dir=self.config.data_parent_dir)

    def print_preview(self, data):
        print(data)
        idx2pos = data.idx2pos()
        for tok, pos_idx in zip(data.toks_bert[0], data.y[0]):
            print(tok, idx2pos[pos_idx])

    def train_dataloader(self):
        self.print_preview(self.train)
        return DataLoader(self.train.tensor_dataset(),
                          batch_size=self.config.batch_size)

    def val_dataloader(self):
        self.print_preview(self.val)
        return DataLoader(self.val.tensor_dataset(),
                          batch_size=self.config.batch_size)

    def test_dataloader(self):
        self.print_preview(self.test)
        return DataLoader(self.test.tensor_dataset(),
                          batch_size=self.config.batch_size)

    def predict_dataloader(self):
        self.print_preview(self.test)
        return DataLoader(self.test.tensor_dataset(),
                          batch_size=self.config.batch_size)
