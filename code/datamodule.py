from data import Data, read_raw_input
from tokenizer import SCATokenizer

import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


class PosDataModule(pl.LightningDataModule):
    def __init__(self, config, pos2idx, train_sfx="", dev_sfx="", test_sfx=""):
        super().__init__()
        self.config = config
        self.pos2idx = pos2idx
        self.train_name = config.name_train + train_sfx
        if config.name_dev:
            self.dev_names = [name_dev + test_sfx
                              for name_dev in config.name_dev.split(",")]
        else:
            self.dev_names = []
        if config.name_test:
            self.test_names = [name_test + test_sfx
                               for name_test in config.name_test.split(",")]
        else:
            self.test_names = []
        self.test_sfx = test_sfx
        self.tokenizer = None
        self.use_sca_tokenizer = config.use_sca_tokenizer
        if self.use_sca_tokenizer:
            self.tokenizer = SCATokenizer(config.tokenizer_name)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.tokenizer_name)

    def idx2pos(self):
        return {self.pos2idx[pos]: pos for pos in self.pos2idx}

    def prepare_data(self):
        # Training/validation data: HRL tokens

        if self.config.prepare_input_train:
            if self.config.orig_dir_train:
                print("Constructing new train data dir based on existing")
                train = Data(self.train_name,
                             other_dir=self.config.orig_dir_train)
            else:
                print("Extracting data from train corpus")
                train = Data(self.train_name, pos2idx=self.pos2idx,
                             raw_data_path=self.config.orig_file_train,
                             max_sents=self.config.max_sents_train,
                             subset_selection=self.config.subset_selection)
            alphabet = train.alphabet()
            train.add_noise(self.config.noise_type, self.config.noise_lvl,
                            alphabet)
            train.prepare_xy(self.tokenizer, self.config.T,
                             self.config.subtoken_rep,
                             alias_tokenizer=self.use_sca_tokenizer)
            train.save(self.config.data_parent_dir)
            print(f"Subtoken ratio ({self.train_name}): {train.subtok_ratio(return_all=True)}")
            print(f"UNK ratio ({self.train_name}): {train.unk_ratio(return_all=True)}")
            print(f"Label distribution ({self.train_name}): {train.pos_y_distrib()}")

        if self.config.prepare_input_dev:
            # if not alphabet:
            #     train = Data(self.train_name,
            #                  load_parent_dir=self.config.data_parent_dir)
            #     alphabet = train.alphabet()
            for dev_name, orig_file_dev in [
                    i for i in zip(self.dev_names,
                                   self.config.orig_file_dev.split(","))]:
                if self.config.orig_dir_dev:
                    print("Constructing new dev data dir based on existing")
                    dev = Data(dev_name, other_dir=self.config.orig_dir_dev)
                else:
                    print("Extracting data from dev corpus")
                    dev = Data(dev_name, pos2idx=self.pos2idx,
                               raw_data_path=orig_file_dev,
                               max_sents=self.config.max_sents_dev,
                               subset_selection=self.config.subset_selection)
                # dev.add_noise(self.config.noise_type,
                #               self.config.noise_lvl, alphabet)
                dev.prepare_xy(self.tokenizer, self.config.T,
                               self.config.subtoken_rep,
                               alias_tokenizer=self.use_sca_tokenizer)
                dev.save(self.config.data_parent_dir)
                print(f"Subtoken ratio ({dev_name}): {dev.subtok_ratio(return_all=True)}")
                print(f"UNK ratio ({dev_name}): {dev.unk_ratio(return_all=True)}")
                print(f"Label distribution ({dev_name}): {dev.pos_y_distrib()}")

        # Test data: LRL tokens
        if self.config.prepare_input_test:
            for test_name, orig_file_test in [
                    i for i in zip(self.test_names,
                                   self.config.orig_file_test.split(","))]:
                test = Data(test_name,
                            raw_data_path=orig_file_test,
                            pos2idx=self.pos2idx,
                            max_sents=self.config.max_sents_test,
                            subset_selection=self.config.subset_selection)
                test.prepare_xy(self.tokenizer, self.config.T,
                                self.config.subtoken_rep,
                                alias_tokenizer=self.use_sca_tokenizer)
                test.save(self.config.data_parent_dir)
                print(f"Subtoken ratio ({test_name}): {test.subtok_ratio(return_all=True)}")
                print(f"UNK ratio ({test_name}): {test.unk_ratio(return_all=True)}")
                print(f"Label distribution ({test_name}): {test.pos_y_distrib()}")

    def setup(self, stage):
        if stage == 'fit':
            self.train = Data(self.train_name,
                              load_parent_dir=self.config.data_parent_dir)
            if self.dev_names:
                self.vals = [Data(
                    dev_name, load_parent_dir=self.config.data_parent_dir)
                    for dev_name in self.dev_names]
            else:
                self.vals = []
        elif stage in ['test', 'predict']:
            if self.test_names:
                self.tests = [Data(
                    test_name, load_parent_dir=self.config.data_parent_dir)
                    for test_name in self.test_names]
            else:
                self.tests = []

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
        for val in self.vals:
            self.print_preview(val)
        return [DataLoader(val.tensor_dataset(),
                           batch_size=self.config.batch_size)
                for val in self.vals]

    def test_dataloader(self):
        for test in self.tests:
            self.print_preview(test)
        return [DataLoader(test.tensor_dataset(),
                           batch_size=self.config.batch_size)
                for test in self.tests]

    def predict_dataloader(self):
        self.print_preview(self.test)
        return DataLoader(self.test.tensor_dataset(),
                          batch_size=self.config.batch_size)
