import cust_logger
from data import Data, read_raw_input
from model import Model

from argparse import ArgumentParser
import sys

from sklearn.model_selection import train_test_split
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler
from transformers import AutoTokenizer, BertForTokenClassification, \
    get_linear_schedule_with_warmup


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-r", dest="file_td",
                        help="training/development data file",
                        default="../datasets/hamburg-dependency-treebank/"
                                "train_DHT_STTS.txt")
    parser.add_argument("-e", dest="file_test",
                        help="test data file",
                        default="../datasets/NOAH-corpus/test_GSW_STTS.txt")
    parser.add_argument("-nr", dest="n_sents_td",
                        help="no. of sentences in the training/dev data",
                        default=206794, type=int)
    parser.add_argument("-ne", dest="n_sents_test",
                        help="no. of sentences in the test data",
                        default=7320, type=int)
    parser.add_argument("-m", dest="max_len",
                        help="max. number of tokens per sentence "
                        "(incl. CLS and SEP)",
                        default=54, type=int)
    parser.add_argument("-d", dest="dev_ratio",
                        help="dev:dev+train ratio",
                        default=0.1, type=float)
    # parser.add_argument("-q", "--quiet", action="store_false", dest="verbose",
    #                     default=True, help="no messages to stdout")

    args = parser.parse_args()
    sys.stdout = cust_logger.Logger("run", include_timestamp=True)
    for arg in vars(args):
        print(arg, getattr(args, arg))

    tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-german-cased")

    # --- Continued pre-training ---
    # TODO

    # --- Finetuning ---

    # Prepare training/validation data: (modified) HRL tokens
    toks_td, pos_td = read_raw_input(args.file_td, args.n_sents_td)
    toks_orig_train, toks_orig_dev, pos_train, pos_dev = train_test_split(
        toks_td, pos_td, test_size=args.dev_ratio)

    # Unmodified tokens for finetuning
    train_deu_orig = Data("train_deu_orig", toks_orig=toks_orig_train,
                          pos_orig=pos_train)
    train_deu_orig.prepare_xy(tokenizer, args.max_len)
    train_deu_orig.save()
    print("Subtoken ratio (train_deu_orig)",
          train_deu_orig.subtok_ratio(return_all=True))
    dev_deu_orig = Data("dev_deu_orig", toks_orig=toks_orig_dev,
                        pos_orig=pos_dev, pos2idx=train_deu_orig.pos2idx)
    dev_deu_orig.prepare_xy(tokenizer, args.max_len)
    dev_deu_orig.save()
    print("Subtoken ratio (dev_deu_orig)",
          dev_deu_orig.subtok_ratio(return_all=True))

    # Test data: LRL tokens
    test_gsw = Data("test_gsw", raw_data_path=args.file_test,
                    max_sents=args.n_sents_test,
                    pos2idx=train_deu_orig.pos2idx)
    test_gsw.prepare_xy(tokenizer, args.max_len)
    test_gsw.save()
    print("Subtoken ratio (test)", test_gsw.subtok_ratio(return_all=True))
    # visualize(x_test, f"x_test_{args.n_sents_test}")
    # visualize(y_test, f"y_test_{args.n_sents_test}")
    # visualize(y_train, f"y_train_{args.n_sents_test}")
    # visualize(real_pos_test, f"real_pos_test_{args.n_sents_test}")
    # visualize(input_mask_test, f"input_mask_test_{args.n_sents_test}")
    alphabet_gsw = test_gsw.alphabet()

    # Tokens with random noise for finetuning
    train_deu_rand = Data("train_deu_rand",
                          toks_orig=train_deu_orig.copy_toks_orig(),
                          pos_orig=train_deu_orig.copy_pos_orig(),
                          pos2idx=train_deu_orig.pos2idx)
    train_deu_rand.add_random_noise(0.1, 0.15, alphabet_gsw)
    train_deu_rand.prepare_xy(tokenizer, args.max_len)
    train_deu_rand.save()
    print("Subtoken ratio (train_deu_rand)",
          train_deu_rand.subtok_ratio(return_all=True))
    dev_deu_rand = Data("dev_deu_rand",
                        toks_orig=dev_deu_orig.copy_toks_orig(),
                        pos_orig=dev_deu_orig.copy_pos_orig(),
                        pos2idx=train_deu_orig.pos2idx)
    dev_deu_rand.add_random_noise(0.1, 0.15, alphabet_gsw)
    dev_deu_rand.prepare_xy(tokenizer, args.max_len)
    dev_deu_rand.save()
    print("Subtoken ratio (dev_deu_rand)",
          dev_deu_rand.subtok_ratio(return_all=True))

    # Tokens with somewhat less random noise for finetuning
    train_deu_gen = Data("train_deu_gen",
                         toks_orig=train_deu_orig.copy_toks_orig(),
                         pos_orig=train_deu_orig.copy_pos_orig(),
                         pos2idx=train_deu_orig.pos2idx)
    train_deu_gen.add_custom_noise_general(0.1, 0.15, alphabet_gsw)
    train_deu_gen.prepare_xy(tokenizer, args.max_len)
    train_deu_gen.save()
    print("Subtoken ratio (train_deu_gen)",
          train_deu_gen.subtok_ratio(return_all=True))
    dev_deu_gen = Data("dev_deu_gen",
                       toks_orig=dev_deu_orig.copy_toks_orig(),
                       pos_orig=dev_deu_orig.copy_pos_orig(),
                       pos2idx=train_deu_orig.pos2idx)
    dev_deu_gen.add_custom_noise_general(0.1, 0.15, alphabet_gsw)
    dev_deu_gen.prepare_xy(tokenizer, args.max_len)
    dev_deu_gen.save()
    print("Subtoken ratio (dev_deu_gen)",
          dev_deu_gen.subtok_ratio(return_all=True))

    # Tokens with handcrafted GSW-ifying features for finetuning
    train_deu_gsw_low = Data("train_deu_gsw_low",
                             toks_orig=train_deu_orig.copy_toks_orig(),
                             pos_orig=train_deu_orig.copy_pos_orig(),
                             pos2idx=train_deu_orig.pos2idx)
    train_deu_gsw_low.add_custom_noise_general(0.1, 0.15, alphabet_gsw)
    train_deu_gsw_low.prepare_xy(tokenizer, args.max_len)
    train_deu_gsw_low.save()
    print("Subtoken ratio (train_deu_gsw_low)",
          train_deu_gsw_low.subtok_ratio(return_all=True))
    dev_deu_gsw_low = Data("dev_deu_gsw_low",
                           toks_orig=dev_deu_orig.copy_toks_orig(),
                           pos_orig=dev_deu_orig.copy_pos_orig(),
                           pos2idx=train_deu_orig.pos2idx)
    dev_deu_gsw_low.add_custom_noise_general(0.1, 0.15, alphabet_gsw)
    dev_deu_gsw_low.prepare_xy(tokenizer, args.max_len)
    dev_deu_gsw_low.save()
    print("Subtoken ratio (dev_deu_gsw_low)",
          dev_deu_gsw_low.subtok_ratio(return_all=True))
    train_deu_gsw_mid = Data("train_deu_gsw_mid",
                             toks_orig=train_deu_orig.copy_toks_orig(),
                             pos_orig=train_deu_orig.copy_pos_orig(),
                             pos2idx=train_deu_orig.pos2idx)
    train_deu_gsw_mid.add_custom_noise_general(0.5, 0.5, alphabet_gsw)
    train_deu_gsw_mid.prepare_xy(tokenizer, args.max_len)
    train_deu_gsw_mid.save()
    print("Subtoken ratio (train_deu_gsw_mid)",
          train_deu_gsw_mid.subtok_ratio(return_all=True))
    dev_deu_gsw_mid = Data("dev_deu_gsw_mid",
                           toks_orig=dev_deu_orig.copy_toks_orig(),
                           pos_orig=dev_deu_orig.copy_pos_orig(),
                           pos2idx=train_deu_orig.pos2idx)
    dev_deu_gsw_mid.add_custom_noise_general(0.5, 0.5, alphabet_gsw)
    dev_deu_gsw_mid.prepare_xy(tokenizer, args.max_len)
    dev_deu_gsw_mid.save()
    print("Subtoken ratio (dev_deu_gsw_mid)",
          dev_deu_gsw_mid.subtok_ratio(return_all=True))
    train_deu_gsw_high = Data("train_deu_gsw_high",
                              toks_orig=train_deu_orig.copy_toks_orig(),
                              pos_orig=train_deu_orig.copy_pos_orig(),
                              pos2idx=train_deu_orig.pos2idx)
    train_deu_gsw_high.add_custom_noise_general(1.0, 1.0, alphabet_gsw)
    train_deu_gsw_high.prepare_xy(tokenizer, args.max_len)
    train_deu_gsw_high.save()
    print("Subtoken ratio (train_deu_gsw_high)",
          train_deu_gsw_high.subtok_ratio(return_all=True))
    dev_deu_gsw_high = Data("dev_deu_gsw_high",
                            toks_orig=dev_deu_orig.copy_toks_orig(),
                            pos_orig=dev_deu_orig.copy_pos_orig(),
                            pos2idx=train_deu_orig.pos2idx)
    dev_deu_gsw_high.add_custom_noise_general(1.0, 1.0, alphabet_gsw)
    dev_deu_gsw_high.prepare_xy(tokenizer, args.max_len)
    dev_deu_gsw_high.save()
    print("Subtoken ratio (dev_deu_gsw_high)",
          dev_deu_gsw_high.subtok_ratio(return_all=True))

    model = Model("dbmdz/bert-base-german-cased",
                  n_labels=len(train_deu_orig.pos2idx))
    print(model)
    if torch.cuda.is_available():
        model.cuda()  # TODO
        device = 'cuda'
    else:
        device = 'cpu'
    optimizer = AdamW(model.parameters())
    n_epochs = 2  # TODO move various settings to the console args
    batch_size = 32

    dataset_train = train_deu_orig.tensor_dataset()
    iter_train = DataLoader(dataset_train,
                            sampler=RandomSampler(dataset_train),
                            batch_size=batch_size)
    dataset_dev = dev_deu_orig.tensor_dataset()
    iter_dev = DataLoader(dataset_dev,
                          sampler=RandomSampler(dataset_dev),
                          batch_size=batch_size)
    dataset_test = test_gsw.tensor_dataset()
    iter_test = DataLoader(dataset_test,
                           sampler=RandomSampler(dataset_test),
                           batch_size=batch_size)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0,
        num_training_steps=len(iter_train) * n_epochs)

    model.finetune(device, iter_train, iter_dev, iter_test,
                   optimizer, scheduler, n_epochs)
