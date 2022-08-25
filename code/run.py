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
    sys.stdout = cust_logger.Logger("eda")
    for arg in vars(args):
        print(arg, getattr(args, arg))

    tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-german-cased")

    # Prepare training/validation data: (modified) HRL tokens
    toks_td, pos_td = read_raw_input(args.file_td, args.n_sents_td)
    toks_orig_train, toks_orig_dev, pos_train, pos_dev = train_test_split(
        toks_td, pos_td, test_size=args.dev_ratio)

    train_data = Data("train_deu_orig", toks_orig=toks_orig_train,
                      pos_orig=pos_train)
    train_data.prepare_xy(tokenizer, args.max_len)
    train_data.save()
    print("Subtoken ratio (training)", train_data.subtok_ratio())

    dev_data = Data("dev_deu_orig", toks_orig=toks_orig_dev, pos_orig=pos_dev,
                    pos2idx=train_data.pos2idx)
    dev_data.prepare_xy(tokenizer, args.max_len)
    dev_data.save()
    print("Subtoken ratio (development)", dev_data.subtok_ratio())

    # TODO modify the tokens
    # visualize(x_train, f"x_train_{x_train.shape[0]}")
    # visualize(y_train, f"y_train_{x_train.shape[0]}")
    # visualize(real_pos_train, f"real_pos_train_{x_train.shape[0]}")
    # visualize(input_mask_train, f"input_mask_train_{x_train.shape[0]}")
    # visualize(x_dev, f"x_dev_{x_dev.shape[0]}")
    # visualize(y_dev, f"y_dev_{x_dev.shape[0]}")
    # visualize(real_pos_dev, f"real_pos_dev_{x_dev.shape[0]}")
    # visualize(input_mask_dev, f"input_mask_dev_{x_dev.shape[0]}")

    # Prepare test data: LRL tokens
    test_data = Data("test_gsw", raw_data_path=args.file_test,
                     max_sents=args.n_sents_test, pos2idx=train_data.pos2idx)
    test_data.prepare_xy(tokenizer, args.max_len)
    test_data.save()
    print("Subtoken ratio (test)", test_data.subtok_ratio())
    # visualize(x_test, f"x_test_{args.n_sents_test}")
    # visualize(y_test, f"y_test_{args.n_sents_test}")
    # visualize(y_train, f"y_train_{args.n_sents_test}")
    # visualize(real_pos_test, f"real_pos_test_{args.n_sents_test}")
    # visualize(input_mask_test, f"input_mask_test_{args.n_sents_test}")

    model = Model("dbmdz/bert-base-german-cased",
                  n_labels=len(train_data.pos2idx))
    print(model)
    if torch.cuda.is_available():
        model.cuda()  # TODO
        device = 'cuda'
    else:
        device = 'cpu'
    optimizer = AdamW(model.parameters())
    n_epochs = 2  # TODO move various settings to the console args
    batch_size = 32

    dataset_train = train_data.tensor_dataset()
    iter_train = DataLoader(dataset_train,
                            sampler=RandomSampler(dataset_train),
                            batch_size=batch_size)
    dataset_dev = dev_data.tensor_dataset()
    iter_dev = DataLoader(dataset_dev,
                          sampler=RandomSampler(dataset_dev),
                          batch_size=batch_size)
    dataset_test = test_data.tensor_dataset()
    iter_test = DataLoader(dataset_test,
                           sampler=RandomSampler(dataset_test),
                           batch_size=batch_size)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0,
        num_training_steps=len(iter_train) * n_epochs)

    model.finetune(device, iter_train, iter_dev, iter_test,
                   optimizer, scheduler, n_epochs)
