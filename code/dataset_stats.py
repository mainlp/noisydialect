from glob import glob

from data import Data


def dataset_stats(tagset, out_file, data_folder="../data/"):
    with open(out_file, "w+", encoding="utf8") as f_out:
        f_out.write("DATASET\tN_SENTS\tMAX_SENT_TOKS\tSUBTOKS_PER_TOK\t"
                    "UNKS_PER_SUBTOK\tLABEL_DISTRIBUTION\n")
        for path in glob(data_folder + "*" + tagset):
            data = Data(name=path.split("/")[-1], load_parent_dir=data_folder)
            infos = [data.name, *data.x.shape, data.subtok_ratio(),
                     data.unk_ratio(), data.pos_y_distrib()]
            print(infos)
            f_out.write("\t".join([str(info) for info in infos]))
            f_out.write("\n")


if __name__ == "__main__":
    dataset_stats("stts", "../results/data_statistics_stts.tsv")
    dataset_stats("upos", "../results/data_statistics_upos.tsv")
