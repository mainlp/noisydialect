"""
Converts annotated corpus files into the format used here:
https://github.com/noe-eva/NOAH-Corpus/blob/master/test_GSW_STTS.txt
Each line contains a token and its POS tag, separated by a single
blank space (this assumes no POS tags include blank spaces).
Sentence boundaries are indicated by empty lines.
"""

from argparse import ArgumentParser


def ud(input_files, out_file, upos=True, verbose=True):
    if upos:
        pos_idx = 3
    else:  # XPOS
        pos_idx = 4
    with open(out_file, 'w', encoding="utf8") as f_out:
        for in_file in input_files:
            if verbose:
                print("Reading " + in_file)
            with open(in_file, encoding="utf8") as f_in:
                first_sent = True
                for line in f_in:
                    line = line.strip()
                    if not line:
                        if not first_sent:
                            f_out.write("\n")
                        continue
                    if line[0] == "#":
                        # comment
                        continue
                    first_sent = False
                    cells = line.split("\t")
                    try:
                        form = cells[1]
                        upos = cells[pos_idx]
                        f_out.write(f"{form}\t{upos}\n")
                    except IndexError:
                        print("!!! malformed line:")
                        print(line)
                        print(in_file)
                        print("(exiting)")


def noah(in_file, out_file):
    with open(out_file, 'w', encoding="utf8") as f_out:
        with open(in_file, encoding="utf8") as f_in:
            for line in f_in:
                line = line.strip()
                if not line:
                    f_out.write("\n")
                    continue
                line = line.replace("\xa0", " ").replace("\t", " ")
                word, _, word_pos = line.rpartition(" ")
                f_out.write(f"{word}\t{word_pos}\n")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--type", choices=["ud", "noah"])
    parser.add_argument("--dir", default="",
                        help="data directory root")
    parser.add_argument("--files", default="",
                        help="input file(s) within the data directory, "
                        "comma-separated")
    parser.add_argument("--out", help="output file")
    parser.add_argument("--xpos", dest="upos", action="store_false",
                        default=True)
    args = parser.parse_args()
    if args.type == "ud":
        input_files = [args.dir + "/" + f.strip()
                       for f in args.files.split(",")]
        ud(input_files, args.out, args.upos)
    elif args.type == "noah":
        noah(args.dir + "/" + args.files, args.out)
