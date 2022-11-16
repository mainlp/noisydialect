"""
Converts annotated corpus files into the format used here:
https://github.com/noe-eva/NOAH-Corpus/blob/master/test_GSW_STTS.txt
Each line contains a token and its POS tag, separated by a single
blank space (this assumes no POS tags include blank spaces).
Sentence boundaries are indicated by empty lines.
"""

from argparse import ArgumentParser
from glob import glob


def ud(input_files, out_file, tagfix, upos=True, phono=False, verbose=True):
    if upos:
        pos_idx = 3
    else:  # XPOS
        pos_idx = 4
    next_has_phono = False
    n_sents = 0
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
                            next_has_phono = False
                            f_out.write("\n")
                        continue
                    if line[0] == "#":
                        if line.startswith("# text_orig"):
                            next_has_phono = True
                        elif line.startswith("# sent_id"):
                            n_sents += 1
                        continue
                    first_sent = False
                    if phono and not next_has_phono:
                        continue
                    cells = line.split("\t")
                    try:
                        form = None
                        if phono:
                            misc_entries = cells[-1].split("|")
                            for entry in misc_entries:
                                if entry.startswith("Phono="):
                                    form = entry[6:]
                                    break
                            if not form:
                                print("!!! Phonetic information missing:")
                                print(line)
                                print(in_file)
                                print("(exiting)")
                                return
                        else:
                            form = cells[1]
                        pos = cells[pos_idx]
                        pos = tagfix.get(pos, pos)
                        if pos == "_":
                            continue
                        f_out.write(f"{form}\t{pos}\n")
                    except IndexError:
                        print("!!! malformed line:")
                        print(line)
                        print(in_file)
                        print("(exiting)")
                        return
    print(f"Wrote {n_sents} sentences to {out_file}.")


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
    parser.add_argument("--glob", default="",
                        help="glob pattern (ignores --dir and --files)")
    parser.add_argument("--out", help="output file")
    parser.add_argument("--xpos", dest="upos", action="store_false",
                        default=True)
    parser.add_argument("--phono", action="store_true", default=False)
    parser.add_argument("--tigerize", action="store_true", default=False)
    args = parser.parse_args()
    tagfix = {}
    if args.tigerize:
        # TIGER-style STTS tag
        tagfix["PAV"] = "PROAV"  # rename
        tagfix["PIDAT"] = "PIAT"  # merge
    if args.type == "ud":
        if args.glob:
            input_files = glob(args.glob)
        else:
            input_files = [args.dir + "/" + f.strip()
                           for f in args.files.split(",")]
        ud(input_files, args.out, tagfix, args.upos, args.phono)
    elif args.type == "noah":
        noah(args.dir + "/" + args.files, args.out)
