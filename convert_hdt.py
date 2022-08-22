import cust_logger

from argparse import ArgumentParser
import os
import re
import sys


def convert(data_dir, parts, out_file, tagfix, include_filename, verbose):
    with open(out_file, 'w', encoding="utf8") as f_out:
        for part in parts:
            for (dirpath, dirnames, filenames) in os.walk(
                    os.sep.join([data_dir, part])):
                if verbose:
                    print(dirpath)
                for filename in filenames:
                    if filename.endswith('.cda'):
                        if verbose and filename.endswith('000.cda'):
                            print(filename)
                        with open(os.sep.join([dirpath, filename])) as f_in:
                            cur_tok = None
                            for line in f_in:
                                line = line.strip()[:-1]
                                if len(line) == 0:
                                    continue
                                try:
                                    # Lines introducing tokens start with an
                                    # integer!
                                    int(line[0])
                                    cur_tok = re.split(r"(?<!\\)'", line)[1]
                                except ValueError:
                                    if line.startswith("'cat'"):
                                        try:
                                            pos = re.split(
                                                r"(?<!\\)'",
                                                line.split(" / ")[1])[1]
                                            pos = tagfix.get(pos, pos)
                                            f_out.write(cur_tok + " " + pos)
                                            if include_filename:
                                                f_out.write(" " + filename)
                                            f_out.write("\n")
                                        except IndexError:
                                            print("!!! malformed line"
                                                  "(trying to read POS tag):")
                                            print(line)
                                            print(filename)
                                            print("(exiting)")
                                            sys.exit(1)
                                except IndexError:
                                    print("!!! malformed line"
                                          "(trying to read token):")
                                    print(line)
                                    print(filename)
                                    print("(exiting)")
                                    sys.exit(1)
                            f_out.write("\n")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-d", "--dir", dest="dir", help="data directory root",
                        default="data/hamburg-dependency-treebank/")
    parser.add_argument("-p", "--parts", dest="parts",
                        help="data directory subfolders, comma-separated",
                        default="part_A,part_B")
    parser.add_argument("-o", "--out", dest="out", help="output file",
                        default="data/hamburg-dependency-treebank/"
                                "train_DHT_STTS.txt")
    parser.add_argument("-f", "--filenames", dest="include_filename",
                        action="store_true", default=False,
                        help="add each token's file name in an extra column")
    parser.add_argument("-s", "--pure_stts", dest="tigerize",
                        action="store_false", default=True,
                        help="use pure (non-TIGER) STTS tags")
    parser.add_argument("-t", "--typos", dest="fix_typos",
                        action="store_false", default=True,
                        help="don't fix POS typos")
    parser.add_argument("-q", "--quiet", action="store_false", dest="verbose",
                        default=True, help="no messages to stdout")

    args = parser.parse_args()
    if args.verbose:
        sys.stdout = cust_logger.Logger("hdt_conversion")
        for arg in vars(args):
            print(arg, getattr(args, arg))
    tagfix = dict()
    if args.tigerize:
        # The NOAH test set uses TIGER-ized STTS tags
        tagfix["PAV"] = "PROAV"  # rename
        tagfix["PIDAT"] = "PIAT"  # merge
    if args.fix_typos:
        tagfix["VAIZU"] = "VVIZU"
        tagfix["PPOSSAT"] = "PPOSAT"
    convert(args.dir, [part.strip() for part in args.parts.split(',')],
            args.out, tagfix, args.include_filename, args.verbose)
