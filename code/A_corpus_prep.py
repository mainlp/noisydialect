"""
Converts annotated corpus files with the format:
TOKEN	TAG
(tab-separated).
Sentence boundaries are indicated by empty lines.
"""

from argparse import ArgumentParser
from glob import glob


def ud(input_files, out_file, tagfix, upos=True, translit=False,
       gloss_comp=False, incl_source_file=False, verbose=True):
    if upos:
        pos_idx = 3
    else:  # XPOS
        pos_idx = 4
    sents_added = 0
    sents_skipped = 0
    with open(out_file, 'w+', encoding="utf8") as f_out:
        for in_file in input_files:
            if verbose:
                print("Reading " + in_file)
            with open(in_file, encoding="utf8") as f_in:
                if incl_source_file:
                    filename = in_file.split("/")[-1].split("\\")[-1]
                else:
                    filename = None
                first_sent = True
                sent = []
                gloss_diff = not gloss_comp
                skip_sent = False
                for line in f_in:
                    line = line.strip()
                    if not line:
                        if sent:
                            if gloss_diff and not skip_sent:
                                sents_added += 1
                                for form, pos, filename in sent:
                                    if filename:
                                        f_out.write(
                                            f"{form}\t{pos}\t{filename}\n")
                                    else:
                                        f_out.write(f"{form}\t{pos}\n")
                            else:
                                sents_skipped += 1
                                if not gloss_diff:
                                    print(
                                        "Identical to gloss (skipping sent.)")
                                else:
                                    print(
                                        "Missing POS tag (skipping sent.)")
                                print(" ".join((x[0] for x in sent)))
                            gloss_diff = not gloss_comp
                            sent = []
                        if not first_sent:
                            f_out.write("\n")
                        continue
                    if line[0] == "#":
                        continue
                    first_sent = False
                    cells = line.split("\t")
                    try:
                        if translit:
                            form = ""
                            miscs = cells[-1].split("|")
                            for misc in miscs:
                                if misc.startswith("Translit="):
                                    form = misc.split("=", 1)[1]
                                    break
                        else:
                            form = cells[1]
                        form = form.strip()
                        pos = cells[pos_idx]
                        if "+" in pos:
                            print("Unknown POS tag: " + form + " " + pos)
                            print("Skipping sentence!")
                            skip_sent = True
                        else:
                            pos = tagfix.get(pos, pos)
                            if not pos:
                                print("POS tag missing for " + form)
                                print("Skipping sentence!")
                                skip_sent = True
                            if pos == "_":
                                continue
                            if not form:
                                if translit:
                                    print("Transliteration missing!")
                                    print(line)
                                    print(in_file)
                                    print("(exiting)")
                                    return
                                else:
                                    print("Form missing:")
                                    print(line)
                                    continue
                            if gloss_comp:
                                miscs = cells[-1].split("|")
                                gloss = ""
                                for misc in miscs:
                                    if misc.startswith("Gloss="):
                                        gloss = misc.split("=", 1)[1]
                                        break
                                if gloss != form:
                                    gloss_diff = True
                            sent.append((form, pos, filename))
                    except IndexError:
                        print("!!! malformed line:")
                        print(line)
                        print(in_file)
                        print("(exiting)")
                        return
    print(f"Added {sents_added} sentences.")
    print(f"Skipped {sents_skipped} sentences.")


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


def noah_excl(in_file, out_file, excl_file):
    excl = []
    with open(excl_file) as f:
        sent = ""
        for line in f:
            if not line.strip():
                if sent:
                    excl.append(sent)
                    sent = ""
                continue
            if sent:
                sent += " "
            sent += line.split("\t")[0]
    n_skipped = 0
    with open(out_file, 'w', encoding="utf8") as f_out:
        with open(in_file, encoding="utf8") as f_in:
            sent = []
            sent_pos = []
            for line in f_in:
                line = line.strip()
                if not line:
                    joined_sent = " ".join(sent)
                    if joined_sent in excl:
                        print("Skipping sentence: " + joined_sent)
                        n_skipped += 1
                    else:
                        for word, word_pos in zip(sent, sent_pos):
                            f_out.write(f"{word}\t{word_pos}\n")
                        f_out.write("\n")
                    sent = []
                    sent_pos = []
                    continue
                line = line.replace("\xa0", " ").replace("\t", " ")
                word, _, word_pos = line.rpartition(" ")
                sent.append(word)
                sent_pos.append(word_pos)
            if sent:
                for word, word_pos in zip(sent, sent_pos):
                    f_out.write(f"{word}\t{word_pos}\n")
                f_out.write("\n")
    print(f"Skipped {n_skipped} sentences.")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--type",
                        choices=["ud", "noah"])
    parser.add_argument("--dir", default="",
                        help="data directory root")
    parser.add_argument("--files", default="",
                        help="input file(s) within the data directory, "
                        "comma-separated")
    parser.add_argument("--glob", default="",
                        help="glob pattern (ignores --dir and --files)")
    parser.add_argument("--out", help="output file")
    parser.add_argument("--excl", default="",
                        help="exclude sentences from the following file "
                             "(only relevant for NOAH, "
                             "when UD Swiss German UZH is also used)")
    parser.add_argument("--xpos", dest="upos", action="store_false",
                        default=True)
    parser.add_argument("--ortho", action="store_true", default=False,
                        help="Use orthographic versions of "
                             "words with phonetic annotations")
    parser.add_argument("--tigerize", action="store_true", default=False)
    parser.add_argument("--translit", action="store_true", default=False)
    parser.add_argument("--glosscomp", action="store_true", default=False)
    parser.add_argument("--incl_source_file", action="store_true",
                        default=False)
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
        ud(input_files, args.out, tagfix, args.upos, args.translit,
           args.glosscomp, args.incl_source_file)
    elif args.type == "noah":
        if args.excl:
            noah_excl(args.dir + "/" + args.files, args.out, args.excl)
        else:
            noah(args.dir + "/" + args.files, args.out)
