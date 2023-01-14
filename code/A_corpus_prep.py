"""
Converts annotated corpus files with the format:
TOKEN	TAG
(tab-separated).
Sentence boundaries are indicated by empty lines.
"""

from argparse import ArgumentParser
from glob import glob
import re


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
                        pos = cells[pos_idx]
                        if "+" in pos:
                            print("Splitting " + pos + " and " + cells[2]
                                  + " (Incompatible w/ translit or glosscomp)")
                            for i, (subpos, lemma) in enumerate(
                                    zip(pos.split("+"), cells[2].split("_"))):
                                subpos = tagfix.get(subpos, subpos)
                                if i == 0 and form[0] == form[0].upper():
                                    lemma = lemma[0].upper() + lemma[1:]
                                sent.append((lemma, subpos, filename))
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


def preprocess_narabizi(in_file, out_file, tagset_file):
    # Fix mixed UTF-8/MacRoman encoding and whitespace issues
    tags = set()
    with open(tagset_file, encoding="utf8") as f:
        for line in f:
            tag = line.strip()
            if tag:
                tags.add(tag)
    with open(out_file, 'w+', encoding="utf8") as f_out:
        with open(in_file, encoding="mac-roman") as f_in:
            for line in f_in:
                if not line.startswith("#") and len(line) > 1:
                    line = re.sub("\\s\\s+", "\t", line)
                    for tag in tags:
                        line = line.replace(" " + tag, "\t" + tag)
                    cells = line.split("\t")
                    if not cells[3] in tags:
                        if len(cells) < 10:
                            lemma_pos = "\t".join(cells[1].rsplit(" ", 1))
                            line = cells[0] + "\t" + lemma_pos + "\t" +\
                                "\t".join(cells[2:])
                            print("Split '" + cells[1] + "'")
                        elif len(cells) > 10:
                            line = "\t".join(cells[0:2]) + "\t" +\
                                cells[2] + " " + cells[3] + "\t" +\
                                "\t".join(cells[4:])
                            print("Merged '" + cells[2] + "' and '"
                                  + cells[3] + "'")
                        cells = line.split("\t")
                    if len(cells) != 10:
                        print(line)
                try:
                    line = line.encode("mac-roman").decode()
                except UnicodeDecodeError:
                    pass
                f_out.write(line)


def ara(in_file, out_file, include_tag_details=True, print_mapping=False,
        print_segment_details=True):
    replace3 = {
        "DET+ADJ+CASE": "ADJ+__+__",
        "DET+ADJ+NSUFF": "ADJ+__+__",
        "DET+NOUN+CASE": "NOUN+__+__",
        "DET+NOUN+NSUFF": "NOUN+__+__",
    }
    replace2 = {
        "+CASE": "+__",
        "+NSUFF": "+__",
        "DET+ADJ": "ADJ+__",
        "DET+NOUN": "NOUN+__",
        "PROG_PART+V": "V+__",
    }
    replace1 = {
        "CONJ": "CCONJ",
        "EMOT": "SYM",
        "FOREIGN": "X",
        "FUT_PART": "AUX",
        "HASH": "X",
        "MENTION": "PROPN",
        "NEG_PART": "PART",
        "PREP": "ADP",
        "PUNC": "PUNCT",
        "URL": "SYM",
        "V": "VERB",
    }
    replace_last = {
        "ADVERB": "ADV",  # introduced by V->VERB change
    }
    replacement_dicts = (replace3, replace2, replace1, replace_last)
    replace_fulltok = {
        # On their own:
        # https://universaldependencies.org/u/dep/all.html#al-u-dep/goeswith
        "CASE": "X",
        "NSUFF": "X",
        "PROG_PART": "X",
        # Annotation mistake?
        "PROG_PART+NOUN": "X+NOUN",
        # Segmentation mistake?
        "PART+PROG_PART": "PART+X",
    }

    part2sconj_forms = ("إن", "ان،", "أن،")
    tag_map = {}
    n_not_split = 0
    n_segments_ok = 0
    segment_issues = []
    with open(out_file, 'w+', encoding="utf8") as f_out:
        with open(in_file, encoding="utf8") as f_in:
            first_line = True
            sent = ""
            skip_sent = False
            for line in f_in:
                line = line.strip()
                if not line:
                    continue
                if first_line:
                    first_line = False
                    continue
                cells = line.split("\t")
                form = cells[4]
                pos = cells[6]
                if pos == "EOS":
                    if not skip_sent:
                        f_out.write(sent + "\n")
                    sent = ""
                    skip_sent = False
                    continue
                if skip_sent:
                    continue
                if not form:
                    print("Line with empty token (skipping sentence):")
                    print(line)
                    skip_sent = True
                    continue
                for repl_dict in replacement_dicts:
                    for repl in repl_dict:
                        if repl in pos:
                            pos = pos.replace(repl, repl_dict[repl])
                for re_full in replace_fulltok:
                    if re_full == pos:
                        pos = replace_fulltok[re_full]
                tag_map[cells[6]] = pos
                use_segments = False
                tags = pos.split("+")
                if len(tags) > 1:
                    for tag in tags[1:]:
                        if tag != "__":
                            use_segments = True
                            break
                    if not use_segments:
                        sent += f"{form}\t{tags[0]}"
                        if include_tag_details:
                            sent += f"\t{cells[6]}\t{0}:{len(tags)}\n"
                        else:
                            sent += "\n"
                        n_not_split += 1
                        continue
                if use_segments:
                    segments = cells[5].split("+")
                    n = len(tags)
                    assert n == len(segments)
                    i = 0
                    while i < n:
                        j = i + 1
                        while j < n:
                            if tags[j] == "__":
                                j += 1
                            else:
                                break
                        joined_form = "".join(segments[i:j])
                        all_joined = "".join(segments)
                        if all_joined == form:
                            n_segments_ok += 1
                        else:
                            segment_issues.append(
                                (form, all_joined, cells[5]))
                        joined_tag = tags[i]
                        sent += f"{joined_form}\t{joined_tag}"
                        if include_tag_details:
                            sent += f"\t{cells[6]}\t{i}:{j}\n"
                        else:
                            sent += "\n"
                        i = j
                    continue
                if pos == "PART" and form in part2sconj_forms:
                    pos = "SCONJ"
                sent += f"{form}\t{pos}\n"
                n_not_split += 1
    if print_mapping:
        print("Mapped the original tags as follows:")
        for orig_tag in tag_map:
            print(f"{orig_tag}\t{tag_map[orig_tag]}")
    if print_segment_details:
        print("Full token used:", n_not_split)
        print("Segmentation used -- segmentation OK:", n_segments_ok)
        print("Segmentation used -- issues with segmentation:",
              len(segment_issues))
        for issue in segment_issues:
            print(f"tok {issue[0]} merged {issue[1]} segments {issue[2]}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--type",
                        choices=["ud", "noah", "narabizi", "ara", "kenpos"])
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
                             "(only relevant for NOAH)")
    parser.add_argument("--xpos", dest="upos", action="store_false",
                        default=True)
    parser.add_argument("--ortho", action="store_true", default=False,
                        help="Use orthographic versions of "
                             "words with phonetic annotations")
    parser.add_argument("--tigerize", action="store_true", default=False)
    parser.add_argument("--translit", action="store_true", default=False)
    parser.add_argument("--tagset", default="",
                        help="tagset file (only relevant for NArabizi)")
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
    elif args.type == "narabizi":
        preprocess_narabizi(args.dir + "/" + args.files, args.out, args.tagset)
    elif args.type == "ara":
        ara(args.dir + "/" + args.files, args.out)
