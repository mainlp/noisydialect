"""
Converts annotated corpus files with the format:
TOKEN	TAG
(tab-separated).
Sentence boundaries are indicated by empty lines.
"""

from argparse import ArgumentParser
from glob import glob
import re


def ud(input_files, out_file, tagfix, upos=True, translit=True, verbose=True):
    if upos:
        pos_idx = 3
    else:  # XPOS
        pos_idx = 4
    with open(out_file, 'w+', encoding="utf8") as f_out:
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
                        pos = tagfix.get(pos, pos)
                        if pos == "_":
                            continue
                        if not form:
                            print("Transliteration missing!")
                            print(line)
                            print(in_file)
                            print("(exiting)")
                            return
                        f_out.write(f"{form}\t{pos}\n")
                    except IndexError:
                        print("!!! malformed line:")
                        print(line)
                        print(in_file)
                        print("(exiting)")
                        return


def ud_phono(input_files, out_file, tagfix, upos=True, ortho=False,
             verbose=True):
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
                    if not next_has_phono:
                        continue
                    cells = line.split("\t")
                    try:
                        if ortho:
                            form = cells[1]
                        else:
                            form = None
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


def kenpos(directory, out_file, keep_original_tags=True,
           filewise_updates=False, include_file_details=True):
    n_sents_total, n_sents_skipped_total = 0, 0
    n_toks_total, n_toks_skipped_total = 0, 0
    if keep_original_tags:
        tag_map = {}
    else:
        tag_map = {
            "A": "ADP",
            "ADO": "ADP",
            "ADJE": "ADJ",
            "ADVB": "ADV",
            "C": "CCONJ",
            "C0NJ": "CCONJ",
            "COJ": "CCONJ",
            "CONJ": "CCONJ",
            "DP": "ADP",
            "INTER": "INTJ",
            "N": "NOUN",
            "NN": "NOUN",
            "NNN": "NOUN",
            "PART": "PRT",
            "PI": "PUNCT",
            "PR": "PRON",
            "PR0N": "PRON",
            "PRO": "PRON",
            "PROUN": "PRON",
            "PUN": "PUNCT",
            "PUNC": "PUNCT",
            "PUNT": "PUNCT",
            "V": "VERB",
            "VRB": "VERB",
        }
    with open(out_file, 'w+', encoding="utf8") as f_out:
        for path in glob(directory + "/*csv"):
            n_sents, n_sents_skipped = 0, 0
            n_toks, n_toks_skipped = 0, 0
            with open(path, encoding="utf8") as f:
                filename = path.split("/")[-1].split("\\")[-1]
                first_line = True
                sent = []
                skip_sent = False
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if first_line:
                        first_line = False
                        continue
                    if line == "WORD\tPOS":
                        continue
                    try:
                        cells = line.split("\t")
                        form = cells[0]
                        pos = cells[1].strip().upper()
                        pos = tag_map.get(pos, pos)
                        sent.append((form, pos))
                        if form in (".", "?", "!"):  # TODO quotation marks
                            if skip_sent:
                                n_sents_skipped += 1
                                n_toks_skipped += len(sent)
                            else:
                                n_sents += 1
                                n_toks += len(sent)
                                for (form, pos) in sent:
                                    f_out.write(f"{form}\t{pos}")
                                    if include_file_details:
                                        f_out.write("\t" + filename)
                                    f_out.write("\n")
                                f_out.write("\n")
                            sent = []
                    except IndexError:
                        # print("Missing POS tag:")
                        # print(line)
                        # print("(Skipping sentence.)")
                        n_toks_skipped += 1
                        skip_sent = True
            if filewise_updates:
                print(path)
                print(f"Added {n_sents} sentences ({n_toks} tokens)")
                print(f"Skipped {n_sents_skipped} sentences ({n_toks_skipped} tokens)")
            n_sents_total += n_sents
            n_sents_skipped_total += n_sents_skipped
            n_toks_total += n_toks
            n_toks_skipped_total += n_toks_skipped
    print("TOTAL")
    print(f"Added {n_sents_total} sentences ({n_toks_total} tokens)")
    print(f"Skipped {n_sents_skipped_total} sentences ({n_toks_skipped_total} tokens)")


def murre(outfile, infiles,print_src_file=False):
    skip_lines = ("<clause", "</clause",
                  "<paragraph", "</paragraph",
                  "<text", "</text",
                  "<!--")
    sentences_skipped = 0
    sentences_added = 0
    with open(outfile, "w+", encoding="utf8") as f_out:
        for filename in infiles:
            with open(filename, encoding="utf8") as f_in:
                sent = None
                skip_sent = False
                for line in f_in:
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("<sentence"):
                        sent = []
                        skip_sent = False
                        continue
                    if line.startswith("</sentence>"):
                        if skip_sent:
                            sentences_skipped += 1
                        else:
                            sentences_added += 1
                            for word, tag in sent:
                                f_out.write(f"{word}\t{tag}")
                                if print_src_file:
                                    f_out.write("\t" + filename)
                                f_out.write("\n")
                            f_out.write("\n")
                        sent = None
                        continue
                    if skip_sent:
                        continue
                    skip_line = False
                    for skip_start in skip_lines:
                        if line.startswith(skip_start):
                            skip_line = True
                            break
                    if skip_line:
                        continue
                    if line.startswith("<"):
                        print(line)
                    cells = line.split("\t")
                    form = cells[1]
                    pos = cells[3]
                    if pos == "__UNDEF__":
                        skip_sent = True
                    sent.append((form, pos))
                    # TODO skip slashes?
    print(f"Added {sentences_added} sentences.")
    print(f"Skipped {sentences_skipped} sentences.")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--type",
                        choices=["ud", "noah", "narabizi", "ara", "kenpos",
                                 "murre"])
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
    parser.add_argument("--phono", action="store_true", default=False,
                        help="Use only sentences with phonetic annotations")
    parser.add_argument("--ortho", action="store_true", default=False,
                        help="Use orthographic versions of "
                             "words with phonetic annotations")
    parser.add_argument("--tigerize", action="store_true", default=False)
    parser.add_argument("--translit", action="store_true", default=False)
    parser.add_argument("--tagset", default="",
                        help="tagset file (only relevant for NArabizi)")
    parser.add_argument("--kenpostags", action="store_true", default=False)
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
        if args.phono:
            ud_phono(input_files, args.out, tagfix, args.upos, args.ortho)
        else:
            ud(input_files, args.out, tagfix, args.upos, args.translit)
    elif args.type == "noah":
        if args.excl:
            noah_excl(args.dir + "/" + args.files, args.out, args.excl)
        else:
            noah(args.dir + "/" + args.files, args.out)
    elif args.type == "narabizi":
        preprocess_narabizi(args.dir + "/" + args.files, args.out, args.tagset)
    elif args.type == "ara":
        ara(args.dir + "/" + args.files, args.out)
    elif args.type == "kenpos":
        kenpos(args.dir, args.out, args.kenpostags)
    elif args.type == "murre":
        murre(args.out, glob(args.glob))
