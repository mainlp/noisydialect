def get_entries(input_files, upos=True, ortho=False, verbose=True):
    if upos:
        pos_idx = 3
    else:  # XPOS
        pos_idx = 4
    next_has_phono = False
    n_sents = 0
    dialect2group = {
        "aal": "east",
        "bardu": "east",  # Indre Troms
        "eidsberg": "east",
        "gol": "east",
        "austevoll": "west",
        "farsund": "west",
        "giske": "west",
        "lierne": "troender",
        "flakstad": "north",
        "vardoe": "north",
    }
    group2utt = {
        "east": [],
        "west": [],
        "troender": [],
        "north": [],
    }
    for in_file in input_files:
        if verbose:
            print("Reading " + in_file)
        with open(in_file, encoding="utf8") as f_in:
            first_sent = True
            cur_sent = []
            cur_group = None
            for line in f_in:
                line = line.strip()
                if not line:
                    if not first_sent:
                        group2utt[cur_group].append(cur_sent)
                        next_has_phono = False
                        cur_group = None
                        cur_sent = None
                    continue
                if line[0] == "#":
                    if line.startswith("# text_orig"):
                        next_has_phono = True
                    elif line.startswith("# sent_id"):
                        n_sents += 1
                    elif line.startswith("# dialect: "):
                        dialect = line[11:].split(" ")[0]
                        cur_group = dialect2group[dialect]
                        cur_sent = []
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
                    cur_sent.append((form, pos))
                except IndexError:
                    print("!!! malformed line:")
                    print(line)
                    print(in_file)
                    print("(exiting)")
                    return
            if cur_sent:
                group2utt[cur_group].append(cur_sent)
    return group2utt


if __name__ == "__main__":
    input_folder = "../datasets/UD_Norwegian-NynorskLIA_dialect/"
    input_files = ("no_nynorsklia_dialect-ud-train.conllu",
                   "no_nynorsklia_dialect-ud-dev.conllu",
                   "no_nynorsklia_dialect-ud-test.conllu",)
    group2utt = get_entries((input_folder + f for f in input_files))

    group2status = {
        "west": "dev",
        "east": "test",
        "north": "test"}
    for group in group2status:
        with open("../datasets/" + group2status[group]
                  + "_LIA-" + group + "_UPOS.tsv",
                  "w+", encoding="utf8") as f_out:
            for utterance in group2utt[group]:
                for word, pos in utterance:
                    f_out.write(f"{word}\t{pos}\n")
                f_out.write("\n")
