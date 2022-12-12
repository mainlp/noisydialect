from glob import glob
import re


def murre(infiles, out_folder,
          group_by="region",  # group [25]; eight; region [6]; two; no_groups
          dev={"east", "SAV", "6sav", "VarP"},
          print_details=False,
          ):
    if group_by not in ("group", "eight", "region", "two", "no_groups"):
        print("Did not recognize group_by value. Aborting.")
        return
    skip_lines = ("<clause", "</clause",
                  "<paragraph", "</paragraph",
                  "</text",
                  "<!--")
    auxiliaries = ("olla", "ei", "voida", "pitää", "saattaa", "täytyä",
                   "joutua", "aikoa", "taitaa", "tarvita", "mahtaa")
    copula = "olla"
    tag_map = {"a": "ADJ",
               "a:pron": "ADJ",
               "a:pron:dem": "ADJ",
               "a:pron:int": "ADJ",
               "a:pron:rel": "ADJ",
               "num:ord": "ADJ",
               "num:ord_pron": "ADJ",
               "p:post": "ADP",
               "p:pre": "ADP",
               "neg": "AUX",
               "adv": "ADV",
               "adv:pron": "ADV",
               "adv:pron:dem": "ADV",
               "adv:pron:int": "ADV",
               "adv:pron:rel": "ADV",
               "adv:q": "ADV",
               "p:adv": "ADV",
               "cnj:coord": "CCONJ",
               "intj": "INTJ",
               "n": "NOUN",
               "num:card": "NUM",
               "num:murto": "NUM",
               "pron": "PRON",
               "pron:dem": "PRON",
               "pron:int": "PRON",
               "pron:pers": "PRON",
               "pron:pers12": "PRON",
               "pron:ref": "PRON",
               "pron:rel": "PRON",
               "n:prop": "PROPN",
               "n:prop:pname": "PROPN",
               "punct": "PUNCT",
               "cnj:rel": "SCONJ",
               "cnj:sub": "SCONJ",
               "muu": "X",
               "v": "VERB"}
    q2adj = ('ensi', 'eri', 'koko', 'kumma', 'monias', 'oma', 'samainen',
             'omas')
    q2pron = ('ainoa', 'ainut', 'eräs', 'joka', 'jokainen', 'jokin', 'joku',
              'okunen', 'jompikumpi', 'kaikki', 'kuka', 'kukaan', 'kumpi',
              'kumpikin', 'mikin', 'mikä', 'mikään', 'molemmat', 'moni', 'muu',
              'muuan', 'muutama', 'sama',
              'itsekukin', 'kumpainen', 'kumpainenkaan', 'kumpainenkin',
              'jokaan', 'jokunen', 'jompi', 'ken', 'kukin', 'usea', 'harva',
              'joka ainoa', 'joka ikinen',
              '???', 'oooo',
              )
    group2eight = {
        "VarE": "1lou",
        "VarP": "1lou",
        "SatE": "2louvae",
        "SatL": "2louvae",
        "VarU": "2louvae",
        "VarY": "2louvae",
        "HämE": "3haeme",
        "HämK": "3haeme",
        "HämP": "3haeme",
        "Kym": "3haeme",
        "SatP": "3haeme",
        "PohE": "4epoh",
        "PohK": "5kppoh",
        "PohP": "5kppoh",
        "PerP": "5perpoh",
        "LänP": "5perpoh",
        "KesE": "6sav",
        "KesL": "6sav",
        "KesP": "6sav",
        "KarP": "6sav",
        "Kai": "6sav",
        "SavE": "6sav",
        "SavP": "6sav",
        "KarE": "7kaa",
        "KarK": "7kaa",
    }
    region2two = {
        "LOU": "west",
        "LVÄ": "west",
        "HÄM": "west",
        "POH": "west",
        "SAV": "east",
        "KAA": "east",
    }
    sentences_skipped = 0
    utt_map = {}
    for filename in infiles:
        with open(filename, encoding="utf8") as f_in:
            key = None
            sent = None
            skip_sent = False
            sent_may_contain_aux = False
            highest_pred = None
            for line in f_in:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("<text info"):
                    if group_by == "no_groups":
                        key = "all"
                    elif group_by == "region" or group_by == "two":
                        dialect_region = re.search(
                            r'(?<=dialect_region=\")\w+', line).group()
                        if group_by == "region":
                            key = dialect_region
                        else:
                            key = region2two[dialect_region]
                    else:
                        dialect_group = re.search(r'(?<=dialect_group=\")\w+',
                                                  line).group()
                        if group_by == "group":
                            key = dialect_group
                        else:
                            key = group2eight[dialect_group]
                    continue
                if line.startswith("<sentence"):
                    sent = []
                    skip_sent = False
                    sent_may_contain_aux = False
                    highest_pred = None
                    continue
                if line.startswith("</sentence>"):
                    if skip_sent:
                        sentences_skipped += 1
                    else:
                        utt = ""
                        for word, lemma, pos, syn_func, details in sent:
                            # Update the tags of auxiliary verbs,
                            # as detected via their lemma and by the
                            # fact that they're not the "last" predicate
                            # in the sentence.
                            if sent_may_contain_aux and pos == "VERB"\
                                    and lemma in auxiliaries\
                                    and syn_func != highest_pred:
                                pos = "AUX"
                            utt += word + "\t" + pos
                            if print_details:
                                utt += "\t" + lemma + "\t" + details
                            utt += "\n"
                        try:
                            utt_map[key].append(utt)
                        except KeyError:
                            utt_map[key] = [utt]
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
                pos = cells[3]
                if pos == "__UNDEF__":
                    skip_sent = True
                    continue
                pos = tag_map.get(pos, pos)
                form = cells[1]
                lemma = cells[2]
                syn_func = cells[5]
                if pos == "VERB" and lemma in auxiliaries:
                    if lemma == copula:
                        # Tag copula as aux.
                        # This works since olla is otherwise also an
                        # auxiliary anyway!
                        pos = "AUX"
                    else:
                        sent_may_contain_aux = True
                elif pos == "q":
                    if lemma in q2adj:
                        pos = "ADJ"
                    elif lemma in q2pron:
                        pos = "PRON"
                if syn_func == "pred3" or \
                        (highest_pred != "pred3" and syn_func == "pred2"):
                    highest_pred = syn_func
                details = ""
                if print_details:
                    details = cells[4] + "\t" + cells[5] + "\t" + cells[8]
                sent.append((form, lemma, pos, syn_func, details))
    print(f"Skipped {sentences_skipped} sentences.")
    for key in utt_map:
        filename = out_folder + "/"
        filename += "dev" if key in dev else "test"
        filename += "_murre-" + key + "_UPOS.tsv"
        print(f"Writing {len(utt_map[key])} utterances to {filename}")
        with open(filename, "w+", encoding="utf8") as f:
            for utt in utt_map[key]:
                f.write(utt + "\n")


if __name__ == "__main__":
    murre(glob("../datasets/LA-murre-vrt/lam_*.vrt"),
          "../datasets/")
