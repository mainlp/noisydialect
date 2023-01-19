from glob import glob
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr


def pretty_print_plm(plm):
    if plm == "mbert":
        return "mBERT"
    if plm == "xlmr":
        return "XLM-R"
    if plm == "beto":
        return "BETO"
    plm = plm.replace("bert", "BERT")
    plm = plm[0].upper() + plm[1:]
    return plm


def pretty_print_target(target):
    target = target.lower()
    if target == "rpic":
        return "Picard"
    if target == "roci":
        return "Occitan"
    if target == "rals":
        return "Alsatian G."
    if target == "noah":
        return "Swiss German"
    if target == "dar-egy":
        return "Egyptian A."
    if target == "dar-glf":
        return "Gulf Arabic"
    if target == "dar-mgr":
        return "Maghrebi A."
    if target == "dar-lev":
        return "Levantine A."
    if target == "narabizi" or target == "narbizi":  # Typo
        return "Algerian Arabizi"
    if target == "lia-west":
        return "West N."
    if target == "lia-east":
        return "East N."
    if target == "lia-north":
        return "North N."
    if target == "lia-west":
        return "West N."
    if target == "lsdc":
        return "Low Saxon"
    if target == "murre-lou":
        return "SW Finnish"
    if target == "murre-lvä":
        return "SW transition area"
    if target == "murre-häm":
        return "Tavastian F."
    if target == "murre-poh":
        return "Ostrobothnian F."
    if target == "murre-sav":
        return "Savonian F."
    if target == "murre-kaa":
        return "SE Finnish"
    return target


def pretty_print_train(train):
    train = train.lower()
    if train == "gsd":
        return "French"
    if train == "ancora-spa":
        return "Spanish"
    if train == "hdt":
        return "German"
    if train == "ndt-nno":
        return "Nynorsk"
    if train == "ndt-nob":
        return "Bokmål"
    if train == "tdt":
        return "Finnish"
    if train == "padt":
        return "MSA"
    if train == "padt-translit":
        return "MSA (transliterated)"
    if train == "mudt":
        return "Maltese"
    if train == "hdt":
        return "German"
    if train == "alpino":
        return "Dutch"
    return train


def train2monolingual(train):
    if train == "French":
        return "CamemBERT"
    if train == "ancora-spa" or train == "ancoraspa":
        return "BETO"
    if train == "German":
        return "GBERT"
    if train == "Nynorsk" or train == "Bokmål":
        return "NorBERT"
    if train == "Finnish":
        return "FinBERT"
    if train == "MSA":
        return "AraBERT"
    if train == "Maltese":
        return "BERTu"
    if train == "Dutch":
        return "BERTje"
    return "?"


def pretty_score_label(y_score):
    if y_score == "F1_MACRO":
        return "F1 macro"
    if y_score == "ACCURACY":
        return "Accuracy"
    return y_score


def pretty_tokenization_label(token_metric):
    if token_metric == "subtoken_ratio":
        return "Subtoken ratio (train)"
    if token_metric == "unk_ratio":
        return "Subtoken-level UNK ratio (train)"
    if token_metric == "ttr":
        return "Subtoken-level TTR (train)"
    if token_metric == "split_token_ratio":
        return "Split-word ratio (train)"
    if token_metric == "TARGET_SUBTOK_TYPES_IN_TRAIN":
        return "% of target subtoken types also in train"
    if token_metric == "TARGET_SUBTOKS_IN_TRAIN":
        return "% of target subtokens also in train"
    if token_metric == "TARGET_WORD_TYPES_IN_TRAIN":
        return "% of target word types also in train"
    if token_metric == "TARGET_WORD_TOKENS_IN_TRAIN":
        return "% of target word tokens also in train"
    return token_metric


def pretty_colorbar_label(hue):
    if hue.startswith("subtoken_ratio"):
        return "Absolute subtoken ratio difference"
    if hue.startswith("unk_ratio"):
        return "Absolute subtoken-level UNK ratio difference"
    if hue.startswith("ttr"):
        return "Absolute subtoken-level TTR difference"
    if hue.startswith("split_token_ratio"):
        return "Absolute split word ratio difference"
    if hue == "TARGET_SUBTOK_TYPES_IN_TRAIN":
        return "% of target subtoken types also in train"
    if hue == "TARGET_SUBTOKS_IN_TRAIN":
        return "% of target subtokens also in train"
    if hue == "TARGET_WORD_TYPES_IN_TRAIN":
        return "% of target word types also in train"
    if hue == "TARGET_WORD_TOKENS_IN_TRAIN":
        return "% of target word tokens also in train"
    return hue


def process_data_stats(filename):
    df = pd.read_csv(filename, sep='\t')
    df["train"] = df["TRAIN_SET"].apply(
        lambda x: x.split("_", 2)[1].replace("-full", ""))
    df["target_is_stdlang"] = df.apply(
        lambda x: x.TARGET_SET.startswith("dev_" + x.train)
        or x.TARGET_SET.startswith("test_" + x.train),
        axis=1)
    df["train"] = df["train"].apply(lambda x: pretty_print_train(x))
    df.drop(df[df.target_is_stdlang].index, inplace=True)
    df[["PLM", "noise"]] = df.TRAIN_SET.str.split(
        "_", 2).str[2].str.split("_", expand=True)
    df["PLM"] = df["PLM"].apply(lambda x: pretty_print_plm(x))
    df["PLM_mono"] = df.train.apply(lambda x: train2monolingual(x))
    df["used_multi"] = df.PLM.apply(lambda x: x == "mBERT" or x == "XLM-R")
    df = df[(df.PLM == df.PLM_mono) | (df.used_multi)]
    df["PLM_target"] = df.TARGET_SET.str.split("_", 3).str[2]
    df["PLM_target"] = df["PLM_target"].apply(lambda x: pretty_print_plm(x))
    df = df.drop(df[df.PLM != df.PLM_target].index)
    df["target"] = df["TARGET_SET"].apply(
        lambda x: pretty_print_target(x.split("_", 2)[1]))
    df["setup"] = df["PLM"] + " " + df["train"] + "→" + df["target"]
    df["noise"] = df["noise"].apply(lambda x: 0 if x == "orig" else int(x[4:]))
    for diff in ("SUBTOKEN_RATIO_DIFF", "UNK_RATIO_DIFF",
                 "TTR_DIFF", "SPLIT_TOKEN_RATIO_DIFF"):
        df[diff.lower() + "_abs"] = df[diff].apply(lambda x: abs(x))
    missing_data = pd.concat((df[df.ACCURACY == -1],
                              df[df.ACCURACY.isnull()]))[
        ["SETUP_NAME", "SEED", "TARGET_SET"]]
    df = df.drop(missing_data.index)
    return df, missing_data


def plot(df, y_score, token_metric, palette_name, png_name):
    if token_metric.startswith("TARGET_"):
        hue = token_metric
        y_tok = token_metric
    else:
        hue = token_metric + "_diff_abs"
        y_tok = token_metric.upper() + "_TRAIN"

    target2row = {tgt: i * 2 for i, tgt in enumerate(df["target"].unique())}
    plm2col = {col: i for i, col in enumerate(df["PLM"].unique())}
    n_targets = len(target2row)
    n_plms = len(plm2col)
    width_ratios = [4 for _ in range(n_plms)] + [1]
    fig, axes = plt.subplots(
        2 * n_targets, n_plms + 1,
        figsize=((n_plms + 1) * 3, n_targets * 5),
        sharey="row", sharex="col",
        gridspec_kw={"width_ratios": width_ratios}
    )
    y_pos_corr = 1.15 * (df[y_tok].max() - df[y_tok].min()) + df[y_tok].min()

    vmin = df[hue].min()
    vmax = df[hue].max()
    palette = {x: plt.cm.get_cmap(palette_name)
               ((x - vmin) / (vmax - vmin))
               for x in df[hue]}
    corr_stats = {}

    for setup, df_for_setup in df.groupby("setup"):
        target = df_for_setup["target"].unique()[0]
        plm = df_for_setup["PLM"].unique()[0]
        row = target2row[target]
        col = plm2col[plm]

        # Plot F1/accuracy
        score_plot = sns.scatterplot(
            data=df_for_setup, x="noise", y=y_score,
            hue=hue, palette=palette,
            ax=axes[row, col],
        )
        score_plot.set_title(setup)

        if not token_metric.startswith("TARGET_"):
            # Draw lines indicating where the optimal values
            # (train ratio == target ratio) would be.
            y_target = token_metric.upper() + "_TARGET"
            zero_diff_col = plt.cm.get_cmap(palette_name)(0)
            axes[row + 1, col].axhline(
                df.loc[df['setup'] == setup, y_target].iloc[0],
                c=zero_diff_col)

        # Plot the tokenization measure
        token_plot = sns.scatterplot(
            data=df_for_setup, x="noise", y=y_tok,
            hue=hue, palette=palette,
            ax=axes[row + 1, col],
        )
        token_plot.set_xticks([0, 15, 35, 55, 75, 95])

        # Add correlation info
        if len(df_for_setup[hue]) >= 2:
            label = ""
            try:
                r, r_p = pearsonr(df_for_setup[hue],
                                  df_for_setup[y_score])
                label = f"r = {r:.2f} (p = {r_p:.2f})\n"
            except ValueError:
                r, r_p = "??", "??"
                label = "r = ?? (p = ??)\n"
            try:
                rho, rho_p = spearmanr(df_for_setup[hue],
                                       df_for_setup[y_score])
                label += f"ρ = {rho:.2f} (p = {rho_p:.2f})"
            except ValueError:
                rho, rho_p = "??", "??"
                label += "ρ = ?? (p = ??)"
            axes[row + 1, col].text(0, y_pos_corr, label, fontsize=9)
            try:
                corr_stats[target][plm] = (rho, rho_p)
            except KeyError:
                corr_stats[target] = {plm: (rho, rho_p)}

        # Remove visual clutter
        for j in (row, row + 1):
            ax = axes[j, col]
            try:
                ax.get_legend().remove()
            except AttributeError:
                print("Already removed: " + setup + " "
                      + str(j) + ", " + str(col))
            ax.xaxis.grid()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
        axes[row + 1, col].set_xlabel(None)

    # Update the axis labels
    for i in range(len(target2row)):
        axes[i * 2, 0].set_ylabel(pretty_score_label(y_score))
        axes[i * 2 + 1, 0].set_ylabel(pretty_tokenization_label(token_metric))
        fig.text(0.5, 0, "Noise (%)", ha='center')
        axes[i * 2, -1].axis("off")
        axes[i * 2 + 1, -1].axis("off")

    # Add colour bar
    points = plt.scatter([], [], c=[],
                         vmin=vmin, vmax=vmax,
                         cmap=plt.cm.get_cmap(palette_name))
    cbar = plt.colorbar(points,
                        ax=[axes[i, -1] for i in range(len(target2row))],
                        aspect=35,
                        shrink=0.7,
                        pad=0,)
    cbar.ax.tick_params(right=False)
    cbar.ax.tick_params(axis='y', pad=0)
    cbar.ax.set_yticks([math.ceil(vmin * 100) / 100,
                        math.floor(vmax * 100) / 100])

    cbar.set_label(pretty_colorbar_label(hue), fontsize=11)

    plt.show()
    if png_name:
        fig.savefig(png_name)

    # print correlation stats
    train = df_for_setup["train"].unique()[0]
    mono = train2monolingual(train)
    stats = []
    for target in corr_stats:
        try:
            stats_mono = corr_stats[target][mono]
        except KeyError:
            stats_mono = ("", "")
        try:
            stats_mbert = corr_stats[target]["mBERT"]
        except KeyError:
            stats_mbert = ("", "")
        try:
            stats_xlmr = corr_stats[target]["XLM-R"]
        except KeyError:
            stats_xlmr = ("", "")
        stat = "\t".join((
            train, target,
            str(stats_mono[0]), str(stats_mono[1]),
            str(stats_mbert[0]), str(stats_mbert[1]),
            str(stats_xlmr[0]), str(stats_xlmr[1])))
        print(stat)
        stats.append(stat)
    return stats


def reverse_palette(palette_name):
    if palette_name.endswith("_r"):
        return palette_name[:-2]
    return palette_name + "_r"


def sort_score(item):
    if item.startswith("German"):
        if "Low Saxon" in item:
            return 0.5
        return 0
    if item.startswith("Dutch"):
        return 1
    if item.startswith("Bokmål"):
        return 2
    if item.startswith("Nynorsk"):
        return 3
    if item.startswith("French"):
        if "Occitan" in item:
            return 4.5
        return 4
    if item.startswith("Spanish"):
        return 5
    if item.startswith("MSA ("):
        return 7
    if item.startswith("MSA"):
        return 6
    if item.startswith("Maltese"):
        return 8
    return 9


if __name__ == "__main__":
    sns.set_theme(style="whitegrid")

    token_metrics = ("subtoken_ratio", "unk_ratio",
                     "split_token_ratio", "ttr",
                     "TARGET_SUBTOKS_IN_TRAIN",
                     "TARGET_SUBTOK_TYPES_IN_TRAIN",
                     "TARGET_WORD_TOKENS_IN_TRAIN",
                     "TARGET_WORD_TYPES_IN_TRAIN")

    with open("../results/correlation_missing.log", "w+") as f_o:
        pass
    palette_name = "plasma"  # "plasma", "hot", "YlGnBu_r"
    metric2stats = {}
    for stats_file in glob("../results/stats-*"):
        print("Processing " + stats_file)
        df, missing_data = process_data_stats(stats_file)
        if not missing_data.empty:
            print(missing_data)
            with open("../results/correlation_missing.log", "a") as f_o:
                f_o.write(str(missing_data))
                f_o.write("\n")
        train_name = df.train.unique()[0]
        target_name = "_".join(df.target.unique()).replace(" ", "-")
        for token_metric in token_metrics:
            if token_metric.startswith("TARGET"):
                palette = reverse_palette(palette_name)
            else:
                palette = palette_name
            folder = "../figures/" + token_metric.lower()
            Path(folder).mkdir(parents=True, exist_ok=True)
            for y_score in ("F1_MACRO", "ACCURACY"):
                score_short = "f1" if y_score.startswith("F1") else "acc"
                png_name = f"{folder}/{train_name}_{target_name}_{score_short}_{token_metric}.png"
                stats = plot(df, y_score, token_metric, palette, png_name)
                _stats = metric2stats.get(token_metric + "_" + y_score, [])
                metric2stats[token_metric + "_" + y_score] = _stats + stats

    old_setups = ("Spanish  ancoraspa", "Spanish    Picard",
                  "Dutch    Swiss German", "Dutch   Alsatian G.",)

    for metric in metric2stats:
        filename = "../results/correlation_" + metric + ".tsv"
        with open(filename, "w+", encoding="utf8") as f_out:
            print("Writing correlation stats to " + filename)
            f_out.write("TRAIN\tTARGET\tRHO_MONOLINGUAL\tP_MONOLINGUAL\t"
                        "RHO_MBERT\tP_MBERT\tRHO_XLMR\tP_XLMR\n")
            stats = metric2stats[metric]
            for s in sorted(stats, key=lambda x: sort_score(x)):
                skip_item = False
                for old in old_setups:
                    if s.startswith(old):
                        skip_item = True
                        break
                if not skip_item:
                    f_out.write(s + "\n")
