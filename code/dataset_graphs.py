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
    return pretty_print_train(target)


def pretty_print_train(train):
    train = train.lower()
    if train == "gsd":
        return "French"
    if train == "ancora-spa" or train == "ancoraspa":
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
    if train == "Spanish":
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
        return "Subtoken-level UNK ratio"
    if token_metric == "ttr":
        return "Subtoken-level TTR"
    if token_metric == "split_token_ratio":
        return "Split word ratio"
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


def process_data_stats(filename):
    df = pd.read_csv(filename, sep='\t')
    df["train"] = df["TRAIN_SET"].apply(
        lambda x: x.split("_", 2)[1].replace("-full", ""))
    df["target_is_stdlang"] = df.apply(
        lambda x: x.TARGET_SET.startswith("dev_" + x.train)
        or x.TARGET_SET.startswith("test_" + x.train),
        axis=1)
    df["train"] = df["train"].apply(lambda x: pretty_print_train(x))
#     df.drop(df[df.target_is_stdlang].index, inplace=True)
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
        df[diff.lower().replace("diff", "ratio")] = df[
            diff.replace("DIFF", "TARGET")] / df[diff.replace("DIFF", "TRAIN")]
        df[diff.replace("DIFF", "RATIO_TARGET")] = 1.0
    missing_data = pd.concat((df[df.ACCURACY == -1],
                              df[df.ACCURACY.isnull()],
                              df[df.F1_MACRO == -1],
                              df[df.F1_MACRO.isnull()]))[
        ["SETUP_NAME", "SEED", "TARGET_SET"]]
    df = df.drop(missing_data.index)
    return df, missing_data


def plot(df, y_score, token_metric, palette_name, png_name,
         analyze_correlation=True, color_bar_vertical=True,
         full_setup_in_title=True,
         custom_score_ticklabels=None, custom_tok_ticklabels=None):
    if (token_metric.startswith("TARGET_")
        or token_metric.endswith("ratio_ratio")
            or token_metric == "ttr_ratio"):
        hue = token_metric
        y_tok = token_metric
    else:
        hue = token_metric + "_diff_abs"
        y_tok = token_metric.upper() + "_TRAIN"

    target2row = {tgt: i * 2 for i, tgt in enumerate(df["target"].unique())}
    plm2col = {col: i for i, col in enumerate(df["PLM"].unique())}
    n_targets = len(target2row)
    n_plms = len(plm2col)
    n_cols = n_plms
    n_rows = 2 * n_targets
    width_ratios = [4 for _ in range(n_cols)]
    height_ratios = [x for doub in [(4, 3.5) for _ in range(n_targets)]
                     for x in doub]
    if color_bar_vertical:
        width_ratios += [1]
        n_cols += 1
    else:
        height_ratios += [1]
        n_rows += 1
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 2.5, n_rows * 2.2),
        sharey="row", sharex="col",
        gridspec_kw={"width_ratios": width_ratios,
                     "height_ratios": height_ratios,
                     # vertical/horizontal space between
                     # subplots:
                     # "hspace": 0.1,
                     "wspace": 0.14}
    )
    y_pos_corr = 1.15 * (df[y_tok].max() - df[y_tok].min()) + df[y_tok].min()

    if token_metric.startswith("TARGET_"):
        vmax = 1.0
        vmin = df[hue].min()
    else:
        vmin = 0.0
        vmax = df[hue].max()
    palette = {x: plt.cm.get_cmap(palette_name)
               ((x - vmin) / (vmax - vmin))
               for x in df[hue]}
    corr_stats = {}
    measure_stats = {}
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
        if full_setup_in_title:
            score_plot.set_title(setup)
        else:
            score_plot.set_title(plm)

        # Draw lines indicating where the optimal values
        # (train ratio == target ratio) would be.
        target_measure = None
        if not token_metric.startswith("TARGET_"):
            y_target = token_metric.upper() + "_TARGET"
            zero_diff_col = plt.cm.get_cmap(palette_name)(0)
            target_measure = df.loc[df['setup'] == setup, y_target].iloc[0]
            axes[row + 1, col].axhline(target_measure, c=zero_diff_col)
            axes[row + 1, col].text(100, target_measure,
                                    f"{target_measure:.2f}", fontsize=9)

        # Plot the tokenization measure
        token_plot = sns.scatterplot(
            data=df_for_setup, x="noise", y=y_tok,
            hue=hue, palette=palette,
            ax=axes[row + 1, col],
        )
        token_plot.set_xticks([0, 15, 35, 55, 75, 95])

        # Add correlation info
        if analyze_correlation:
            label = ""
            try:
                avg_score = df_for_setup[y_score].mean()
            except ValueError:
                avg_score = "??"
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
            try:
                corr_stats[target][plm] = [
                    str(avg_score), str(rho), str(rho_p)]
            except KeyError:
                corr_stats[target] = {
                    plm: [str(avg_score), str(rho), str(rho_p)]}
            axes[row + 1, col].text(0, y_pos_corr, label, fontsize=9)

        acc_averages = []
        measure_train_averages = []
        noise_lvls = (0, 15, 35, 55, 75, 95)
        acc_0_std = None
        for noise in noise_lvls:
            avg_acc = df_for_setup[
                df_for_setup.noise == noise][y_score].mean()
            acc_averages.append(avg_acc)
            axes[row, col].text(noise + 2, avg_acc,
                                f"{avg_acc:.2f}", fontsize=9)
            if noise == 0:
                acc_0_std = df_for_setup[
                    df_for_setup.noise == noise][y_score].std()
            avg_measure_train = df_for_setup[
                df_for_setup.noise == noise][y_tok].mean()
            measure_train_averages.append(str(avg_measure_train))
            axes[row + 1, col].text(noise + 2, avg_measure_train,
                                    f"{avg_measure_train:.2f}", fontsize=9)
        if analyze_correlation:
            saved_scores = corr_stats[target][plm]
            corr_stats[target][plm] = [saved_scores[0]] \
                + [str(a) for a in acc_averages] + saved_scores[1:]
        try:
            measure_stats[target][plm] = measure_train_averages
        except KeyError:
            measure_stats[target] = {plm: measure_train_averages}

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
    for i in range(n_targets):
        row1 = i * 2
        row2 = row1 + 1
        # Add y axis labels to the leftmost figures
        axes[row1, 0].set_ylabel(pretty_score_label(y_score))
        axes[row2, 0].set_ylabel(pretty_tokenization_label(token_metric))
        axes[row1, 0].tick_params(axis='y', which='major', pad=-2)
        axes[row2, 0].tick_params(axis='y', which='major', pad=-2)
        if custom_score_ticklabels:
            axes[row1, 0].set_yticklabels(custom_score_ticklabels)
        if custom_tok_ticklabels:
            axes[row2, 0].set_yticklabels(custom_tok_ticklabels)
        if color_bar_vertical:
            # Clean the column to be used for the colourbar
            axes[row1, -1].axis("off")
            axes[row2, -1].axis("off")
    if color_bar_vertical:
        fig.text(0.5, 0, "Noise (%)", ha='center')
    else:
        fig.text(0.5, (0.4 + 1 * height_ratios[-1]) / sum(height_ratios),
                 "Noise (%)", ha='center')
        # Clean the row to be used for the colourbar
        # Add the noise ticks instead to the row above
        for j in range(n_plms):
            axes[-1, j].axis("off")
            axes[n_rows - 2, j].xaxis.set_tick_params(labelbottom=True)

    # Add colour bar
    if color_bar_vertical:
        cbar_axes = [axes[i, -1] for i in range(n_targets)]
        orientation = "vertical"
        aspect = 35
        shrink = 1
        fraction = 1.3 / n_rows
        labelpad = 0
    else:
        cbar_axes = [axes[-1, j] for j in range(n_plms)]
        orientation = "horizontal"
        aspect = 35
        shrink = 0.75
        fraction = 0.5 / n_cols
        labelpad = -10.0
    points = plt.scatter([], [], c=[],
                         vmin=0,  # vmin=vmin,
                         vmax=vmax,
                         cmap=plt.cm.get_cmap(palette_name))
    cbar = plt.colorbar(points,
                        orientation=orientation,
                        ax=cbar_axes,
                        aspect=aspect,
                        shrink=shrink,
                        fraction=fraction,
                        pad=0,)
    cbar.ax.tick_params(right=False)
    if color_bar_vertical:
        cbar.ax.tick_params(axis='y', pad=0)
        cbar.ax.set_yticks([vmin, vmax])
        cbar.ax.set_yticklabels([math.ceil(vmin * 100) / 100,
                                 math.floor(vmax * 100) / 100])
    else:
        cbar.ax.tick_params(axis='x', pad=1)
        cbar.ax.set_xticks([vmin, vmax])
        cbar.ax.set_xticklabels([math.ceil(vmin * 100) / 100,
                                 math.floor(vmax * 100) / 100], fontsize=11)
    cbar.set_label(pretty_colorbar_label(hue), fontsize=11,
                   labelpad=labelpad)
    plt.show()
    if png_name:
        fig.savefig(png_name, pad_inches=0)

    # Make correlation/result stats print-ready
    stats_c = []
    stats_m = []
    if analyze_correlation:
        train = df_for_setup["train"].unique()[0]
        mono = train2monolingual(train)
        for target in corr_stats:
            try:
                stats_c_mono = corr_stats[target][mono]
                stats_m_mono = measure_stats[target][mono]
            except KeyError:
                stats_c_mono = ("", "", "", "", "", "", "", "", "")
                stats_m_mono = ("", "", "", "", "", "", "")
            try:
                stats_c_mbert = corr_stats[target]["mBERT"]
                stats_m_mbert = measure_stats[target]["mBERT"]
            except KeyError:
                stats_c_mbert = ("", "", "", "", "", "", "", "", "")
                stats_m_mbert = ("", "", "", "", "", "", "")
            try:
                stats_c_xlmr = corr_stats[target]["XLM-R"]
                stats_m_xlmr = measure_stats[target]["XLM-R"]
            except KeyError:
                stats_c_xlmr = ("", "", "", "", "", "", "", "", "")
                stats_m_xlmr = ("", "", "", "", "", "", "")
            stat_c = "\t".join((train, target, *stats_c_mono,
                                *stats_c_mbert, *stats_c_xlmr))
            stat_m = "\t".join((train, target, *stats_m_mono,
                                *stats_m_mbert, *stats_m_xlmr))
            print(stat_c)
            print(stat_m)
            stats_c.append(stat_c)
            stats_m.append(stat_m)
    return stats_c, stats_m


def reverse_palette(palette_name):
    if palette_name.endswith("_r"):
        return palette_name[:-2]
    return palette_name + "_r"


if __name__ == "__main__":
    sns.set_theme(style="whitegrid")

    # Correlation stats and figures for all set-ups

    token_metrics = (
        "subtoken_ratio",
        "unk_ratio",
        "split_token_ratio",
        "ttr",
        # "subtoken_ratio_ratio", "split_token_ratio_ratio", "ttr_ratio",
        "TARGET_SUBTOKS_IN_TRAIN",
        "TARGET_SUBTOK_TYPES_IN_TRAIN",
        "TARGET_WORD_TOKENS_IN_TRAIN",
        "TARGET_WORD_TYPES_IN_TRAIN",
    )

    # with open("../results/correlation_missing.log", "w+") as f_o:
    #     pass
    palette_name = "plasma"  # "plasma", "hot", "YlGnBu_r"
    metric2stats_c = {}
    metric2stats_m = {}
    # for stats_file in glob("../results/stats-nob*"):
    for stats_file in glob("../results/stats-*"):
        if "mudt" in stats_file or "translit" in stats_file:
            continue
        print("Processing " + stats_file)
        df, missing_data = process_data_stats(stats_file)
    #     if not missing_data.empty:
    #         print(missing_data)
    #         with open("../results/correlation_missing.log", "a") as f_o:
    #             f_o.write(str(missing_data))
    #             f_o.write("\n")
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
                stats_c, stats_m = plot(df, y_score, token_metric,
                                        palette, png_name)
                _stats_c = metric2stats_c.get(token_metric + "_" + y_score, [])
                metric2stats_c[token_metric + "_" + y_score] = _stats_c + stats_c
                if y_score == "ACCURACY":
                    # This is independent of the score metric, so only keep
                    # track of it once.
                    _stats_m = metric2stats_m.get(token_metric, [])
                    metric2stats_m[token_metric] = _stats_m + stats_m

    old_setups = ("Spanish\tancoraspa", "Spanish\tPicard",
                  "Dutch\tSwiss German", "Dutch\tAlsatian G.",)

    for metric in metric2stats_c:
        filename = "../results/correlation_" + metric + ".tsv"
        with open(filename, "w+", encoding="utf8") as f_out:
            print("Writing correlation stats to " + filename)
            f_out.write("TRAIN\tTARGET")
            for model in ("MONOLINGUAL", "MBERT", "XLMR"):
                f_out.write(f"\tAVG_ALL_{model}\tAVG_0_{model}")
                f_out.write(f"\tAVG_15_{model}\tAVG_35_{model}")
                f_out.write(f"\tAVG_55_{model}\tAVG_75_{model}")
                f_out.write(f"\tAVG_95_{model}\tRHO_{model}\tP_{model}")
            f_out.write("\n")
            stats = metric2stats_c[metric]
            for s in sorted(stats, key=lambda x: sort_score(x)):
                skip_item = False
                for old in old_setups:
                    if s.startswith(old):
                        skip_item = True
                        break
                if not skip_item:
                    f_out.write(s + "\n")

    for metric in metric2stats_m:
        filename = "../results/averages_" + metric + ".tsv"
        print("Writing measure averages to " + filename)
        with open(filename, "w+", encoding="utf8") as f_out:
            f_out.write("TRAIN\tTARGET")
            for model in ("MONOLINGUAL", "MBERT", "XLMR"):
                f_out.write(f"\tAVG_0_{model}\tAVG_15_{model}")
                f_out.write(f"\tAVG_35_{model}\tAVG_55_{model}")
                f_out.write(f"\tAVG_75_{model}\tAVG_95_{model}")
            f_out.write("\n")
            stats = metric2stats_m[metric]
            for s in sorted(stats, key=lambda x: sort_score(x)):
                skip_item = False
                for old in old_setups:
                    if s.startswith(old):
                        skip_item = True
                        break
                if not skip_item:
                    f_out.write(s + "\n")

    # Prettier figure for article
    df, _ = process_data_stats("../results/stats-padt.tsv")
    df["target_not_egy"] = df.apply(
        lambda x: "egy" not in x.TARGET_SET, axis=1)
    df.drop(df[df.target_not_egy].index, inplace=True)
    df["model_is_xlmr"] = df.apply(
        lambda x: "xlm" in x.SETUP_NAME, axis=1)
    df.drop(df[df.model_is_xlmr].index, inplace=True)
    plot(df, "ACCURACY", "split_token_ratio", palette_name,
         png_name="../figures/egy.png",
         analyze_correlation=False, color_bar_vertical=False,
         full_setup_in_title=False,
         custom_score_ticklabels=("", "", 0.5, "", 0.6, "", 0.7),
         custom_tok_ticklabels=("", 0.3, "", 0.5, "", 0.7),
         )
