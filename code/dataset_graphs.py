from collections import OrderedDict
from glob import glob
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr


def pretty_print_plm(plm):
    if plm == "mbert":
        return "mBERT"
    if plm == "xlmr":
        return "XLM-R"
    if plm == "beto":
        return "Beto"
    plm = plm.replace("bert", "BERT")
    plm = plm[0].upper() + plm[1:]
    return plm


def process_data_stats(filename):
    df = pd.read_csv(filename, sep='\t')
    df["dev_is_stdlang"] = df.apply(
        lambda x: x.DEV_SET.startswith(
            "dev_" + x.TRAIN_SET.split("_", 1)[1].split("-", 1)[0]), axis=1)
    df.drop(df[df.dev_is_stdlang].index, inplace=True)
    df[["PLM", "noise"]] = df.TRAIN_SET.str.split(
        "_", 2).str[2].str.split("_", expand=True)
    df["PLM"] = df["PLM"].apply(lambda x: pretty_print_plm(x))
    df["noise"] = df["noise"].apply(lambda x: 0 if x == "orig" else int(x[4:]))
    for diff in ("SUBTOKEN_RATIO_DIFF", "UNK_RATIO_DIFF",
                 "TTR_DIFF", "SPLIT_TOKEN_RATIO_DIFF"):
        df[diff.lower() + "_abs"] = df[diff].apply(lambda x: abs(x))
    for col in ("F1_MACRO_AVG_DEV", "F1_MACRO_STD_DEV",
                "ACCURACY_AVG_DEV", "ACCURACY_STD_DEV"):
        df[col] = df[col].replace(-1, np.nan)
    return df


def make_colour_bar(df, hue, token_global_max, token_global_min, g,
                    palette_name):
    # Replace the one-item-per-scatterplot-value legend
    # with a colour bar
    g._legend.remove()
    vmin = token_global_min
    vmax = token_global_max
    g.fig.subplots_adjust(right=.94)
    # add_axes: left, bottom, width, height
    cax = g.fig.add_axes([.95, .2, .01, .33])
    points = plt.scatter([], [], c=[],
                         vmin=vmin, vmax=vmax,
                         cmap=plt.cm.get_cmap(palette_name))
    cbar = g.fig.colorbar(points, cax=cax)
    cbar.ax.tick_params(right=False)
    cbar.ax.tick_params(axis='y', pad=0)
    cbar.ax.set_yticks([math.ceil(vmin * 100) / 100,
                        # round(vmax * 50 - vmin * 50) / 100,
                        math.floor(vmax * 100) / 100])
    label = "Absolute "
    if hue.startswith("subtoken_ratio"):
        label += "subtoken ratio"
    elif hue.startswith("unk_ratio"):
        label += "subtoken-level UNK ratio"
    elif hue.startswith("ttr"):
        label += "subtoken-level TTR"
    elif hue.startswith("split_token_ratio"):
        label += "split word ratio"
    label += " difference"
    cbar.set_label(label, fontsize=11)


def plot_performance(df, token_metric, plms,
                     f1=True,
                     raw_value=False,
                     token_global_max=1,
                     token_global_min=0,
                     use_global_upper=True,
                     use_global_lower=False,
                     palette_name="inferno",
                     png_name=None,):
    y = "F1_MACRO_AVG_DEV" if f1 else "ACCURACY_AVG_DEV"
    ylabel = "F1 macro" if f1 else "Accuracy"
    hue = token_metric
    if not raw_value:
        hue += "_diff_abs"

    if not use_global_upper:
        token_global_max = df[hue].max()
    if not use_global_lower:
        token_global_min = df[hue].min()
    divisor = token_global_max - token_global_min
    palette = {x: plt.cm.get_cmap(palette_name)
               ((x - token_global_min) / divisor)
               for x in df[hue]}

    g = sns.catplot(
        data=df, x="noise", y=y, col="PLM",
        hue=hue, palette=palette,
        kind="point", linestyles="",
        height=6, aspect=0.3,
    )
    g.set(xlabel="", ylabel=ylabel)
    g.set_titles("{col_name}")
    g.fig.text(x=0.5, y=0, horizontalalignment='center',
               s='Noise level')

    # Add correlation info to each subplot
    y_pos_corr = df[y].min()
    for ax, plm in zip(g.axes.flat, plms):
        df_sub = df[df.PLM == plm]
        if len(df_sub[hue]) < 2:
            continue
        r, r_p = pearsonr(df_sub[hue], df_sub[y])
        rho, rho_p = spearmanr(df_sub[hue], (df_sub[y]))
        ax.text(
            0, y_pos_corr,
            f"r = {r:.2f} (p = {r_p:.2f})\nρ = {rho:.2f} (p = {rho_p:.2f})",
            fontsize=9)

    g.despine(left=True)
    make_colour_bar(df, hue, token_global_max, token_global_min, g,
                    palette_name)

    if png_name:
        g.savefig(png_name)


def plot_subtoks(df, token_metric, plms,
                 raw_value=False,
                 token_global_max=1,
                 token_global_min=0,
                 use_global_upper=True,
                 use_global_lower=False,
                 palette_name="inferno",
                 png_name=None,):
    if token_metric == "subtoken_ratio":
        ylabel = "Subtoken ratio (train)"
    elif token_metric == "unk_ratio":
        ylabel = "Subtoken-level UNK ratio (train)"
    elif token_metric == "ttr":
        ylabel = "Subtoken-level TTR (train)"
    elif token_metric == "split_token_ratio":
        ylabel = "Split-word ratio (train)"
    elif token_metric == "DEV_SUBTOK_TYPES_IN_TRAIN":
        ylabel = "% of dev subtoken types also in train"
    elif token_metric == "DEV_SUBTOKS_IN_TRAIN":
        ylabel = "% of dev subtokens also in train"
    elif token_metric == "DEV_WORD_TYPES_IN_TRAIN":
        ylabel = "% of dev word types also in train"
    elif token_metric == "DEV_WORD_TOKENS_IN_TRAIN":
        ylabel = "% of dev word tokens also in train"
    else:
        ylabel = token_metric

    if raw_value:
        y = token_metric
        hue = token_metric
    else:
        y = token_metric.upper() + "_TRAIN"
        hue = token_metric + "_diff_abs"

    if not use_global_upper:
        token_global_max = df[hue].max()
    if not use_global_lower:
        token_global_min = df[hue].min()
    divisor = token_global_max - token_global_min
    palette = {x: plt.cm.get_cmap(palette_name)
               ((x - token_global_min) / divisor)
               for x in df[hue]}
#     else:
#         # NOT adjusting the minimum since we want it
#         # to remain at 0.
#         token_global_max = df[hue].max()
#         palette = {x: plt.cm.get_cmap(palette_name)
#                    (x / token_global_max)
#                    for x in df[hue]}
    g = sns.catplot(
        data=df, x="noise", y=y, col="PLM",
        hue=hue, palette=palette,
        kind="point", linestyles="",
        height=6, aspect=0.3,
    )
    g.set(xlabel="", ylabel=ylabel)
    g.set_titles("{col_name}")
    g.fig.text(x=0.5, y=0, horizontalalignment='center',
               s='Noise level')

    if not raw_value or not y.startswith("DEV_"):
        # Draw lines indicating where the optimal values
        # (train ratio == dev ratio) would be.
        y_dev = token_metric.upper() + "_DEV"
        zero_diff_col = plt.cm.get_cmap(palette_name)(0)
        for i, plm in enumerate(plms):
            g.axes[0][i].axhline(
                df.loc[df['PLM'] == plm, y_dev].iloc[0],
                c=zero_diff_col)

    g.despine(left=True)
    make_colour_bar(df, hue, token_global_max, token_global_min, g,
                    palette_name)

    if png_name:
        g.savefig(png_name)


if __name__ == "__main__":
    sns.set_theme(style="whitegrid")

    token_metrics = ("subtoken_ratio", "unk_ratio",
                     "split_token_ratio", "ttr",
                     "DEV_SUBTOKS_IN_TRAIN",
                     "DEV_SUBTOK_TYPES_IN_TRAIN",
                     "DEV_WORD_TOKENS_IN_TRAIN",
                     "DEV_WORD_TYPES_IN_TRAIN")

    dfs = []
    max_scores = {}
    min_scores = {}
    for stats_file in glob("../results/stats-*"):
        df = process_data_stats(stats_file)
        dfs.append(df)
        for metric in token_metrics:
            met = metric
            if metric.startswith("DEV"):
                met = metric
            else:
                met = metric + "_diff_abs"
            _max = df[met].max()
            if metric not in max_scores or _max > max_scores[metric]:
                max_scores[metric] = _max
            _min = df[met].min()
            if metric not in min_scores or _min > min_scores[metric]:
                min_scores[metric] = _min

    palette_name = "plasma"  # or "hot", "cool"
    for df in dfs:
        plms = list(OrderedDict.fromkeys(df['PLM']))
        train_name = list(df.TRAIN_SET.apply(lambda x: x.split("_", 2)[1]))[0]
        dev_name = list(df.DEV_SET.apply(lambda x: x.split("_", 2)[1]))[0]
        for token_metric in token_metrics:
            token_global_max = max_scores[token_metric]
            token_global_min = min_scores[token_metric]
            raw_value = token_metric.startswith("DEV")
            folder = "../figures/" + token_metric.lower()
            Path(folder).mkdir(parents=True, exist_ok=True)
            file_pfx = f"{folder}/{train_name}_{dev_name}_"
            for use_global_palette in (True, False):
                if use_global_palette:
                    file_sfx = "_global-colours.png"
                else:
                    file_sfx = "_local-colours.png"

                plot_performance(df, token_metric, plms, f1=True,
                                 raw_value=raw_value,
                                 token_global_max=token_global_max,
                                 token_global_min=0,
                                 use_global_upper=True, use_global_lower=True,
                                 palette_name=palette_name,
                                 png_name=file_pfx + "f1" + file_sfx)
                plot_subtoks(df, token_metric, plms, raw_value=raw_value,
                             token_global_max=token_global_max,
                             token_global_min=0,
                             use_global_upper=True, use_global_lower=True,
                             palette_name=palette_name,
                             png_name=file_pfx + "subtok" + file_sfx)
