from collections import OrderedDict
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
    df["subtok_ratio_diff_abs"] = df["SUBTOKEN_RATIO_DIFF"].apply(
        lambda x: abs(x))
    return df


def make_colour_bar(df, hue, token_global_max, g,
                    palette_name):
    # Replace the one-item-per-scatterplot-value legend
    # with a colour bar
    g._legend.remove()
    vmin = 0
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


def make_colour_bar(df, hue, token_global_max, g,
                    palette_name):
    # Replace the one-item-per-scatterplot-value legend
    # with a colour bar
    g._legend.remove()
    vmin = 0
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


def plot_performance(df, token_metric, token_global_max, plms,
                     png_name=None, f1=True,
                     palette_name="inferno",
                     use_global_palette=True):
    y = "F1_MACRO_AVG_DEV" if f1 else "ACCURACY_AVG_DEV"
    ylabel = "F1 macro" if f1 else "Accuracy"
    hue = token_metric + "_diff_abs"
    if use_global_palette:
        palette = {x: plt.cm.get_cmap(palette_name)
                   (x / token_global_max)
                   for x in df[hue]}
    else:
        # NOT adjusting the minimum since we want it
        # to remain at 0.
        token_global_max = df[hue].max()
        palette = {x: plt.cm.get_cmap(palette_name)
                   (x / token_global_max)
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
    make_colour_bar(df, hue, token_global_max, g,
                    palette_name)

    if png_name:
        g.savefig(png_name)


def plot_subtoks(df, token_metric, token_global_max, plms,
                 png_name=None,
                 palette_name="inferno",
                 use_global_palette=True):
    if token_metric == "subtoken_ratio":
        ylabel = "Subtoken ratio (train)"
    elif token_metric == "unk_ratio":
        ylabel = "Subtoken-level UNK ratio (train)"
    elif token_metric == "ttr":
        ylabel = "Subtoken-level TTR (train)"
    elif token_metric == "split_token_ratio":
        ylabel = "Split-word ratio (train)"
    else:
        print("Can't recognize token_metric=" + token_metric)
        return

    y = token_metric.upper() + "_TRAIN"
    hue = token_metric + "_diff_abs"
    if use_global_palette:
        palette = {x: plt.cm.get_cmap(palette_name)
                   (x / token_global_max)
                   for x in df[hue]}
    else:
        # NOT adjusting the minimum since we want it
        # to remain at 0.
        token_global_max = df[hue].max()
        palette = {x: plt.cm.get_cmap(palette_name)
                   (x / token_global_max)
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

    # Draw lines indicating where the optimal values
    # (train ratio == dev ratio) would be.
    y_dev = token_metric.upper() + "_DEV"
    zero_diff_col = plt.cm.get_cmap(palette_name)(0)
    for i, plm in enumerate(plms):
        g.axes[0][i].axhline(
            df.loc[df['PLM'] == plm, y_dev].iloc[0],
            c=zero_diff_col)

    g.despine(left=True)
    make_colour_bar(df, hue, token_global_max, g,
                    palette_name)

    if png_name:
        g.savefig(png_name)


if __name__ == "__main__":
    sns.set_theme(style="whitegrid")

    token_metrics = ("subtoken_ratio", "unk_ratio",
                     "split_token_ratio", "ttr")

    dfs = []
    max_scores = {}
    for stats_file in glob("../results/stats-*"):
        df = process_data_stats(stats_file)
        dfs.append(df)
        for metric in token_metrics:
            _max = df[metric + "_diff_abs"].max()
            if metric not in max_scores or _max > max_scores[metric]:
                max_scores[metric] = _max

    palette_name = "plasma"  # or "hot", "cool"
    for df in dfs:
        plms = list(OrderedDict.fromkeys(df['PLM']))
        train_name = list(df.TRAIN_SET.apply(lambda x: x.split("_", 2)[1]))[0]
        dev_name = list(df.DEV_SET.apply(lambda x: x.split("_", 2)[1]))[0]
        for token_metric in token_metrics:
            token_global_max = max_scores[token_metric]
            folder = "../figures/" + token_metric
            Path(folder).mkdir(parents=True, exist_ok=True)
            file_pfx = f"{folder}/{train_name}_{dev_name}_"
            for use_global_palette in (True, False):
                if use_global_palette:
                    file_sfx = "_global-colours.png"
                else:
                    file_sfx = "_local-colours.png"
                plot_performance(df, token_metric, token_global_max, plms,
                                 file_pfx + "f1" + file_sfx,
                                 palette_name=palette_name,
                                 use_global_palette=use_global_palette)
                # plot_performance(df, token_metric, token_global_max, plms,
                #                  file_pfx + "acc" + file_sfx, f1=False,
                #                  palette_name=palette_name)
                plot_subtoks(df, token_metric, token_global_max, plms,
                             file_pfx + "subtok" + file_sfx,
                             palette_name=palette_name,
                             use_global_palette=use_global_palette)
