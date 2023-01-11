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


def pretty_score_label(y_score):
    if y_score == "F1_MACRO_AVG_DEV":
        return "F1 macro"
    if y_score == "ACCURACY_AVG_DEV":
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
    if token_metric == "DEV_SUBTOK_TYPES_IN_TRAIN":
        return "% of dev subtoken types also in train"
    if token_metric == "DEV_SUBTOKS_IN_TRAIN":
        return "% of dev subtokens also in train"
    if token_metric == "DEV_WORD_TYPES_IN_TRAIN":
        return "% of dev word types also in train"
    if token_metric == "DEV_WORD_TOKENS_IN_TRAIN":
        return "% of dev word tokens also in train"
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
    if hue == "DEV_SUBTOK_TYPES_IN_TRAIN":
        return "% of dev subtoken types also in train"
    if hue == "DEV_SUBTOKS_IN_TRAIN":
        return "% of dev subtokens also in train"
    if hue == "DEV_WORD_TYPES_IN_TRAIN":
        return "% of dev word types also in train"
    if hue == "DEV_WORD_TOKENS_IN_TRAIN":
        return "% of dev word tokens also in train"
    return hue


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


def plot(df, y_score, token_metric, palette_name, png_name):
    if token_metric.startswith("DEV_"):
        hue = token_metric
        y_tok = token_metric
    else:
        hue = token_metric + "_diff_abs"
        y_tok = token_metric.upper() + "_TRAIN"

    width_ratios = [4 for _ in range(len(plms))] + [1]
    fig, axes = plt.subplots(2, len(plms) + 1, figsize=(8, 6),
                             sharey="row", sharex="col",
                             gridspec_kw={
                                 "width_ratios": width_ratios},)

    y_pos_corr = 1.15 * (df[y_tok].max() - df[y_tok].min()) + df[y_tok].min()

    vmin = df[hue].min()
    vmax = df[hue].max()
    palette = {x: plt.cm.get_cmap(palette_name)
               ((x - vmin) / (vmax - vmin))
               for x in df[hue]}

    for i, (plm, df_for_plm) in enumerate(df.groupby("PLM")):
        # Plot F1/accuracy
        score_plot = sns.scatterplot(
            data=df_for_plm, x="noise", y=y_score,
            hue=hue, palette=palette,
            ax=axes[0, i],
        )
        score_plot.set_title(plm)

        if not token_metric.startswith("DEV_"):
            # Draw lines indicating where the optimal values
            # (train ratio == dev ratio) would be.
            y_dev = token_metric.upper() + "_DEV"
            zero_diff_col = plt.cm.get_cmap(palette_name)(0)
            axes[1, i].axhline(
                df.loc[df['PLM'] == plm, y_dev].iloc[0],
                c=zero_diff_col)

        # Plot the tokenization measure
        token_plot = sns.scatterplot(
            data=df_for_plm, x="noise", y=y_tok,
            hue=hue, palette=palette,
            ax=axes[1, i],
        )
        token_plot.set_xticks([0, 15, 35, 55, 75, 95])

        # Add correlation info
        df_sub = df[df.PLM == plm]
        if len(df_sub[hue]) >= 2:
            r, r_p = pearsonr(df_sub[hue], df_sub[y_score])
            rho, rho_p = spearmanr(df_sub[hue], (df_sub[y_score]))
            axes[1, i].text(
                0, y_pos_corr,
                f"r = {r:.2f} (p = {r_p:.2f})\nρ = {rho:.2f} (p = {rho_p:.2f})",
                fontsize=9)

        # Remove visual clutter
        for row in (0, 1):
            ax = axes[row, i]
            ax.get_legend().remove()
            ax.xaxis.grid()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
        axes[1, i].set_xlabel(None)

    # Update the axis labels
    axes[0, 0].set_ylabel(pretty_score_label(y_score))
    axes[1, 0].set_ylabel(pretty_tokenization_label(token_metric))
    fig.text(0.5, 0, "Noise (%)", ha='center')

    # Add colour bar
    axes[0, -1].axis("off")
    axes[1, -1].axis("off")
    points = plt.scatter([], [], c=[],
                         vmin=vmin, vmax=vmax,
                         cmap=plt.cm.get_cmap(palette_name))
    cbar = plt.colorbar(points,
                        ax=[axes[0, -1], axes[1, -1]],
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


def reverse_palette(palette_name):
    if palette_name.endswith("_r"):
        return palette_name[:-2]
    return palette_name + "_r"


if __name__ == "__main__":
    sns.set_theme(style="whitegrid")

    token_metrics = ("subtoken_ratio", "unk_ratio",
                     "split_token_ratio", "ttr",
                     "DEV_SUBTOKS_IN_TRAIN",
                     "DEV_SUBTOK_TYPES_IN_TRAIN",
                     "DEV_WORD_TOKENS_IN_TRAIN",
                     "DEV_WORD_TYPES_IN_TRAIN")


    palette_name = "plasma"  # "plasma", "hot", "YlGnBu_r"
    for stats_file in glob("../results/stats-*"):
        df = process_data_stats(stats_file)
        plms = list(OrderedDict.fromkeys(df['PLM']))
        train_name = list(df.TRAIN_SET.apply(lambda x: x.split("_", 2)[1]))[0]
        dev_name = list(df.DEV_SET.apply(lambda x: x.split("_", 2)[1]))[0]
        for token_metric in token_metrics:
            if token_metric.startswith("DEV"):
                palette = reverse_palette(palette_name)
            else:
                palette = palette_name
            folder = "../figures/" + token_metric.lower()
            Path(folder).mkdir(parents=True, exist_ok=True)
            y_score = "F1_MACRO_AVG_DEV"
            png_name = f"{folder}/{train_name}_{dev_name}_f1_{token_metric}.png"
            plot(df, y_score, token_metric, palette, png_name)
