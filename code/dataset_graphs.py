from collections import OrderedDict
from glob import glob
import math

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


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


# TODO comparable colours across plots!

def make_colour_bar(df, g, palette_name):
    # Replace the one-item-per-scatterplot-value legend
    # with a colour bar
    g._legend.remove()
    vmin = df.subtok_ratio_diff_abs.min()
    vmax = df.subtok_ratio_diff_abs.max()
#     vmin, vmax = 0, 1.3
    g.fig.subplots_adjust(right=.94)
    # left, bottom, width, height
    cax = g.fig.add_axes([.95, .2, .01, .33])
    points = plt.scatter([], [], c=[],
                         vmin=vmin, vmax=vmax,
                         cmap=plt.cm.get_cmap(palette_name))
    cbar = g.fig.colorbar(points, cax=cax)
    cbar.ax.tick_params(right=False)
    cbar.ax.tick_params(axis='y', pad=1)
    cbar.ax.set_yticks([x / 10 for x in range(
        math.ceil(vmin * 10), math.ceil(vmax * 10), 2)])
#     cbar.ax.set_yticks([0, 0.4, 0.8, 1.2])
    cbar.set_label("Absolute subtoken ratio difference",
                   fontsize=11)


def plot_performance(df, png_name, f1=True, palette_name="coolwarm"):
    y = "F1_MACRO_AVG_DEV" if f1 else "ACCURACY_AVG_DEV"
    ylabel = "F1 macro" if f1 else "Accuracy"
#     palette = {x: plt.cm.get_cmap(palette_name)(x)
#                for x in df.subtok_ratio_diff_abs}
    palette = palette_name
    g = sns.catplot(
        data=df, x="noise", y=y, col="PLM",
        hue="subtok_ratio_diff_abs",
        palette=palette,
        kind="point", height=6, aspect=0.3,
    )
    g.set(xlabel="", ylabel=ylabel)
    g.set_titles("{col_name}")
    g.fig.text(x=0.5, y=0, horizontalalignment='center',
               s='Noise level')
    g.despine(left=True)
    make_colour_bar(df, g, palette_name)

    g.savefig(png_name)


def plot_subtoks(df, plms, png_name, difference=False,
                 palette_name="coolwarm"):
    if difference:
        y = "SUBTOKEN_RATIO_DIFF"
        ylabel = "Subtoken ratio difference (dev – train)"
    else:
        y = "SUBTOKEN_RATIO_TRAIN"
        ylabel = "Subtoken ratio (train)"
    g = sns.catplot(
        data=df, x="noise", y=y, col="PLM",
        hue="subtok_ratio_diff_abs",
        palette=palette_name,
        kind="point", height=6, aspect=0.3,
    )
    g.set(xlabel="", ylabel=ylabel)
    g.set_titles("{col_name}")
    g.fig.text(x=0.5, y=0, horizontalalignment='center',
               s='Noise level')
    g.despine(left=True)

    # Draw lines indicating where the optimal values
    # (train ratio == dev ratio) would be.
    zero_diff_col = plt.cm.get_cmap("coolwarm")(0)
    if difference:
        g.map(plt.axhline, y=0, ls='-', c='black')
    else:
        for i, plm in enumerate(plms):
            g.axes[0][i].axhline(
                df.loc[df['PLM'] == plm, 'SUBTOKEN_RATIO_DEV'].iloc[0],
                c=zero_diff_col)

    make_colour_bar(df, g, palette_name)

    g.savefig(png_name)


sns.set_theme(style="whitegrid")

for stats_file in glob("../results/stats-*"):
    df = process_data_stats(stats_file)
    plms = list(OrderedDict.fromkeys(df['PLM']))
    train_name = list(df.TRAIN_SET.apply(lambda x: x.split("_", 2)[1]))[0]
    dev_name = list(df.DEV_SET.apply(lambda x: x.split("_", 2)[1]))[0]
    plot_performance(df, f"../figures/{train_name}_{dev_name}_f1.png")
    plot_subtoks(df, plms,
                 f"../figures/{train_name}_{dev_name}_subtok-diff.png")
