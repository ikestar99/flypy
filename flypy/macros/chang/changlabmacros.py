#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 09:00:00 2024
@author: ike
"""


import numpy as np
import pandas as pd
import os.path as op
import seaborn as sns
import matplotlib.pyplot as plt

from flypy.macros.chang.utils import *
from flypy.macros.chang.constants import *


# set graphing backend
sns.set_theme(style="ticks")
palette = sns.color_palette("rocket_r")


def data_loading_pipeline(
        raw_data_path: str,
        raw_data_key: str,
        dict_save_path: str,
        count_blocks: bool,
):
    """
    Load analysis output if it exists, otherwise start from scratch.
    """
    graph_data = (
        load_analyzed_data(dict_save_path) if op.isfile(dict_save_path)
        else dict())

    """
    Load raw data from pickle file and process into TimeSeriesDataset object
    for easy manipulation.
    """
    data_df = load_raw_dataframe(
        file=raw_data_path,
        col_dict=raw_data_key,
        cols_include=[C_TIME, C_BLOCK, IDENTIFIER, C_NATO, C_TRACE],
        cols_explode=[IDENTIFIER, C_NATO, C_TRACE]
    )
    data_df = filter_format_dataframe(
        data=data_df,
        date_format=[C_TIME, slice(-19, -9), "%Y_%m_%d", C_EDAY],
        label_format=[IDENTIFIER, C_GROUND],
        binary_thresh=[(C_BLOCK, ">=", FIRST_BLOCK)],
        include_filter=[(IDENTIFIER, list(range(458, 484)))],
        cols_sorting=[C_TIME, C_BLOCK, C_GROUND],
        cols_subset={C_INTERVAL: (C_EDAY, N_DAYS)}
    )
    time_data = dataframe_to_time_dataset(
        data=data_df.copy(),
        col_trace=C_TRACE,
        cols_meta=[C_INTERVAL, C_BLOCK, C_GROUND],
        timepoints=np.arange(T0, T1, 1 / HZ),
        axis_split={-1: N_ELECTRODES},
        expand=[
            (-2, C_FREQUENCY, F_LABEL),
            (-1, C_ELECTRODE, list(range(N_ELECTRODES)))],
        f_norm=NORM_FUNC
    )
    print("raw data loaded, converted to TimeSeries dataset")

    if count_blocks:
        name = "block count vs elapsed time"
        graph_data[name] = (
            data_df[[C_INTERVAL, C_EDAY, C_BLOCK]].copy().drop_duplicates(
            ).groupby([C_INTERVAL, C_EDAY])[[C_BLOCK]].count().reset_index())

    return time_data


def trial_aggregated_pipeline(
        time_data: TimeSeries,
        dict_save_path: str,
        correlate_groups: bool,
        find_similarity: bool
):
    """
    Load analysis output if it exists, otherwise start from scratch.
    """
    graph_data = (
        load_analyzed_data(dict_save_path) if op.isfile(dict_save_path)
        else dict())

    """
    Aggregate traces across trials and split dataset into reference and
    analysis sets.
    """
    time_data = time_data.apply_2d_function(
        cols=[C_INTERVAL, C_GROUND, C_FREQUENCY, C_ELECTRODE],
        func=TRIALS_FUNC, axis=0)
    time_data = time_data.clip_traces(
            np.arange(DTS0 * HZ, DTSI * HZ, dtype=int))
    print("raw data aggregated, clipped to signal, reference set extracted")

    if correlate_groups:
        """
        Compute pairwise correlations in raw traces per electrode within and
        between labels across a unit of time, using an initial set of
        observations as a continuous standard of comparison.
        """
        name = "correlation {} vs elapsed day"
        w_name = "within"
        a_name = "across"
        s_name = "separability"
        within, across, separability = correlation_over_time(
            data=time_data,
            refs=time_data.filter(C_INTERVAL, "<=", REF_INTERVAL),
            col_label=C_GROUND,
            col_group=C_INTERVAL,
            cols_source=[C_FREQUENCY, C_ELECTRODE],
        )
        graph_data[name.format(w_name)] = within
        graph_data[name.format(a_name)] = across
        graph_data[name.format(s_name)] = separability
        save_analyzed_data(dict_save_path, graph_data)
        print("data correlated by group over time")

    if find_similarity:
        """
        Compute pairwise cosine similarities in feature vectors per electrode
        within and between labels across a unit of time, using an initial set
        of observations as a continuous standard of comparison.
        """
        vector_data = timeseries_to_vector_dataset(
            data=time_data,
            f_feature=RAW_FEATURE,
            cols_source=[C_INTERVAL, C_GROUND])
        name = "cosine similarity {} vs elapsed day"
        w_name = "within"
        a_name = "across"
        s_name = "separability"
        within, across, separability = cosine_similarity_over_time(
            data=vector_data,
            refs=vector_data[vector_data.labels[:, 0] <= REF_INTERVAL],
            col_label=C_GROUND,
            col_group=C_INTERVAL,
            mapping={C_INTERVAL: 0, C_GROUND: 1}
        )
        graph_data[name.format(w_name)] = within
        graph_data[name.format(a_name)] = across
        graph_data[name.format(s_name)] = separability
        save_analyzed_data(dict_save_path, graph_data)
        print("data similarities computed by group over time")


def individual_trial_pipeline(
        time_data: TimeSeries,
        dict_save_path: str,
        search_grid: bool,
        calculate_salience: bool,
        do_dim_reduction: bool
):
    """
    Load analysis output if it exists, otherwise start from scratch.
    """
    graph_data = (
        load_analyzed_data(dict_save_path) if op.isfile(dict_save_path)
        else dict())

    if search_grid:
        """
        Perform a grid search across a set of feature extraction functions and
        low-level ML models to identify the (model, feature) pair that
        maximizes classification accuracy.
        """
        for s in [F_LABEL] + F_LABEL:
            name = f"classification vs feature function {s}"
            accuracy, model, feature = classification_grid_search(
                data=time_data[{C_FREQUENCY: s}],
                func_dict=FEATURES,
                model_dict=CLASSIFIERS,
                cols_source=[C_BLOCK, C_GROUND],
                col_label=C_GROUND
            )
            graph_data[name] = accuracy
            print(f"model: {model}\nfeature: {feature}\nfrequency: {s}\n")
            save_analyzed_data(dict_save_path, graph_data)

        print("grid search completed for all (feature, model) combinations")

    if calculate_salience:
        """
        Identify indices of electrodes with greatest variance across trials.
        The dataset will be limited to these salient electrodes to improve
        compute time in further steps.
        """
        name = f"label {TOP_FEATURE} " + "{}"
        c_name = "central tendency"
        v_name = "coefficient of variation"
        center, cvs = source_label_variance(
            data=time_data,
            f_feature=FEATURES[TOP_FEATURE],
            col_label=C_GROUND,
            col_group=C_INTERVAL,
            cols_source=[C_ELECTRODE, C_FREQUENCY],
            f_labels=np.mean,
        )
        graph_data[name.format(c_name)] = center
        graph_data[name.format(v_name)] = cvs
        save_analyzed_data(dict_save_path, graph_data)
        print("salient sources extracted for top features")

    if do_dim_reduction:
        """
        Perform dimensionality reduction on extracted feature vectors.
        """
        vector_data = timeseries_to_vector_dataset(
            data=time_data[{C_FREQUENCY: TOP_FREQUENCIES}],
            f_feature=FEATURES[TOP_FEATURE],
            cols_source=[C_INTERVAL, C_GROUND])
        name = f"{DIM_ENCODE} {TOP_FEATURE} " + "{}"
        c_name = "components"
        v_name = "variance ratio"
        reducer, kwargs = (ENCODERS[DIM_ENCODE] + [{}])[:2]
        components, variance = encode_with_dim_reducer(
            data=vector_data,
            refs=vector_data[vector_data.labels[:, 0] <= REF_INTERVAL],
            reducer=reducer(**kwargs),
            mapping={C_INTERVAL: 0, C_GROUND: 1}
        )
        graph_data[name.format(c_name)] = components
        graph_data[name.format(v_name)] = variance
        save_analyzed_data(dict_save_path, graph_data)
        print("Dimensions reduced and variance accounted for")


def generate_plots(
        dict_save_path,
        figure_save_template,
        graph_block_counts,
        graph_group_correlations,
        graph_group_similarities,
        graph_search_grid,
        graph_source_salience,
        graph_encode,
        n_boot=10,
        offset=2
):
    graph_data = load_analyzed_data(dict_save_path)

    if graph_block_counts:
        name = "block count vs elapsed time"
        data = graph_data[name]
        for x_col in (C_INTERVAL, C_EDAY):
            save = figure_save_template.format(f"{name} {x_col}")
            ax = sns.histplot(data=data, x=x_col, binwidth=4)
            sns.despine()
            f = ax.get_figure()
            plt.tight_layout()
            f.savefig(save, dpi=400)
            plt.clf()

        print("block counts graphed")

    if graph_group_correlations:
        w_name = "within"
        a_name = "across"
        s_name = "separability"
        c_value = "correlation"
        s_value = "separability"
        name = "correlation {} vs elapsed day"
        for subset in (w_name, a_name):
            sub_name = name.format(subset)
            data = graph_data[sub_name]
            data = data.groupby(
                [C_INTERVAL, C_GROUND, C_ELECTRODE, C_FREQUENCY]).mean()
            data = data.reset_index()
            for group in (C_GROUND, C_ELECTRODE):
                save = figure_save_template.format(f"{sub_name} {group}")
                f = sns.relplot(
                    data=data, x=C_INTERVAL, y=c_value, hue=group,
                    row=C_FREQUENCY, legend='brief', kind='line', aspect=2,
                    n_boot=n_boot)
                sns.despine(offset=offset, trim=True)
                f.add_legend()
                plt.tight_layout()
                f.savefig(save)
                plt.clf()

        sub_name = name.format(s_name)
        save = figure_save_template.format(sub_name)
        data = graph_data[sub_name].groupby(
            [C_INTERVAL, C_ELECTRODE, C_FREQUENCY]).mean().reset_index()
        ax = sns.lineplot(
            data=data, x=C_INTERVAL, y=s_value, hue=C_FREQUENCY,
            legend="full", n_boot=n_boot)
        sns.despine(offset=offset, trim=True)
        # ax.set_aspect(0.5)
        f = ax.get_figure()
        plt.tight_layout()
        f.savefig(save, dpi=400)
        plt.clf()
        print("group correlations graphed")

    if graph_group_similarities:
        w_name = "within"
        a_name = "across"
        s_name = "separability"
        c_value = "cosine"
        s_value = "separability"
        name = "cosine similarity {} vs elapsed day"
        for subset in (w_name, a_name):
            sub_name = name.format(subset)
            save = figure_save_template.format(f"{sub_name} {subset}")
            data = graph_data[sub_name]
            ax = sns.lineplot(
                data=data, x=C_INTERVAL, y=c_value, hue=C_GROUND,
                legend="brief", n_boot=n_boot)
            sns.despine(offset=offset, trim=True)
            # ax.set_aspect(0.5)
            f = ax.get_figure()
            plt.tight_layout()
            f.savefig(save, dpi=400)
            plt.clf()

        sub_name = name.format(s_name)
        save = figure_save_template.format(sub_name)
        data = graph_data[sub_name]
        ax = sns.lineplot(data=data, x=C_INTERVAL, y=s_value, n_boot=n_boot)
        sns.despine(offset=offset, trim=True)
        # ax.set_aspect(0.5)
        plt.tight_layout()
        f = ax.get_figure()
        f.savefig(save, dpi=400)
        plt.clf()
        print("group similarities graphed")

    if graph_search_grid:
        for s in [F_LABEL] + F_LABEL:
            name = f"classification vs feature function {s}"
            save = figure_save_template.format(name)
            data = graph_data[name]
            ax = sns.heatmap(
                data=data, cbar=True, vmin=0, vmax=np.max(data.values))
            f = ax.get_figure()
            ax.set_aspect(data.shape[1]/data.shape[0])
            plt.tight_layout()
            f.savefig(save, dpi=400)
            plt.clf()

        print("grid search graphed")

    if graph_source_salience:
        name = f"label {TOP_FEATURE} " + "{}"
        c_name = "central tendency"
        v_name = "coefficient of variation"
        c_value = "value"
        for group in (c_name, v_name):
            sub_name = name.format(group)
            data = graph_data[sub_name].groupby(
                [C_INTERVAL, C_FREQUENCY, C_ELECTRODE]).mean().reset_index()
            for freq in TOP_FREQUENCIES:
                save = figure_save_template.format(f"{sub_name} {freq}")
                sub_data = data.loc[data[C_FREQUENCY] == freq].pivot(
                    index=C_ELECTRODE, columns=C_INTERVAL, values=c_value)
                ax = sns.heatmap(data=sub_data, cbar=True)
                f = ax.get_figure()
                plt.tight_layout()
                f.savefig(save, dpi=400)
                plt.clf()

        print("source salience graphed")

    if graph_encode:
        c_name = "components"
        v_name = "variance ratio"
        d_value = "value"
        v_value = "variance ratio"
        name = f"{DIM_ENCODE} {TOP_FEATURE} " + "{}"
        subname = name.format(c_name)
        save = figure_save_template.format(subname)
        data = graph_data[subname].iloc[:, :4].melt(
            id_vars=[C_INTERVAL, C_GROUND], var_name=c_name,
            value_name=d_value)
        f = sns.relplot(
            data=data, x=C_INTERVAL, y=d_value, hue=C_GROUND, row=c_name,
            legend='brief', kind='line', aspect=2, n_boot=n_boot)
        sns.despine(offset=offset, trim=True)
        f.add_legend()
        plt.tight_layout()
        f.savefig(save)
        plt.clf()

        subname = name.format(v_name)
        save = figure_save_template.format(subname)
        data = graph_data[subname]
        data[v_value] = data[v_value].cumsum()
        ax = sns.lineplot(data=data, x=c_name, y=v_value, n_boot=0)
        sns.despine(offset=offset, trim=True)
        # ax.set_aspect(0.5)
        f = ax.get_figure()
        plt.tight_layout()
        f.savefig(save, dpi=400)
        plt.clf()
        print("dim reduction graphed")


if __name__ == "__main__":
    pickle_partial = "/Users/ike/Documents/Lab/Chang Lab/Data/df_ike.pkl"
    pickle_total = "/Users/ike/Documents/Lab/Chang Lab/Data/all_nato_data.pkl"

    # dict_save = "/Users/ike/Desktop/Chang Lab/graph_data.pkl"
    dict_save = "/Users/ike/Desktop/Chang Lab/graph_data_total_grid_search.pkl"
    figure_path = "/Users/ike/Desktop/Chang Lab/Figures/{}.png"

    # raw_time_series_dataset = data_loading_pipeline(
    #     raw_data_path=pickle_total,
    #     raw_data_key="df",
    #     dict_save_path=dict_save,
    #     count_blocks=False,
    # )
    #
    # trial_aggregated_pipeline(
    #     time_data=raw_time_series_dataset,
    #     dict_save_path=dict_save,
    #     correlate_groups=False,
    #     find_similarity=False,
    # )
    #
    # individual_trial_pipeline(
    #     time_data=raw_time_series_dataset,
    #     dict_save_path=dict_save,
    #     search_grid=True,
    #     calculate_salience=False,
    #     do_dim_reduction=False
    # )

    generate_plots(
        dict_save_path=dict_save,
        figure_save_template=figure_path,
        graph_block_counts=False,
        graph_group_correlations=False,
        graph_group_similarities=False,
        graph_search_grid=True,
        graph_source_salience=False,
        graph_encode=False
    )
