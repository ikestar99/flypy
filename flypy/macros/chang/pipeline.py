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

from flypy.macros.chang.macros import *
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
        include=[C_TIME, C_BLOCK, IDENTIFIER, C_NATO, C_TRACE],
        explode=[IDENTIFIER, C_NATO, C_TRACE]
    )
    data_df = filter_format_dataframe(
        data=data_df,
        include_filter=[(IDENTIFIER, list(range(458, 484)))],
        date_format=[C_TIME, slice(-19, -9), "%Y_%m_%d", C_EDAY],
        label_format=[IDENTIFIER, C_GROUND],
        binary_thresh=[(C_BLOCK, ">=", FIRST_BLOCK)],
        sortby=[C_TIME, C_BLOCK, C_GROUND],
        subset={C_INTERVAL: (C_EDAY, N_DAYS)}
    )

    # instantiate time series with metadata subset
    time_data = dataframe_to_time_dataset(
        data=data_df[[C_INTERVAL, C_BLOCK, C_GROUND, C_TRACE]].copy(),
        col_trace=C_TRACE,
        timepoints=np.arange(T0, T1, 1 / HZ),
        axis_split={-1: N_ELECTRODES},
        expand=[
            (-2, C_FREQUENCY, F_LABEL),
            (-1, C_ELECTRODE, list(range(N_ELECTRODES)))]
    )

    # filter out irrelevant electrodes, downsample trace data
    time_data = time_data[{C_ELECTRODE: SALIENT_ELECTRODES}].downsample(DOWN)
    print(
        f"Loaded shape: ({len(time_data)} traces, "
        f"{time_data.data.shape[1]} timepoints)")

    if count_blocks:
        name = "block count vs elapsed time"
        graph_data[name] = (
            data_df[[C_INTERVAL, C_EDAY, C_BLOCK]].copy().drop_duplicates(
            ).groupby([C_INTERVAL, C_EDAY])[[C_BLOCK]].count().reset_index())

    return time_data


def raw_analysis_pipeline(
        time_data: TimeSeries,
        dict_save_path: str,
        find_peak_window: bool,
        extract_snr: bool,
        correlate_groups: bool,
        # find_separability: bool
):
    """
    Load analysis output if it exists, otherwise start from scratch.
    """
    graph_data = (
        load_analyzed_data(dict_save_path) if op.isfile(dict_save_path)
        else dict())

    if find_peak_window:
        name = "cosine separability vs trial time"
        within = time_data.to_time_vector(
            levels=[C_ELECTRODE],
            col_time=C_TRIAL_TIME
        ).grouped_control(
            cont=None,
            func=EUCLIDEAN_SEP,
            label=C_GROUND,
            group=C_TRIAL_TIME,
            ignore=[C_BLOCK]
        )
        # .extract_statistic(
        #     func=EUCLIDEAN_SEP,
        #     col="value",
        #     levels=[C_GROUND]
        # )
        graph_data[name.format(name)] = within
        save_analyzed_data(dict_save_path, graph_data)
        print("data separability computed by group over trial time")

    if extract_snr:
        name = f"{NOISE_FEATURE} vs elapsed week"
        graph_data[name] = time_data.extract_statistic(
            FEATURES[NOISE_FEATURE], NOISE_FEATURE)

        save_analyzed_data(dict_save_path, graph_data)
        print("feature trajectories over time extracted")

    time_data = time_data.subsample(SIGNAL)
    if correlate_groups:
        """
        Compute pairwise cross correlation separability within electrodes and
        labels.
        """
        name = "cross correlation vs elapsed week"
        within = time_data[{C_INTERVAL: (">", REF_INTERVAL)}].cont_cross_corr(
            cont=time_data[{C_INTERVAL: ("<=", REF_INTERVAL)}],
            label=C_GROUND,
            group=C_INTERVAL,
            ignore=[C_BLOCK]
        )
        graph_data[name.format(name)] = within
        save_analyzed_data(dict_save_path, graph_data)
        print("data correlated by group over time")

    # if find_separability:
    #     """
    #     Compute pairwise cosine separability on feature vectors of time-reduced
    #     feature vectors across electrodes.
    #     """
    #     name = "cosine separability vs elapsed week"
    #     within = time_data.to_feature_vector(
    #         func=FEATURES[TOP_FEATURE],
    #         levels=[C_ELECTRODE])
    #     within = within[{C_INTERVAL: (">", REF_INTERVAL)}].cont_cos_similarity(
    #         cont=within[{C_INTERVAL: ("<=", REF_INTERVAL)}],
    #         label=C_GROUND,
    #         group=C_INTERVAL,
    #         ignore=[C_BLOCK]
    #     )
    #     graph_data[name.format(name)] = within
    #     save_analyzed_data(dict_save_path, graph_data)
    #     print("data separability computed by group over time")


def individual_trial_pipeline(
        time_data: TimeSeries,
        dict_save_path: str,
        data_save_dir: str,
        search_grid: bool,
        do_dim_reduction: bool,
        shallow_curve: bool,
        deep_curve: bool
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
                levels=[C_FREQUENCY, C_ELECTRODE],
                label=C_GROUND
            )
            graph_data[name] = accuracy
            print(f"model: {model}\nfeature: {feature}\nfrequency: {s}\n")
            save_analyzed_data(dict_save_path, graph_data)

        print("grid search completed for all (feature, model) combinations")

    # if do_dim_reduction:
    #     """
    #     Perform dimensionality reduction on extracted feature vectors.
    #     """
    #     vector_data = timeseries_to_vector_dataset(
    #         data=time_data[{C_FREQUENCY: TOP_FREQUENCIES}],
    #         f_feature=FEATURES[TOP_FEATURE],
    #         cols_source=[C_INTERVAL, C_GROUND])
    #     name = f"{DIM_ENCODE} {TOP_FEATURE} " + "{}"
    #     c_name = "components"
    #     v_name = "variance ratio"
    #     reducer, kwargs = (ENCODERS[DIM_ENCODE] + [{}])[:2]
    #     components, variance = latent_representation_over_time(
    #         data=vector_data,
    #         refs=vector_data[vector_data.labels[:, 0] <= REF_INTERVAL],
    #         reducer=reducer(**kwargs),
    #         mapping={C_INTERVAL: 0, C_GROUND: 1}
    #     )
    #     graph_data[name.format(c_name)] = components
    #     graph_data[name.format(v_name)] = variance
    #     save_analyzed_data(dict_save_path, graph_data)
    #     print("Dimensions reduced and variance accounted for")

    if shallow_curve:
        """
        Shallow learning curve with linear model
        """
        name = f"{TOP_MODEL} {TOP_FEATURE} learning curve " + "{}"
        vector_data = time_data[{C_FREQUENCY: F_LABEL[0]}].to_feature_vector(
            func=FEATURES[TOP_FEATURE],
            levels=[C_FREQUENCY, C_ELECTRODE]
        ).scale()
        output = shallow_learning_curves(
            data=vector_data,
            model=CLASSIFIERS[TOP_MODEL][0](
                **(CLASSIFIERS[TOP_MODEL] + [{}])[1]),
            label=C_GROUND,
            group=C_BLOCK,
            step=5
        )
        graph_data[name.format("forward")] = output
        output = shallow_learning_curves(
            data=vector_data,
            model=CLASSIFIERS[TOP_MODEL][0](
                **(CLASSIFIERS[TOP_MODEL] + [{}])[1]),
            label=C_GROUND,
            group=C_BLOCK,
            reverse=True,
            step=5
        )
        graph_data[name.format("reverse")] = output
        save_analyzed_data(dict_save_path, graph_data)
        print("Shallow learning curves generated")

    if deep_curve:
        """
        Deep learning curve with RNN model
        """
        name = "RNN deep learning curve " + "{}"
        vector_data = time_data.subsample(SIGNAL).to_time_vector(
            levels=[C_FREQUENCY, C_ELECTRODE],
            col_time=C_TRIAL_TIME
        ).scale()
        output = deep_learning_curves(
            data=vector_data,
            label=C_GROUND,
            group=C_BLOCK,
            times=C_TRIAL_TIME,
            save_dir=data_save_dir,
            step=5
        )
        graph_data[name.format("forward")] = output
        output = deep_learning_curves(
            data=vector_data,
            label=C_GROUND,
            group=C_BLOCK,
            times=C_TRIAL_TIME,
            save_dir=data_save_dir,
            reverse=True,
            step=5
        )
        graph_data[name.format("reverse")] = output
        save_analyzed_data(dict_save_path, graph_data)
        print("Deep learning curves generated")


def generate_plots(
        dict_save_path: str,
        figure_save_template: str,
        graph_block_counts: bool,
        graph_peak_window: bool,
        graph_extracted_snr: bool,
        graph_group_correlations: bool,
        # graph_group_separability: bool,
        graph_search_grid: bool,
        graph_encode: bool,
        graph_shallow_curve: bool,
        graph_deep_curve: bool,
        n_boot: int = 10,
        offset: int = 2,
        aspect: int = 1.5
):
    graph_data = load_analyzed_data(dict_save_path)

    if graph_block_counts:
        name = "block count vs elapsed time"
        data = graph_data[name]
        for x_col in (C_INTERVAL, C_EDAY):
            ax = sns.histplot(data=data, x=x_col, binwidth=4)
            sns.despine()
            f = ax.get_figure()
            plt.tight_layout()
            f.savefig(figure_save_template.format(f"{name} {x_col}"), dpi=400)
            plt.clf()

        print("block counts graphed")

    if graph_peak_window:
        name = "cosine separability vs trial time"
        axis = "Euclidean separability"
        # axis = "cosine separability"
        data = graph_data[name].rename(columns={"value": axis}).reset_index()
        f = sns.relplot(
            data=data, x=C_TRIAL_TIME, y=axis, row=C_FREQUENCY, hue=C_INTERVAL,
            legend='brief', kind='line', aspect=aspect, n_boot=n_boot)
        sns.despine(offset=offset, trim=True)
        f.add_legend()
        plt.tight_layout()
        f.savefig(figure_save_template.format(f"{name}"))
        plt.clf()

        data[C_INTERVAL] = data[C_INTERVAL].apply(lambda x: WEEK_TO_CHUNK(x))
        f = sns.relplot(
            data=data, x=C_TRIAL_TIME, y=axis, row=C_FREQUENCY, hue=C_INTERVAL,
            legend='full', kind='line', aspect=aspect, n_boot=n_boot)
        sns.despine(offset=offset, trim=True)
        f.add_legend()
        plt.tight_layout()
        f.savefig(figure_save_template.format(f"{name} intervals"))
        plt.clf()

    if graph_extracted_snr:
        name = f"{NOISE_FEATURE} vs elapsed week"
        data = graph_data[name].groupby(
            [C_INTERVAL, C_FREQUENCY, C_ELECTRODE]).mean().reset_index(
            drop=False)
        f = sns.relplot(
            data=data, x=C_INTERVAL, y=NOISE_FEATURE, hue=C_ELECTRODE,
            row=C_FREQUENCY, legend='brief', kind='scatter', aspect=aspect)
        # f.set(yscale="log")
        sns.despine(offset=offset, trim=True)
        f.add_legend()
        plt.tight_layout()
        f.savefig(figure_save_template.format(f"{name}"))
        plt.clf()

        f = sns.lmplot(
            data=data, x=C_INTERVAL, y=NOISE_FEATURE, row=C_FREQUENCY,
            legend=True, x_estimator=np.mean, aspect=aspect, n_boot=n_boot)
        f.map_dataframe(ANNOTATE)
        sns.despine(offset=offset, trim=True)
        f.add_legend()
        plt.tight_layout()
        f.savefig(figure_save_template.format(f"{name} total regression"))
        plt.clf()

        data[C_BINNED] = data[C_INTERVAL].apply(lambda x: WEEK_TO_CHUNK(x))
        f = sns.lmplot(
            data=data, x=C_INTERVAL, y=NOISE_FEATURE, row=C_FREQUENCY,
            hue=C_BINNED,
            legend=True, x_estimator=np.mean, aspect=aspect, n_boot=n_boot)
        sns.despine(offset=offset, trim=True)
        f.add_legend()
        plt.tight_layout()
        f.savefig(figure_save_template.format(f"{name} binned regression"))
        plt.clf()

        f = sns.lmplot(
            data=data, x=C_INTERVAL, y=NOISE_FEATURE, row=C_FREQUENCY,
            col=C_BINNED, aspect=1, n_boot=n_boot)
        f.map_dataframe(ANNOTATE)
        sns.despine(offset=offset, trim=True)
        f.add_legend()
        plt.tight_layout()
        f.savefig(figure_save_template.format(f"{name} regression values"))
        plt.clf()

        print("feature trajectories graphed over time")

    if graph_group_correlations:
        name = "cross correlation vs elapsed week"
        axis = "max pearson's R"
        data = graph_data[name].rename(columns={"value": axis}).groupby(
            [C_INTERVAL, C_ELECTRODE, C_FREQUENCY])[[axis]].mean().reset_index(
            drop=False)
        f = sns.relplot(
            data=data, x=C_INTERVAL, y=axis, hue=C_ELECTRODE, row=C_FREQUENCY,
            legend='brief', kind='scatter', aspect=aspect)  #, n_boot=n_boot)
        sns.despine(offset=offset, trim=True)
        f.add_legend()
        plt.tight_layout()
        f.savefig(figure_save_template.format(f"{name}"))
        plt.clf()

        f = sns.lmplot(
            data=data, x=C_INTERVAL, y=axis, row=C_FREQUENCY,
            legend=True, x_estimator=np.mean, aspect=aspect, n_boot=n_boot)
        f.map_dataframe(ANNOTATE)
        sns.despine(offset=offset, trim=True)
        f.add_legend()
        plt.tight_layout()
        f.savefig(figure_save_template.format(f"{name} total regression"))
        plt.clf()

        data[C_BINNED] = data[C_INTERVAL].apply(lambda x: WEEK_TO_CHUNK(x))
        f = sns.lmplot(
            data=data, x=C_INTERVAL, y=axis, row=C_FREQUENCY, hue=C_BINNED,
            legend=True, x_estimator=np.mean, aspect=aspect, n_boot=n_boot)
        sns.despine(offset=offset, trim=True)
        f.add_legend()
        plt.tight_layout()
        f.savefig(figure_save_template.format(f"{name} binned regression"))
        plt.clf()

        f = sns.lmplot(
            data=data, x=C_INTERVAL, y=axis, row=C_FREQUENCY, col=C_BINNED,
            aspect=1, n_boot=n_boot)
        f.map_dataframe(ANNOTATE)
        sns.despine(offset=offset, trim=True)
        f.add_legend()
        plt.tight_layout()
        f.savefig(figure_save_template.format(f"{name} regression values"))
        plt.clf()

        # ax = sns.boxplot(
        #     data, x=C_BINNED, y=axis, hue=C_FREQUENCY,
        #     width=0.5)
        # sns.despine(offset=offset, trim=True)
        # f = ax.get_figure()
        # plt.ylim(0.1, 0.3)
        # plt.tight_layout()
        # f.savefig(figure_save_template.format(f"{name} box plot"), dpi=100)
        # plt.clf()

    # if graph_group_separability:
    #     name = "cosine separability vs elapsed week"
    #     axis = "cosine separability"
    #     data = graph_data[name].rename(columns={"value": axis})
    #     data = data.groupby([C_INTERVAL, C_GROUND, C_FREQUENCY])[
    #         [axis]].mean().reset_index(drop=False)
    #     f = sns.relplot(
    #         data=data, x=C_INTERVAL, y=axis, hue=C_GROUND, row=C_FREQUENCY,
    #         legend='brief', kind='scatter', aspect=aspect)  # , n_boot=n_boot)
    #     sns.despine(offset=offset, trim=True)
    #     f.add_legend()
    #     plt.tight_layout()
    #     f.savefig(figure_save_template.format(f"{name}"))
    #     plt.clf()

    if graph_search_grid:
        for s in [F_LABEL] + F_LABEL:
            plt.clf()
            name = f"classification vs feature function {s}"
            save = figure_save_template.format(name)
            data = graph_data[name]
            ax = sns.heatmap(
                data=data, cbar=True, vmin=0, vmax=0.5)
            f = ax.get_figure()
            ax.set_aspect(data.shape[1]/data.shape[0])
            plt.tight_layout()
            f.savefig(save, dpi=200)
            plt.clf()

        print("grid search graphed")

    # if graph_encode:
    #     c_name = "components"
    #     v_name = "variance ratio"
    #     d_value = "value"
    #     v_value = "variance ratio"
    #     name = False  # f"{DIM_ENCODE} {TOP_FEATURE} " + "{}"
    #     subname = name.format(c_name)
    #     save = figure_save_template.format(subname)
    #     data = graph_data[subname].iloc[:, :4].melt(
    #         id_vars=[C_INTERVAL, C_GROUND], var_name=c_name,
    #         value_name=d_value)
    #     f = sns.relplot(
    #         data=data, x=C_INTERVAL, y=d_value, hue=C_GROUND, row=c_name,
    #         legend='brief', kind='line', aspect=aspect, n_boot=n_boot)
    #     sns.despine(offset=offset, trim=True)
    #     f.add_legend()
    #     plt.tight_layout()
    #     f.savefig(save)
    #     plt.clf()
    #
    #     subname = name.format(v_name)
    #     save = figure_save_template.format(subname)
    #     data = graph_data[subname]
    #     data[v_value] = data[v_value].cumsum()
    #     ax = sns.lineplot(data=data, x=c_name, y=v_value, n_boot=0)
    #     sns.despine(offset=offset, trim=True)
    #     # ax.set_aspect(0.5)
    #     f = ax.get_figure()
    #     plt.tight_layout()
    #     f.savefig(save, dpi=400)
    #     plt.clf()
    #     print("dim reduction graphed")

    if graph_shallow_curve:
        name = f"{TOP_MODEL} {TOP_FEATURE} learning curve"
        matrices = np.stack(
            [graph_data[f"{name} forward"][0],
             graph_data[f"{name} reverse"][0]], axis=0)

        holder = []
        for m in range(2):
            data = np.sum(matrices[m][-1, :, -1].copy(), axis=0)
            print(m, np.sum(data.diagonal()) / np.sum(data))
            data = data / np.sum(data, axis=-1)[..., None]
            holder += [data.copy()]
            order = np.argsort(-data.diagonal())
            data = pd.DataFrame(
                data, index=NATO_CODE_WORDS, columns=NATO_CODE_WORDS)
            ax = sns.heatmap(
                data=data.iloc[order, order], cbar=True, vmin=0, vmax=.7,
                xticklabels=1, yticklabels=1)
            f = ax.get_figure()
            ax.set_aspect(data.shape[1]/data.shape[0])
            plt.tight_layout()
            f.savefig(figure_save_template.format(f"{name} {m} conf"), dpi=200)
            plt.clf()

        data = holder[0] - holder[1]
        order = np.argsort(-data.diagonal())
        data = pd.DataFrame(
            data, index=NATO_CODE_WORDS, columns=NATO_CODE_WORDS)
        ax = sns.heatmap(
            data=data.iloc[order, order], cbar=True, vmin=-.5, vmax=.5,
            xticklabels=1, yticklabels=1, cmap="vlag")
        f = ax.get_figure()
        ax.set_aspect(data.shape[1] / data.shape[0])
        plt.tight_layout()
        f.savefig(figure_save_template.format(f"{name} conf diff"), dpi=200)
        plt.clf()

        mask = np.zeros((26, 26))
        np.fill_diagonal(mask, 1)
        mask = mask.astype(bool)
        matrices = np.sum(matrices * mask, axis=(-2, -1)) / np.sum(
            matrices, axis=(-2, -1))
        matrices = [["forward", matrices[0]], ["reverse", matrices[1]]]
        matrices = pd.DataFrame(matrices, columns=["direction", "accuracy"])
        matrices = matrices.assign(
            **{"block count": [np.arange(10, 121, 5)] * matrices.shape[0]})
        matrices = matrices.explode(["block count", "accuracy"])
        matrices = matrices.assign(
            **{"phase": [["train", "validation", "test"]] * matrices.shape[0]})
        matrices = matrices.explode("accuracy").explode(["phase", "accuracy"])
        matrices = matrices.loc[matrices["phase"].isin(
            ["test", "validation"])]

        f = sns.relplot(
            data=matrices, x="block count", y="accuracy", hue="phase",
            style="direction", legend='full', kind='line', aspect=aspect,
            n_boot=n_boot)
        sns.despine(offset=offset, trim=True)
        f.add_legend()
        plt.tight_layout()
        f.savefig(figure_save_template.format(f"{name} learning curves"))
        plt.clf()

        matrices = matrices[["block count", "accuracy", "direction"]]
        matrices = pd.concat(
            [matrices.loc[matrices["direction"] == "forward"].reset_index(
                 ).rename(columns={"accuracy": "forward test accuracy"}),
             matrices.loc[matrices["direction"] == "reverse"].reset_index(
                 ).rename(columns={"accuracy": "reverse test accuracy"})],
            axis=1)

        matrices = matrices[
            ["forward test accuracy", "reverse test accuracy"]].astype(float)
        f = sns.lmplot(
            data=matrices, x="forward test accuracy",
            y="reverse test accuracy", aspect=aspect, n_boot=n_boot)
        f.map_dataframe(ANNOTATE)
        sns.despine(offset=offset, trim=True)
        f.add_legend()
        plt.tight_layout()
        f.savefig(figure_save_template.format(f"{name} regression"))
        plt.clf()

    if graph_deep_curve:
        name = "RNN deep learning curve"
        forward = graph_data[f"{name} forward"]
        reverse = graph_data[f"{name} reverse"]
        print(forward.columns)

        holder = []
        for m, data in enumerate([forward, reverse]):
            data = data.copy().loc[
                (data["mode"] == "test") & (data["epoch"] == 49)]
            data = data.loc[data["iteration"] == 120]
            data = confusion_matrix(data.iloc[0, 3], data.iloc[0, 4])
            print(m, np.sum(data.diagonal()) / np.sum(data))
            data = data / np.sum(data, axis=-1)[..., None]
            holder += [data.copy()]
            order = np.argsort(-data.diagonal())
            data = pd.DataFrame(
                data, index=NATO_CODE_WORDS, columns=NATO_CODE_WORDS)
            ax = sns.heatmap(
                data=data.iloc[order, order], cbar=True, vmin=0, vmax=.7,
                xticklabels=1, yticklabels=1)
            f = ax.get_figure()
            ax.set_aspect(data.shape[1]/data.shape[0])
            plt.tight_layout()
            f.savefig(figure_save_template.format(f"{name} {m} conf"), dpi=200)
            plt.clf()

        data = holder[0] - holder[1]
        order = np.argsort(-data.diagonal())
        data = pd.DataFrame(
            data, index=NATO_CODE_WORDS, columns=NATO_CODE_WORDS)
        ax = sns.heatmap(
            data=data.iloc[order, order], cbar=True, vmin=-.5, vmax=.5,
            xticklabels=1, yticklabels=1, cmap="vlag")
        f = ax.get_figure()
        ax.set_aspect(data.shape[1] / data.shape[0])
        plt.tight_layout()
        f.savefig(figure_save_template.format(f"{name} conf diff"), dpi=200)
        plt.clf()

        forward["direction"] = "forward"
        reverse["direction"] = "reverse"
        matrices = pd.concat([forward, reverse], axis=0).rename(
            columns={"mode": "phase", "iteration": "block count"})
        matrices = matrices.loc[matrices["epoch"] == 49]
        matrices = matrices.loc[matrices["phase"].isin(
            ["test", "validation"])]
        matrices["accuracy"] = matrices.apply(
            lambda x: accuracy_score(x["label"], x["prediction"]), axis=1)

        f = sns.relplot(
            data=matrices, x="block count", y="accuracy", style="direction",
            hue="phase", legend='full', kind='line', aspect=aspect,
            n_boot=n_boot)
        sns.despine(offset=offset, trim=True)
        f.add_legend()
        plt.tight_layout()
        f.savefig(figure_save_template.format(f"{name} learning curves"))
        plt.clf()

        matrices = matrices[["block count", "accuracy", "direction"]]
        matrices = pd.concat(
            [matrices.loc[matrices["direction"] == "forward"].reset_index(
                 ).rename(columns={"accuracy": "forward test accuracy"}),
             matrices.loc[matrices["direction"] == "reverse"].reset_index(
                 ).rename(columns={"accuracy": "reverse test accuracy"})],
            axis=1)
        matrices = matrices[
            ["forward test accuracy", "reverse test accuracy"]].astype(float)
        f = sns.lmplot(
            data=matrices, x="forward test accuracy",
            y="reverse test accuracy", aspect=aspect, n_boot=n_boot)
        f.map_dataframe(ANNOTATE)
        sns.despine(offset=offset, trim=True)
        f.add_legend()
        plt.tight_layout()
        f.savefig(figure_save_template.format(f"{name} regression"))
        plt.clf()


if __name__ == "__main__":
    root_dir = "/Users/ike/Documents/Lab/Chang Lab"
    proj_dir = "/Users/ike/Desktop/Chang Lab"

    pickle_partial = f"{root_dir}/Data/df_ike.pkl"
    pickle_total = f"{root_dir}/Data/all_nato_data.pkl"

    # dict_save = f"{proj_dir}/graph_data.pkl"
    # dict_save = f"{proj_dir}/graph_data_grid.pkl"
    dict_save = f"{proj_dir}/updated_graph_data.pkl"
    figure_path = f"{proj_dir}/Figures/" + "{}.png"

    # raw_time_series_dataset = data_loading_pipeline(
    #     # raw_data_path=pickle_partial,
    #     # raw_data_key=None,
    #     raw_data_path=pickle_total,
    #     raw_data_key="df",
    #     dict_save_path=dict_save,
    #     count_blocks=False,
    # )
    # raw_analysis_pipeline(
    #     time_data=raw_time_series_dataset,
    #     dict_save_path=dict_save,
    #     find_peak_window=True,
    #     extract_snr=False,
    #     correlate_groups=False,
    # )
    # individual_trial_pipeline(
    #     time_data=raw_time_series_dataset,
    #     dict_save_path=dict_save,
    #     data_save_dir=proj_dir,
    #     search_grid=False,
    #     do_dim_reduction=False,
    #     shallow_curve=False,
    #     deep_curve=False
    # )
    generate_plots(
        dict_save_path=dict_save,
        figure_save_template=figure_path,
        graph_block_counts=False,
        graph_peak_window=False,
        graph_extracted_snr=False,
        graph_group_correlations=False,
        graph_search_grid=False,
        graph_encode=False,
        graph_shallow_curve=False,
        graph_deep_curve=True
    )

    # data = load_analyzed_data(dict_save)
    # print([k for k in data])
    # print(type(data['LDA a.u.c learning curve forward']))
    # # print(data.columns)
    # # print(data.shape)
