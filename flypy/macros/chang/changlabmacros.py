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


def pipeline(
        raw_data_path,
        raw_data_key,
        dict_save_path,
        count_blocks=False,
        correlate_groups=False,
        find_similarity=False,
        search_grid=False,
        calculate_salience=False,
        do_pca=False
):
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
        cols_meta=[C_TIME, C_INTERVAL, C_EDAY, C_BLOCK, C_GROUND, C_NATO],
        timepoints=np.arange(START, STOP, 1 / HZ),
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
        time_data_signal = time_data.clip_traces(
            np.arange(START_S * HZ, STOP_S * HZ, dtype=int))
        within, across, separability = correlation_over_time(
            data=time_data_signal,
            refs=time_data_signal.filter(C_INTERVAL, "<=", REF_INTERVAL),
            col_label=C_GROUND,
            col_group=C_INTERVAL,
            cols_source=[C_FREQUENCY, C_ELECTRODE],
            f_trials=np.median
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
        name = "cosine similarity {} vs elapsed day"
        w_name = "within"
        a_name = "across"
        s_name = "separability"
        time_data_signal = time_data.clip_traces(
            np.arange(START_S * HZ, STOP_S * HZ, dtype=int))
        within, across, separability = cosine_similarity_over_time(
            data=time_data_signal,
            refs=time_data_signal.filter(C_INTERVAL, "<=", REF_INTERVAL),
            f_feature=FEATURES[TOP_FEATURE],
            col_label=C_GROUND,
            col_group=C_INTERVAL,
            cols_source=[C_FREQUENCY, C_ELECTRODE],
            f_trials=np.median
        )
        graph_data[name.format(w_name)] = within
        graph_data[name.format(a_name)] = across
        graph_data[name.format(s_name)] = separability
        save_analyzed_data(dict_save_path, graph_data)
        print("data similarities computed by group over time")

    if search_grid:
        """
        Perform a grid search across a set of feature extraction functions and
        low-level ML models to identify the (model, feature) pair that
        maximizes classification accuracy.
        """
        for s in [F_LABEL] + F_LABEL:
            name = f"classification vs feature function {s}"
            accuracy, model, feature = classification_grid_search(
                data=time_data.filter(C_INTERVAL, "<=", REF_INTERVAL)[
                    {C_FREQUENCY: s}].clip_traces(
                    np.arange(START_S * HZ, STOP_S * HZ, dtype=int)),
                func_dict=FEATURES,
                model_dict=CLASSIFIERS,
                cols_source=[C_BLOCK, C_GROUND],
                col_label=C_GROUND,
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
            data=time_data[{C_FREQUENCY: TOP_FREQUENCIES}],
            f_feature=FEATURES[TOP_FEATURE],
            col_label=C_GROUND,
            cols_source=[C_FREQUENCY, C_ELECTRODE],
            col_group=C_INTERVAL,
            f_trials=np.median,
            f_labels=np.mean,
            idx_signal=np.arange(START_S * HZ, STOP_S * HZ, dtype=int)
        )
        graph_data[name.format(c_name)] = center
        graph_data[name.format(v_name)] = cvs
        save_analyzed_data(dict_save_path, graph_data)
        print("salient sources extracted for top features")

    if do_pca:
        """
        Perform PCA on extracted feature vectors and plot procession in first
        two principal components over time.
        """
        name = f"pca {TOP_FEATURE} " + "{}"
        c_name = "components"
        v_name = "cumulative variance"
        components, variance = encode_with_dim_reducer(
            data=time_data[{C_FREQUENCY: TOP_FREQUENCIES}],
            f_feature=FEATURES[TOP_FEATURE],
            col_label=C_GROUND,
            cols_source=[C_FREQUENCY, C_ELECTRODE],
            col_group=C_INTERVAL,
            f_trials=np.median,
            f_labels=np.mean,
            idx_signal=np.arange(START_S * HZ, STOP_S * HZ, dtype=int)
        )
        graph_data[name.format(c_name)] = center
        graph_data[name.format(v_name)] = cvs
        save_analyzed_data(dict_save_path, graph_data)
        print("salient sources extracted for top features")

    if False:
        pass
        """
        """
        # for f, df in outputs.items():
        #     save = figure_dir.format(f"auc median mean model agnostic {f}")
        #     df = df.sort_values(df.columns[0], ascending = False)
        #     ax = sns.heatmap(data=df, cbar=True) #vmin=0, vmax=2)
        #     f = ax.get_figure()
        #     plt.tight_layout()
        #     f.savefig(save, dpi=400)
        #     plt.clf()

        # save = figure_dir.format(
        #     f"average aggregate electrode salience by feature")
        # tots = np.mean(np.stack(tots, axis=0), axis=0)
        # tots = tots.reshape(*G_SHAPE)
        # tots = pd.DataFrame(
        #     data=tots, index=np.arange(G_SHAPE[0]), columns=np.arange(G_SHAPE[1]))
        # tots.index.name = "grid_y"
        # tots.columns.name = "grid_x"
        # ax = sns.heatmap(data=tots, vmin=0, vmax=1, cbar=True)
        # f = ax.get_figure()
        # plt.tight_layout()
        # f.savefig(save, dpi=400)
        # plt.clf()
        #
        # # save data for graphing
        # name = "Trace Signal to Noise Ratio"
        # snrs = time_data.grouped_snr(CS_SNR).apply_scalar_function(
        #     col_snr, np.mean).meta
        # graph_data[name] = snrs[[col_snr]].reset_index()


        """
        Split vector dataset into train and test sets by cumulative day. Fit
        unsupervised encoders on data prior to freeze day and identify encoder
        that maximizes separability between dissimilar labels after freeze day.
        """
        # # split into train set up to freeze_day cumulative day, all else for test
        # train_split, valid_split = vector_data.boolean_split(
        #     vector_data.labels[..., IDX_G] <= day_freeze)
        #
        # # find encoding paradigm that maximizes separability between train labels
        # encoders = {
        #     k: train_split.fit_model(i[0](**(i + [{}])[1]))
        #     for k, i in unsupervised.items()}
        # separability = {
        #     k: train_split.encode(i).pairwise_cos(IDX_L)[-1]
        #     for k, i in encoders.items()}
        #
        # # save cosine similarity separability data for graphing
        # name = "Unsupervised Encoder Train Cluster Seperability"
        # graph_data[name] = pd.Series(data=separability)
        #
        # # use model with peak separability to transform all vector data
        # max_key = max(separability, key=separability.get)
        # vector_data = vector_data.encode(encoders[max_key])
        # train_split = train_split.encode(encoders[max_key])
        # valid_split = valid_split.encode(encoders[max_key])

        """
        Train classification models on transformed data prior to freeze day, test
        on all data afterwards.
        """
        # r_train = []
        # for key, classifier in supervised.items():
        #     classifier = classifier + [{}]
        #
        #     # train k-fold models
        #     models, train_df = train_k_classifiers(
        #         train_split, classifier[0], k_fold, IDX_L, IDX_G, **classifier[1])
        #     train_df = train_df.assign(**{"classifier": key})
        #
        #     # use trained models for conference of experts testing
        #     test_df = test_k_classifiers(valid_split, models, IDX_L, IDX_G).assign(
        #         **{"classifier": key})
        #     r_train += [train_df, test_df]
        #
        # # save training data for graphing
        # name = "Supervised Classifier Frozen Accuracies"
        # graph_data[name] = pd.concat(r_train, ignore_index=True, sort=True)

        """
        Generate learning curves per classification model.
        """
        # r_curve = []
        # for key, classifier in supervised.items():
        #     classifier = classifier + [{}]
        #
        #     # train classifiers on data from days < n, test on data from nth day
        #     curve_df = train_test_frozen_classifier(
        #         vector_data, classifier[0], IDX_L, IDX_G, start, **classifier[1])
        #     r_curve += [curve_df.assign(**{"classifier": key})]
        #
        # # save learning curve data for graphing
        # name = "Supervised Classifier Leave-One-Out Learning Curves"
        # graph_data[name] = pd.concat(r_curve, ignore_index=True, sort=True)


        """
        TODO: write code to plot accuracy_score as a function of input time range, plot
        electrode contributions
        """

        """
        Take trained model and determine accuracy_score as a function of input timepoints
        """
        # r_accuracy = []
        # r_electrode_contributions = []
        # for idx in range(1, len(T_AXIS)):
        #     time_data = time_data.apply_scalar_function(
        #         col_feature, root_mean_square, indices=slice(0, idx))
        #     valid_split = FeatureVector(
        #         *time_data.get_vector_form(
        #             cols_group=cols_group, cols_sort=cols_sort,
        #             cols_label=cols_labels, col_f=col_feature))
        #     valid_split, _ = valid_split.boolean_split(
        #         valid_split.labels[:, -1] > day_freeze)
        #     r_accuracy += [run_inference(valid_split, models, 0)[-1]]
        #     r_electrode_contributions += [feature_contributions(
        #         valid_split, models, N_ELECTRODES, idx)]

    return


def generate_plots(
        dict_save_path,
        figure_save_template,
        graph_block_counts=False,
        graph_group_correlations=True,
        graph_group_similarities=True,
        graph_search_grid=True,
        graph_source_salience=True,
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

    if graph_source_salience:
        C_INTERVAL, C_FREQUENCY, C_ELECTRODE, "value"
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


if __name__ == "__main__":
    pickle_partial = "/Users/ike/Documents/Lab/Chang Lab/Data/df_ike.pkl"
    pickle_total = "/Users/ike/Documents/Lab/Chang Lab/Data/all_nato_data.pkl"

    dict_save = "/Users/ike/Desktop/Chang Lab/graph_data.pkl"
    figure_path = "/Users/ike/Desktop/Chang Lab/Figures/{}.png"

    pipeline(
        raw_data_path=pickle_total,
        raw_data_key="df",
        dict_save_path=dict_save,
        count_blocks=False,
        correlate_groups=True,
        find_similarity=True,
        search_grid=True,
        calculate_salience=True
    )

    generate_plots(
        dict_save_path=dict_save,
        figure_save_template=figure_path,
        graph_block_counts=True,
        graph_group_correlations=True,
        graph_group_similarities=True,
        graph_search_grid=False,
        graph_source_salience=False
    )


    # data = load_trial_data(
    #     pickle_tot, CS_PICKLE, CS_EXPLODE, "df").reset_index()
    # data[C_TIME] = date_to_cumulative_day(data[C_TIME], D_SLICE, D_FORMAT)
    # temp = np.unique(data[C_TIME]) // 7
    # print(np.unique(temp, return_counts=True))
    # data = np.unique(data[IDENTIFIER])

    # full_list = np.unique(data[IDENTIFIER].values.flatten())
    # full_list = full_list[full_list < 1000]
    # encoder = {v: k for k, v in enumerate(full_list)}
    # data = data.loc[data[IDENTIFIER].isin(full_list)]
    # data["TEMP"] = np.vectorize(lambda x: encoder[x])(data[IDENTIFIER])
    # print(encoder)
    # print(data[[IDENTIFIER, "TEMP"]].head())

    # tmd = taskData.TaskMetadata()
    # txt_labs = [tmd.loadUtterance(e, remove_descriptors=True) for e in
    #             sorted(list(set(full_list)))]
    # data[C_TIME] = date_to_cumulative_day(data[C_TIME], D_SLICE, D_FORMAT)
    # data = data.rename(columns={C_TIME: C_DAY})
    # time_data = pickle_data_to_time_dataset(
    #     data, C_TRACE, CS_METADATA, EXPAND, N_ELECTRODES, T_AXIS)
    # cols = [C_DAY, IDENTIFIER]
    # counts = time_data.get_subgroup_counts(cols).reset_index()
    # counts = counts.groupby(level=cols).size()
    # print(counts)
