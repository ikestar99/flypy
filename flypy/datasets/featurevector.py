#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 02:45:05 2024

@author: ike
"""

import numpy as np
import pandas as pd
import sklearn.preprocessing as spp
import sklearn.model_selection as sms
import sklearn.metrics.pairwise as smp

from flypy.datasets.coredataset import CoreDataset


class FeatureVector(CoreDataset):
    """
    Class to organize, filter, and manipulate vectors and paired metadata.
    Inherits from CoreDataset class.

    ---------------------------------------------------------------------------
    Attributes:
        _TRIAL (str):
            Internal metadata level to distinguish between repeated trials.
        _INDEX (str):
            Internal metadata level to track corresponding index of each datum.
        _OTHER (str):
            Internal suffix used for comparisons against a reference.
        _SCALER (sklearn.preprocessing._data.StandardScaler):
            Used to scale vector features to zero mean and unit variance.
        _SHUFFLER (numpy.random._generator.Generator):
            Used to shuffle metadata-vector pairings.
        data (np.ndarray):
            Stored data.
        meta (pd.DataFrame):
            Hierarchically indexed DataFrame storing metadata.
        axes (int):
            Number of axes per datum.

    ---------------------------------------------------------------------------
    Examples:
        Instantiate using "expand" arg.
        >>> # test_data shape = (8 vectors, 5 features)
        >>> test_data = np.arange(40).reshape(8, 5)
        >>> # test_meta shape = (8 vectors, 3 metadata variables)
        >>> test_meta = pd.DataFrame(
        ... {"group": ["a", "a", "a", "b", "b", "c", "c", "d"],
        ... "label": [1, 1, 2, 2, 3, 3, 4, 4], "type": "fast"})
        >>> test_dataset = FeatureVector(
        ... test_data, test_meta)
        >>> test_dataset.data
        array([[ 0,  1,  2,  3,  4],
               [ 5,  6,  7,  8,  9],
               [10, 11, 12, 13, 14],
               [15, 16, 17, 18, 19],
               [20, 21, 22, 23, 24],
               [25, 26, 27, 28, 29],
               [30, 31, 32, 33, 34],
               [35, 36, 37, 38, 39]])
        >>> test_dataset.meta
                                          _data_index
        group label type _repeated_trial
        a     1     fast 0                          0
                         1                          1
              2     fast 2                          2
        b     2     fast 3                          3
              3     fast 4                          4
        c     3     fast 5                          5
              4     fast 6                          6
        d     4     fast 7                          7

        -----------------------------------------------------------------------
        Scale features to zero mean and unit variance.
        >>> test_scale = test_dataset.scale()
        >>> np.around(test_scale.data, 3)
        array([[-1.528, -1.528, -1.528, -1.528, -1.528],
               [-1.091, -1.091, -1.091, -1.091, -1.091],
               [-0.655, -0.655, -0.655, -0.655, -0.655],
               [-0.218, -0.218, -0.218, -0.218, -0.218],
               [ 0.218,  0.218,  0.218,  0.218,  0.218],
               [ 0.655,  0.655,  0.655,  0.655,  0.655],
               [ 1.091,  1.091,  1.091,  1.091,  1.091],
               [ 1.528,  1.528,  1.528,  1.528,  1.528]])

        -----------------------------------------------------------------------
        Shuffle metadata-vector pairings.
        >>> test_shuffle = test_dataset.shuffle()
        >>> test_shuffle.data
        array([[30, 31, 32, 33, 34],
               [ 5,  6,  7,  8,  9],
               [25, 26, 27, 28, 29],
               [ 0,  1,  2,  3,  4],
               [20, 21, 22, 23, 24],
               [35, 36, 37, 38, 39],
               [10, 11, 12, 13, 14],
               [15, 16, 17, 18, 19]])
        >>> test_shuffle.meta
                                          _data_index
        group label type _repeated_trial
        a     1     fast 0                          0
                         1                          1
              2     fast 2                          2
        b     2     fast 3                          3
              3     fast 4                          4
        c     3     fast 5                          5
              4     fast 6                          6
        d     4     fast 7                          7

        -----------------------------------------------------------------------
        Split instance in two within labels.
        >>> test_split_1, test_split_2 = test_dataset.stratified_split(
        ... 0.5, "label")
        >>> test_split_1.data
        array([[10, 11, 12, 13, 14],
               [25, 26, 27, 28, 29],
               [ 5,  6,  7,  8,  9],
               [35, 36, 37, 38, 39]])
        >>> test_split_1.meta
                                          _data_index
        group label type _repeated_trial
        a     1     fast 1                          2
              2     fast 2                          0
        c     3     fast 5                          1
        d     4     fast 7                          3
        >>> test_split_2.data
        array([[15, 16, 17, 18, 19],
               [30, 31, 32, 33, 34],
               [ 0,  1,  2,  3,  4],
               [20, 21, 22, 23, 24]])
        >>> test_split_2.meta
                                          _data_index
        group label type _repeated_trial
        a     1     fast 0                          2
        b     2     fast 3                          0
              3     fast 4                          3
        c     4     fast 6                          1
    """
    _SCALER = spp.StandardScaler()
    _SHUFFLER = np.random.default_rng()

    def __init__(
            self,
            data: np.ndarray,
            meta: pd.DataFrame,
            expand: list = None
    ):
        """
        Instantiate TimeSeries instance.
        See CoreDataset.__init__ docstring for more detailed description.

        NOTE: data.ndim = 1 (N vectors) + 1 (1d vector) + len(expand).

        Args:
            data (np.ndarray):
                Array of feature vectors for analysis. Must be at least 2d.
                data.shape = (N vectors, ..., N features, ...).
            meta (pd.DataFrame):
                Metadata corresponding to first axis of data arg.
                meta.iloc[i, :] = set of labels for data[i].
            expand (list, optional):
                Mandatory if data.ndim > 2. len(expand) = traces.ndim - 2.
                See CoreDataset.__init__ docstring
        """
        # set instance attributes
        super(FeatureVector, self).__init__(data, meta, axes=1, expand=expand)

    def scale(
            self
    ):
        """
        Scale vector features to zero mean and unit variance.

        Returns:
            (FeatureVector)
                New instance with scaled vector features.
        """
        return FeatureVector(self._SCALER.fit_transform(self.data), self.meta)

    def shuffle(
            self
    ):
        """
        Shuffle metadata-vector pairings.

        Returns:
            (FeatureVector):
                New instance with shuffled labels.
        """
        idx_shuffled = np.arange(len(self))
        self._SHUFFLER.shuffle(idx_shuffled)
        return FeatureVector(self.data[idx_shuffled], self.meta)

    def boolean_split(
            self,
            mask: np.ndarray
    ):
        """
        Split instance into two new instances according to boolean mask.

        Args:
            mask (np.ndarray):
                1d boolean mask. mask.size = len(self)

        TODO: update docstring
        Returns:
            (FeatureVector):
                New instance with vectors labeled True in mask arg.
            (FeatureVector):
                New instance with vectors labeled False in mask arg.
        """
        return self[mask], self[np.logical_not(mask)]

    def group_split(
            self,
            group: str
    ):
        """
        Generate non-overlapping splits of the vector dataset by group.
        Resulting splits contain vectors from a single group only and represent
        the full dataset in aggregate.

        Args:
            group (str):
                Metadata level name to interpret as group identity.

        Yields:
            group (scalar):
                Group label of current iteration.
            (FeatureVector):
                New instance with vectors in specified group.
        """
        # stratify vector data by group
        groups = self[group]
        for group in np.unique(groups):
            yield group, self[groups == group]

    def stratified_split(
            self,
            split: float,
            label: str
    ):
        """
        Split instance into two new instances with fixed data ratio. Split
        maintains the label distribution to the extent possible.

        Args:
            split (float):
                Ratio by which to split instance. 0 < split < 1.
            label (str):
                Metadata level name used as label identity to stratify.

        Returns:
            (FeatureVector):
                New instance with first split.
            (FeatureVector):
                New instance with second split.
        """
        # track indices of splits within labels
        idx_0, idx_1 = sms.train_test_split(
            np.arange(len(self)), train_size=split, stratify=self[label])

        # create new instances with split datasets
        return self[idx_0], self[idx_1]

    def k_fold_split(
            self,
            k: int,
            label: str
    ):
        """
        Generate a k-fold split of the vectors stratified by labels.

        Args:
            k (int):
                Number of folds to generate.
            label (str):
                Metadata level name used to stratify such that each split
                includes a similar distribution of labels across folds.

        Yields:
            (FeatureVector):
                New instance with current fold train split.
            (FeatureVector):
                New instance with current fold validation split.
        """
        splitter = sms.StratifiedKFold(n_splits=k)
        for idx_t, idx_v in splitter.split(np.arange(len(self)), self[label]):
            # return current subset, generate next one when needed
            yield self[idx_t], self[idx_v]

    def leave_last_out_split(
            self,
            group: str,
            start: int = 1,
            reverse: bool = False
    ):
        """
        Generate a leave-one-out split of the vector dataset by group.
        Resulting splits are staggered such that the ith split has a training
        set of all samples in groups j < i and a test set of all samples in
        group i. The final split is the only that includes the full dataset.

        WARNING: leave_last_out_split will not check that a given train split
        represents all possible labels.

        Args:
            group (str):
                Metadata level name used as group identity.
            start (int, optional):
                Index of first unique group to use as test set.
                Defaults to "1", in which case the first split includes a train
                set of group 1 and a test set of group 2.
            reverse (bool, optional):
                If True, sort groups in descending order before splitting.
                Defaults to "False", in which case groups are sorted in
                ascending order.

        Yields:
            group (scalar):
                Validation set group label for current iteration.
            (FeatureVector):
                New instance with current iteration train data.
            (FeatureVector):
                New instance with current iteration validation data.
        """
        # get unique group labels
        groups = self[group]
        unique = np.unique(groups)
        unique = unique[::-1] if reverse else unique

        # iterate over groups, yield current split
        for i, group in enumerate(unique[start:]):
            idx_t = np.in1d(groups, unique[:start + i])
            idx_v = groups == group
            yield group, self[idx_t], self[idx_v]

    def cont_cos_similarity(
            self,
            cont,
            label: str,
            group: str,
            ignore: list = None,
    ):
        """
        Compute pairwise cosine similarity between vectors in instance.

        Args:
            cont (FeatureVector, optional):
                Static instance against which to compute similarity.
            label (str):
                Metadata level name to use as label for each vector prior to
                computing similarity matrix. Must be present in both "self" and
                "cont".
            group (str):
                Metadata level name by which to group traces into subsets for
                comparison against "cont".
            ignore (list, optional):
                Metadata level names to ignore when identifying valid
                comparisons.
                Defaults to "None", in which case comparisons are made within
                all metadata levels outside of those specified by "label" and
                "group" args.

        Returns:
            (pd.DataFrame)
                meta_cos.shape = (N groups, N labels).
                meta_cos.loc[g, l] = [within-label similarities in group g]
                meta_cos.loc[g, -1] = [between-label similarities in group g]
        """
        def _cos_by_label(
                _v_s: np.ndarray,
                _l_s: np.ndarray,
                _v_c: np.ndarray,
                _l_c: np.ndarray,
        ):
            """
            Nested func. Compute cosine similarity between every combination of
            feature vectors in input arrays. Return list of similarities within
            each label and list of correlations between all labels.

            Args:
                _v_s (np.ndarray):
                    _vectors_0.shape = (N vectors_0, N features).
                _l_s (np.ndarray):
                    _labels_0.shape = (N vectors_0,).
                _v_c (np.ndarray):
                    _vectors_1.shape = (N vectors_1, N features).
                _l_c (np.ndarray):
                    _labels_1.shape = (N vectors_1,).

            Returns:
                cos (pd.Series):
                    cos.loc[l] = [cosine separability within label l].
            """
            cos = smp.cosine_similarity(_v_s, _v_c)
            if np.array_equal(_v_s, _v_c):
                np.fill_diagonal(cos, np.nan)

            unique = np.unique(_l_s[np.isin(_l_s, _l_c)])
            cos = [
                cos[(_l_s == i)[:, None] * (_l_c == i)[None]].flatten() -
                np.mean(cos[(_l_s == i)[:, None] * (_l_c != i)[None]])
                for i in unique]
            return pd.Series(cos, index=unique)

        # compute and return similarities
        return self.grouped_control(cont, _cos_by_label, label, group, ignore)

    def fit_model(
            self,
            model,
            label: str = None,
    ):
        """
        Train an encoding model on instance vector data.

        Args:
            model:
                Encoding model. Must define .fit method per sklearn convention.
                For an example, see:
                sklearn.decomposition.PCA.
            label (str, optional):
                Metadata level name to use as vector label for supervised
                models.
                Defaults to "None", in which case model is unsupervised.

        Returns:
            model:
                Trained model.
        """
        labels = None if label is None else self[label]
        return model.fit(self.data, labels)

    def transform_model(
            self,
            model
    ):
        """
        Use a trained encoding model to transform vector data.

        Args:
            model:
                Trained encoding model. Must define .transform method per
                sklearn convention. For an example, see:
                sklearn.decomposition.PCA.

        Returns:
            (FeatureVector):
                New instance with transformed data.
        """
        return FeatureVector(model.transform(self.data), self.meta)

    def predict_model(
            self,
            model,
    ):
        """
        Use a trained encoding model to predict vector labels.

        Args:
            model:
                Trained classification model. Must define .predict method per
                sklearn convention. For an example, see:
                sklearn.neighbors.KNeighborsClassifier.

        Returns:
            (np.ndarray):
                Predicted label for each vector stored in instance.
                shape = (self.data.shape[0],).
        """
        return model.predict(self.data)

    def predict_distribution(
            self,
            model,
    ):
        """
        Use a trained encoding model to predict vector class distributions.

        Args:
            model:
                Trained classification model. Must define .predict_proba method
                per sklearn convention. For an example, see:
                sklearn.neighbors.KNeighborsClassifier.

        Returns:
            (np.ndarray):
                Predicted class label probabilities for each vector stored in
                instance. Each row is a predicted probability distribution
                across all possible labels.
                len = (self.data.shape[0], N unique labels).
        """
        return model.predict_proba(self.data)
