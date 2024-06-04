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


class FeatureVector:
    """
    Class to pair feature vectors with a corresponding label.

    Feature vectors are stored in a 2D array while labels are stored in a 1D
    array.

    TODO: add shuffler
    Attributes:
        SHUFFLER (np.random._generator.Generator): Index shuffling generator.
        COS_GROUP (str): Name given to column of group value used when
            computing cosine similarities. Relevant for pairwise_cos_against_reference
            function return.
        COS_LABEL (str): Name given to column of vector label used when
            computing cosine similarities. Relevant for pairwise_cos_against_reference
            function return.
        COS_VALUE (str): Name given to column of cosine similarity values.
            Relevant for pairwise_cos_against_reference function return.
        vectors (np.ndarray): Feature vectors.
        n_features (int): Number of features in each vector.
        labels (np.ndarray): Corresponding labels for feature vectors.
        shuffle (np.ndarray): Array containing index order of features arg.
    """
    SHUFFLER = np.random.default_rng()
    SCALER = spp.StandardScaler()
    COS_GROUP = "group"
    COS_LABEL = "pair label"
    COS_VALUE = "cosine similarity"
    COS_DELTA = "cosine seperability"

    def __init__(
            self,
            vectors: np.ndarray,
            labels: np.ndarray,
            shuffle: np.ndarray = None
    ):
        """
        Instantiate FeatureVector dataset.

        Args:
            vectors (np.ndarray): Array of feature vectors for analysis. Must
                be 2D with shape = (N samples, N features).
            labels (np.ndarray): Array of labels corresponding to vectors arg.
                Must be 1D with shape = (N samples,) or 2D with shape =
                (N samples, N labels).
            shuffle (np.ndarray, optional): Mandatory when creating a new
                instance from a subset of an existing instance. Array of
                indices that relate the position of a given vector in feature
                arg and corresponding label in labels arg to their indices in
                an initial dataset. Used to update labels after shuffling.
        """
        # set instance attributes
        self.vectors = np.atleast_2d(vectors).copy()
        self.n_features = self.vectors.shape[-1]
        self.labels = np.atleast_2d(labels).copy()
        self.shuffle = np.atleast_1d(
            np.arange(vectors.shape[0]) if shuffle is None else shuffle).copy()

    def __len__(
            self
    ):
        """
        Returns:
            (int): Number of (feature vector, label) pairs stored.
        """
        return self.vectors.shape[0]

    def __add__(
            self,
            other
    ):
        assert type(other) == FeatureVector

        vectors = np.concatenate((self.vectors, other.vectors), axis=0)
        labels = np.concatenate((self.labels, other.labels), axis=0)
        shuffle = np.concatenate((self.shuffle, other.shuffle), axis=0)
        return FeatureVector(vectors, labels, shuffle)

    def __getitem__(
            self,
            idx: int
    ):
        """
        Filter slice of vectors and corresponding labels.

        Args:
            idx (int | slice): Indices of instance features and labels
                attributes to extract

        Returns:
            (FeatureVector): New instance filtered to include the data
                specified by idx arg.
        """
        return FeatureVector(
            self.vectors[idx], self.labels[idx], self.shuffle[idx])

    def scale(
            self
    ):
        return FeatureVector(
            self.SCALER.fit_transform(self.vectors), self.labels, self.shuffle)

    def set_labels(
            self,
            labels: np.ndarray,
            idx_l: int = 0
    ):
        """
        Change the labels describing instance vectors attribute.

        Note: set_labels assumes labels arg corresponds to original set of
        feature vectors used to construct instance, before any shuffle or split
        operations were performed.

        Args:
            labels (np.ndarray): 1D Array of labels with which to update
                instance labels attribute.
            idx_l (int, optional): Column index in which to insert new labels.
                Relevant for instance with multiple labels, in which case only
                labels[:, idx] will be updated. Defaults to 0.

        Returns:
            self (FeatureVectorDataset): Labels attribute updated at idx.

        """
        self.labels[:, idx_l] = labels[self.shuffle]
        return self

    def shuffle_between(
            self
    ):
        """
        Shuffle instance vectors and labels attributes. Shuffle procedure
        maintains pairing between individual feature vectors and their
        corresponding labels.

        Returns:
            self (FeatureVectorDataset): Vectors and labels attributes shuffled
                along first dimension.
        """
        idx_shuffled = np.arange(len(self))
        self.SHUFFLER.shuffle(idx_shuffled)
        self.vectors = self.vectors[idx_shuffled]
        self.labels = self.labels[idx_shuffled]
        self.shuffle = self.shuffle[idx_shuffled]
        return self

    def shuffle_within(
            self
    ):
        """
        Shuffle instance labels attribute. Shuffle procedure is relevant for
        negative control analysis.

        WARNING: Shuffle breaks concordance between features and labels. This
        cannot be undone.

        Returns:
            (FeatureVector): New instance with labels attribute shuffled
                along first dimension.
        """
        idx_shuffled = np.arange(len(self))
        self.SHUFFLER.shuffle(idx_shuffled)
        return FeatureVector(
            self.vectors, self.labels[idx_shuffled], self.shuffle)

    def undo_shuffle_between(
            self
    ):
        """
        Sort instance attributes in ascending index order. Sorting applies on
        indices in instance shuffle attribute and effectively undoes shuffle
        described by FeatureVectorDataset.shuffle_between function.

        Returns:
            self (FeatureVectorDataset): instance array attributes sorted in
                ascending index order.
        """
        sorted_idx = np.argsort(self.shuffle)
        self.vectors = self.vectors[sorted_idx]
        self.labels = self.labels[sorted_idx]
        self.shuffle = self.shuffle[sorted_idx]
        return self

    def boolean_split(
            self,
            mask: np.ndarray
    ):
        """
        Split instance into two new instances according to boolean mask. Make a
        mask with a boolean operation on a set of labels in instance labels
        attribute and create two new instances of all true val

        WARNING: Unlike with stratified split methods, boolean_split will not
        force even label representation in the output datasets. For downstream
        encoding tasks, ensure that dataset used for training is split in such
        a way that all labels are represented.

        Args:
            mask (np.ndarray): 1D boolean mask of the same length as instance.

        Returns:
            unit_t (FeatureVectorDataset): New instance with vectors labeled
                True in mask arg.
            unit_f (FeatureVectorDataset): New instance with vectors labeled
                False in mask arg.
        """
        unit_t = FeatureVector(
            self.vectors[mask], self.labels[mask], self.shuffle[mask])
        mask = np.logical_not(mask)
        unit_f = FeatureVector(
            self.vectors[mask], self.labels[mask], self.shuffle[mask])
        return unit_t, unit_f

    def group_split(
            self,
            idx_g: int
    ):
        """
        Generate non-overlapping splits of the vector dataset by group.
        Resulting splits contain vectors from a single group only and represent
        the full dataset in aggregate.

        Args:
            idx_g (int): Index in instance labels attribute used to split
                groups.

        Yields:
            group (scalar): Group label of current iteration.
            unit_g (FeatureVectorDataset): New instance where all vectors are
                members of group in group yield value.
        """
        # stratify vector data by group
        for group in np.unique(self.labels[:, idx_g]):
            mask = (self.labels[:, idx_g] == group)
            unit_g = FeatureVector(
                self.vectors[mask], self.labels[mask], self.shuffle[mask])
            yield group, unit_g

    def stratified_split(
            self,
            split: float,
            idx_l: int = 0
    ):
        """
        Split instance into two new instances with fixed data ratio. Split is
        performed within labels rather than overall, such that the two output
        FeatureVectorDatasets have roughly the same distribution of labels as
        the initial instance.

        Note: stratified_split is built to return two new instances. To create
        more than two subsets, call function multiple times such that each call
        returns a subset of the desired ratio and another subset that will be
        split further.

        Args:
            split (float): Ratio by which to split instance. 0 < split < 1.
            idx_l (int): Index in instance labels attribute used to stratify.
                Defaults to 0.

        Returns:
            unit_0 (FeatureVectorDataset): New instance with the first split
                fraction of vectors within each label specified by idx.
            unit_1 (FeatureVectorDataset): New instance with the last 1 - split
                fraction of vectors within each label specified by idx.
        """
        # track indices of splits within labels
        indices = np.arange(len(self))
        idx_0, idx_1 = sms.train_test_split(
            indices, train_size=split, stratify=self.labels[:, idx_l])

        # create new instances with split datasets
        unit_0 = FeatureVector(
            self.vectors[idx_0], self.labels[idx_0], self.shuffle[idx_0])
        unit_1 = FeatureVector(
            self.vectors[idx_1], self.labels[idx_1], self.shuffle[idx_1])
        return unit_0, unit_1

    def k_fold_split(
            self,
            k: int,
            idx_l: int = 0
    ):
        """
        Generate a k-fold split of the vector dataset within labels. Resulting
        splits are stratified by vector label and each represent the full
        dataset.

        Args:
            k (int): Number of folds to generate. Each fold is distributed such
                that the train and test sets are roughly (k - 1)/k and 1/k
                fractions of the full dataset, respectively. Must be >= 2.
            idx_l (int): Index in instance labels attribute used to stratify.
                Defaults to 0.

        Yields:
            unit_t (FeatureVectorDataset): New instance with the train data
                split for the current fold.
            unit_v (FeatureVectorDataset): New instance with the validation
                data split for the current fold.
        """
        splitter = sms.StratifiedKFold(n_splits=k)
        indices = np.arange(len(self))
        for idx_t, idx_v in splitter.split(indices, self.labels[:, idx_l]):
            # create new train and validation subsets
            unit_t = FeatureVector(
                self.vectors[idx_t], self.labels[idx_t], self.shuffle[idx_t])
            unit_v = FeatureVector(
                self.vectors[idx_v], self.labels[idx_v], self.shuffle[idx_v])

            # return current subset, generate next one when needed
            yield unit_t, unit_v

    def leave_last_out_split(
            self,
            idx_g: int,
            start: int = 1
    ):
        """
        Generate a leave-one-out split of the vector dataset by group.
        Resulting splits are staggered such that the ith split has a training
        set of all samples in groups j < i and a test set of all samples in
        group i. The final split is the only that includes the full dataset.

        NOTE: Group labels are sorted in ascending order prior to split.

        WARNING: leave_last_out_split will not check that a given train split
        represents all possible labels. Use idx_s arg as applicable to ensure
        all training sets represent all potential labels.

        Args:
            idx_g (int): Index in instance labels attribute used to split
                groups.
            start (int): Index of first unique group to use as test set. idx_
                = 5 indicates that the first split will include a train set of
                the first 5 groups and a test set of the 6th group. Defaults to
                1, in which case the first split includes a train set of group
                1 and a test set of group 2.

        Yields:
            group (scalar): Validation set group label for current iteration.
            unit_t (FeatureVectorDataset): New instance with the train data
                split for the current fold.
            unit_v (FeatureVectorDataset): New instance with the validation
                data split for the current fold.
        """
        groups = np.unique(self.labels[..., idx_g])
        for i, group in enumerate(groups[start:]):
            idx_t = np.in1d(
                self.labels[..., idx_g],
                groups[:np.where(groups == group)[0][0]])
            idx_v = self.labels[..., idx_g] == group
            unit_t = FeatureVector(
                self.vectors[idx_t], self.labels[idx_t], self.shuffle[idx_t])
            unit_v = FeatureVector(
                self.vectors[idx_v], self.labels[idx_v], self.shuffle[idx_v])
            yield group, unit_t, unit_v

    def mask_feature(
            self,
            idx_f: int,
            fill_value: float = None
    ):
        """
        Mask a feature in instance vectors attribute with a set fill value.

        Args:
            idx_f (int/list): Index along second axis of instance vectors
                attribute to mask.
            fill_value (float, optional): Fill value for masked feature.
                Defaults to None, in which case m_idx feature will be dropped
                entirely from vectors attribute.

        Returns:
            unit_m (FeatureVectorDataset): New instance with underlying vectors
                attribute masked.
        """
        unit_m = FeatureVector(self.vectors, self.labels, self.shuffle)
        if fill_value is None:
            unit_m.vectors = np.delete(unit_m.vectors, idx_f, axis=-1)
        else:
            unit_m.vectors[:, idx_f] = fill_value

        return unit_m

    def pairwise_cos_against_reference(
            self,
            other=None,
            idx_l: int = None,
            idx_g: int = None,
            col_group: str = "group",
            label_fill: float = -1
    ):
        """
        Compute pairwise cosine similarity between vectors in instance
        Args:
            other (FeatureVector, optional):
                Another instance against which to compute similarities.
                Defaults to "None", in which case similarities are computed
                against self and reciprocal similarity values are 1.
            idx_l (int, optional):
                Index in instance labels attribute to use as label for each
                vector prior to computing similarity matrix. Must be present in
                both "self" and "other" arg, if provided.
                Defaults to "None", in which case all vectors are treated as if
                they have unique labels.
            idx_g (int, optional):
                Index in instance labels attribute to use as group identity for
                comparison against "other" arg. If "other" is "None", each
                group is compared against itself.
                Defaults to "None", in which case all vectors are treated as
                part of the same group.
            col_group (str, optional):
                Name of the index in the returned DataFrame.
                Defaults to "group".
            label_fill (int, optional):
                Used as label for similarity values between labels. Must not
                be equal to real labels in either FeatureVector arg.
                Defaults to "-1".

        TODO: update returns section of docstring
        Returns:
            groups (np.ndarray): Group order of subsequent similarity arrays.
            r_within (np.ndarray): Average cosine similarity for all within
                label vector pairings for each group.
            r_between (np.ndarray): Average cosine similarity for all between
                label vector pairings for each group.
        """
        def _cos_by_label(
                _vectors_0: np.ndarray,
                _labels_0: np.ndarray,
                _vectors_1: np.ndarray,
                _labels_1: np.ndarray,
                _fill: int
        ):
            """
            Nested func. Compute cosine similarity between every combination of
            feature vectors in input arrays. Return list of similarities within
            each label and list of correlations between all labels.

            Args:
                _vectors_0 (np.ndarray):
                    _vectors_0.shape = (N vectors_0, N features).
                _labels_0 (np.ndarray):
                    _labels_0.shape = (N vectors_0,).
                _vectors_1 (np.ndarray):
                    _vectors_1.shape = (N vectors_1, N features).
                _labels_1 (np.ndarray):
                    _labels_1.shape = (N vectors_1,).

            Returns:
                coss (pd.Series):
                    cos.loc[l] = [similarities within label l].
                    corr.loc[-1] = [similarities between labels].
            """
            mask = np.where(
                _labels_0[:, None] == _labels_1, _labels_1[None, :], _fill)
            unique = np.unique(_labels_0).tolist() + [_fill]
            cos = smp.cosine_similarity(_vectors_0, _vectors_1)
            cos = pd.Series([cos[mask == x] for x in unique], index=unique)
            return cos

        # create a dataframe to segment vector index by source
        _ndx = self.labels.shape[1]
        _vdx = "_vector_idx"
        other = self if other is None else other
        meta_self = self.labels.copy()
        meta_other = other.labels.copy()
        if idx_l is None:
            idx_l = self.labels.shape[1] + 1
            meta_self = np.concatenate(
                (meta_self, np.arange(len(self))[None]), axis=-1)
            meta_other = np.concatenate(
                (meta_other, np.arange(len(other))[None]), axis=-1)
        if idx_g is None:
            idx_g = meta_self.shape[1] + 1
            self.labels = np.concatenate(
                (self.labels, np.zeros(len(self))[None]), axis=-1)

        # add static comparison group "other" for each unique group in "self"
        _label = "_label"
        groups = np.unique(self.labels[:, idx_g])
        meta_self = pd.DataFrame(
            meta_self[:, [idx_g, idx_l]], columns=[col_group, _label])
        meta_other = pd.DataFrame(meta_other[:, idx_l], columns=[_label])
        meta_self = meta_self.assign(
            **{_vdx: np.arange(len(self))}).set_index(col_group, append=False)
        meta_other = meta_other.assign(
            **{_vdx: np.arange(len(other)), col_group: [groups] * len(other)})
        meta_other = meta_other.explode(col_group).set_index(
            col_group, append=False)

        # join self and other metadata, indices within rows will be compared
        _label_other = f"{_label} other"
        _vdx_other = f"{_vdx} other"
        meta_self = meta_self.groupby(
            level=col_group)[[_label, _vdx]].agg(lambda x: list(x))
        meta_other = meta_other.groupby(
            level=col_group)[[_label, _vdx]].agg(lambda x: list(x)).rename(
            columns={_label: _label_other, _vdx: _vdx_other})
        meta_tot = pd.concat(
            [meta_self, meta_other], axis=1, ignore_index=False, join="inner")

        # compute similarities and return output
        meta_tot = meta_tot.apply(
            lambda x: _cos_by_label(
                _vectors_0=self.vectors[x[_vdx]],
                _labels_0=np.array(x[_label]),
                _vectors_1=other.vectors[x[_vdx_other]],
                _labels_1=np.array(x[_label_other]),
                _fill=label_fill), axis=1)
        return meta_tot

    def fit_model(
            self,
            model,
            idx_l: int = 0,
    ):
        return model.fit(self.vectors, self.labels[:, idx_l])

    def encode(
            self,
            encoder
    ):
        return FeatureVector(
            encoder.transform(self.vectors), self.labels, self.shuffle)

    def classify(
            self,
            classifier,
            idx_l: int = 0,
            idx_g: int = None
    ):
        out = [self.labels[:, idx_l], classifier.predict(self.vectors)]
        out = out if idx_g is None else out + [self.labels[..., idx_g]]
        return out

    def predict_distribution(
            self,
            classifier,
    ):
        probabilities = classifier.predict_proba(self.vectors)
        predictions = np.concatenate(
            (self.labels, classifier.predict(self.vectors)), axis=-1)
        return FeatureVector(probabilities, predictions, self.shuffle)
