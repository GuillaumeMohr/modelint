# coding: utf-8
#
# Parse xgboost text dump file and determines the weight of each variable
# in a prediction
#
# author: Guillaume MOHR

import re
import numpy as np
import pandas as pd
import seaborn as sns

# one regular expression for each type of line
rex_booster = re.compile(r'^booster\[(\d+)\]:$')
rex_node = re.compile(r'(\d+):\[(\w+)<(.*)\] yes=(\d+),no=(\d+).*cover=(\d+)')
rex_leaf = re.compile(r'(\d+):leaf=(.*),cover=(\d+)')


class Tree:
    """
    Represents one tree, with n nodes and l leaves.
    It is constructed step by step using a list name `tree` and then,
    it is transformed into several arrays of lengths (n+l) :
        lefts: if i is a node, lefts[i] is left child index
                if i is a leaf, lefts[i] == -1
        rights: if i is a node, rights[i] is left child index
                if i is a leaf, rights[i] == -1
        covs: number of samples covered by this node / leaf
        cuts: if i is a node, cuts[i] is the cut value (if feats[i] < cuts[i] => go left)
              if i is a leaf, cuts[i] is the leaf value
        feats: if i is a node, feats[i] is the feature (index) used to decide the cut
               if i is a leaf, feats[i] == -1
        means: means[i] expected value at node i
               (if i is a leaf, it is equal to cuts[i])
    """

    def __init__(self):
        self.tree = []  # used to construct the tree line by line

    def is_not_empty(self):
        return len(self.tree) > 0

    def add_leaf(self, leaf):
        """
        Adds a leaf
        Parameter
        ---------
        leaf: tuple
        """
        lid, val, cover = leaf
        lid = int(lid)
        val = float(val)
        cover = int(cover)
        self.tree.append((lid, -1, val, -1, -1, cover))

    def add_node(self, node, feat_map):
        """
        Adds a node
        Parameter
        ---------
        node: tuple
        feat_map: dict
        Returns
        -------
        feat_map: dict
        """
        nid, feat, cut, left, right, cover = node
        nid = int(nid)
        cut = float(cut)
        left = int(left)
        right = int(right)
        cover = int(cover)
        if feat not in feat_map:
            feat_map[feat] = len(feat_map)
        feat = feat_map[feat]
        self.tree.append((nid, feat, cut, left, right, cover))
        return feat_map

    def _loop_means(self, i):
        """
        Recursive function that fills means array
        Parameters
        ----------
        i: int
            current node/leaf position
        """
        if self.lefts[i] == -1:
            # we are in a leaf
            self.means[i] = self.cuts[i]
        else:
            # we are in a node
            left_i = self.lefts[i]
            right_i = self.rights[i]
            self._loop_means(left_i)
            self._loop_means(right_i)
            # weights
            covs = self.covers[[left_i, right_i]]
            assert self.covers[i] == np.sum(covs)
            # children means
            mes = self.means[[left_i, right_i]]
            self.means[i] = np.sum(mes * covs) / self.covers[i]

    def make_tree(self):
        """
        Constructs the actual tree representation.
        Should be called after self.tree is complete
        """
        self.tree.sort()  # we order the nodes
        # careful! some ids may not exist!
        ids, feats, cuts, lefts, rights, covers = \
            [np.array(x) for x in zip(*self.tree)]
        max_id = np.max(ids)
        self.feats = - np.ones(max_id + 1, dtype=np.int32)
        self.cuts = - np.ones(max_id + 1, dtype=np.float64)
        self.lefts = - np.ones(max_id + 1, dtype=np.int32)
        self.rights = - np.ones(max_id + 1, dtype=np.int32)
        self.covers = - np.ones(max_id + 1, dtype=np.int64)
        for a, b in [(self.feats, feats),
                     (self.cuts, cuts),
                     (self.lefts, lefts),
                     (self.rights, rights),
                     (self.covers, covers)]:
            a[ids] = b
        self.means = np.zeros_like(self.cuts)
        self._loop_means(0)

    def predict(self, x, feat_map):
        """Predicts according to x
        Parameters
        ----------
        x: (p,) float array
        feat_map: dict
        Returns
        -------
        prediction, feats_parts, bias: float, float array, float
            prediction, features participation in it, bias
        """
        i = 0
        parent_feat = -1
        parent_mean = 0.
        feats_parts = np.zeros(len(feat_map))
        while True:
            # node feature
            f = self.feats[i]
            if parent_feat >= 0:  # exclude the root
                feats_parts[parent_feat] += \
                    self.means[i] - parent_mean
            parent_feat = f
            parent_mean = self.means[i]
            if f >= 0:
                # we are in a node
                if x[f] < self.cuts[i]:
                    i = self.lefts[i]
                else:
                    i = self.rights[i]
            else:
                # we are in a leaf
                return self.cuts[i], feats_parts, self.means[0]


class TreeEnsemble:
    """
    Represents all the trees in a XGBoost model
    """

    def __init__(self, feat_map=None):
        """
        Parameters
        ----------
        feat_map: dictionary or str array
            either a dictionary str -> int
            or an array of string
            mapping a feature name to an index
        """
        self.trees = []
        if feat_map is None:
            self.feat_map = {}
        elif isinstance(feat_map, dict):
            self.feat_map = feat_map
        else:
            # here assume feat_map is a features array
            self.feat_map = {f: i for i, f in enumerate(feat_map)}

    def load_dump(self, fname):
        with open(fname) as f:
            tree = Tree()
            for line in f:
                booster = rex_booster.findall(line)
                if booster:
                    if tree.is_not_empty():
                        # we save the current tree
                        tree.make_tree()
                        self.trees.append(tree)
                    # we start a new tree
                    tree = Tree()
                else:
                    leaf = rex_leaf.findall(line)
                    if leaf:
                        tree.add_leaf(leaf[0])
                    else:
                        node = rex_node.findall(line)
                        if node:
                            self.feat_map = tree.add_node(node[0], self.feat_map)
                        else:
                            raise ValueError('Cannot understand line "{}"'.format(line))
            # we include the last tree
            tree.make_tree()
            self.trees.append(tree)

    def predict(self, x):
        """Predicts according to x
        Parameters
        ----------
        x: (p,) float array
        """
        preds = np.zeros(len(self.trees))
        biases = np.zeros(len(self.trees))
        feats_parts = np.zeros((len(self.trees), len(self.feat_map)))
        for i, t in enumerate(self.trees):
            pred, feats_part, bias = t.predict(x, self.feat_map)
            preds[i] = pred
            biases[i] = bias
            feats_parts[i, :] = feats_part
        return np.sum(preds), np.sum(feats_parts, axis=0), np.sum(biases)

    def plot_parts(self, x, groups=None):
        """ Plots individual parameter importance
        Parameters
        ----------
        x: (p,) array
            input variables (p features)
        groups: dict
            group variables under a common name
        """
        p, fp, b = self.predict(x)
        features = [''] * (len(self.feat_map) + 1)
        for f, i in self.feat_map.items():
            features[i] = f
        features[-1] = 'bias'
        parts = np.r_[fp, b]
        df = pd.DataFrame({'participation': parts,
                           'feature': features})
        df['group'] = df.feature.apply(lambda f: groups.get(f, f))
        df = df.groupby('group', as_index=False).sum()
        df['abs_participation'] = df['participation'].abs()
        sns.barplot(x='participation',
                    y='group',
                    data=df.sort_values('abs_participation',
                                        ascending=False))
