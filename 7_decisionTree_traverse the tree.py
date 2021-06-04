# -*- coding: utf-8 -*-
"""
https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

################## Data
iris = load_iris()
X = iris.data
y = iris.target
################## Split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

################## Fit, predict
clf = DecisionTreeClassifier(max_leaf_nodes=3, random_state=0)
clf.fit(X_train, y_train)
#DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
#            max_features=None, max_leaf_nodes=3, min_impurity_split=1e-07,
#            min_samples_leaf=1, min_samples_split=2,
#            min_weight_fraction_leaf=0.0, presort=False, random_state=0,
#            splitter='best')
##The decision estimator has an attribute called tree which stores the entire tree structure and allows access to low level attributes. The binary tree tree is represented as a number of parallel arrays. The i-th element of each array holds information about the node i. Node 0 is the tree's root. NOTE: Some of the arrays only apply to either leaves or split nodes, resp. In this case the values of nodes of the other type are arbitrary!
#Among those arrays, we have:
#- left_child, id of the left child of the node
#- right_child, id of the right child of the node
#- feature, feature used for splitting the node
#- threshold, threshold value at the node

#Using those arrays, we can parse the tree structure:
n_nodes = clf.tree_.node_count
children_left = clf.tree_.children_left
children_right = clf.tree_.children_right
feature = clf.tree_.feature


node_depth = np.zeros(shape=n_nodes)
is_leaves = np.zeros(shape=n_nodes, dtype=bool)
stack = [(0, -1)]  # seed is the root node id and its parent depth
while len(stack) > 0:
    node_id, parent_depth = stack.pop()
    node_depth[node_id] = parent_depth + 1

    # If we have a test node
    if (children_left[node_id] != children_right[node_id]):
        stack.append((children_left[node_id], parent_depth + 1))
        stack.append((children_right[node_id], parent_depth + 1))
    else:
        is_leaves[node_id] = True

print("The binary tree structure has %s nodes and has "
      "the following tree structure:"
      % n_nodes)
for i in range(n_nodes):
    if is_leaves[i]:
        print("%snode=%s leaf node." % (int ( node_depth[i] ) * "\t", i))
    else:
        print("%snode=%s test node: go to node %s if X[:, %s] <= %ss else to "
              "node %s."
              % (int ( node_depth[i] ) * "\t",
                 i,
                 children_left[i],
                 feature[i],
                 0,
                 children_right[i],
                 ))
print()