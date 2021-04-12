import numpy as np
import sklearn.linear_model as skl
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import Lasso, LinearRegression, RidgeCV

from algorithm.decompose.nnlinear import nnLinearSolver

class TreeNode(object):
    def __init__(self, split_feature, split_value):
        self.split_feature = split_feature
        self.split_value = split_value

    def add_children(self, left_node, right_node):
        self.left_node = left_node
        self.right_node = right_node

class LeafNode(object):
    def __init__(self, number, linear_weight, linear_bias):
        self.number = number
        self.weight = linear_weight
        self.bias = linear_bias

class Splitter(object):
    def __init__(self, min_leaf_size=1, min_error_red=0):
        self.min_leaf_size = min_leaf_size
        self.min_error_red = min_error_red

    def split(self, X, y):
        # X = to_numpy(X)
        if (y.size <= self.min_leaf_size):
            return None, None, None
        X_ori_train, X_ori_test, y_ori_train, y_ori_test = train_test_split(X, y, test_size=0.05)
        # model = LinearRegression(positive=True)
        model = nnLinearSolver()
        model.fit(X_ori_train, y_ori_train)
        y_ori_pred = model.predict(X_ori_test)
        score_ori = np.sum((y_ori_pred - y_ori_test) ** 2)
        score_best = score_ori
        feature_best = 0
        split_best = 0.0
        for i in range(X.shape[1]):
        # for i in range(1):
            # x_uniq = np.unique(np.around(X[:, i], decimals=2))
            x_uniq = np.unique(np.around(X[:, i], decimals=4))
            # x_uniq = np.unique(X[:, i])
            # print(x_uniq.shape)
            for j in range(x_uniq.size - 1):
                split_tmp = (x_uniq[j] + x_uniq[j+1]) / 2.0
                index_left = X[:, i] < split_tmp
                index_right = (1 - index_left).astype(np.bool)
                size_left = np.sum(index_left != 0)
                size_right = index_left.size - size_left
                if (size_left < self.min_leaf_size or size_right < self.min_leaf_size):
                    continue
                X_left = X[index_left]
                y_left = y[index_left]
                X_right = X[index_right]
                y_right = y[index_right]
                X_left_train, X_left_test, y_left_train, y_left_test = train_test_split(X_left, y_left, test_size=0.05)
                # model_left = LinearRegression(positive=True)
                model_left = nnLinearSolver()
                model_left.fit(X_left_train, y_left_train)
                y_left_pred = model_left.predict(X_left_test)
                score_tmp = np.sum((y_left_pred - y_left_test) ** 2)
                X_right_train, X_right_test, y_right_train, y_right_test = train_test_split(X_right, y_right, test_size=0.05)
                # model_right = LinearRegression(positive=True)
                model_right = nnLinearSolver()
                model_right.fit(X_right_train, y_right_train)
                y_right_pred = model_right.predict(X_right_test)
                score_tmp += np.sum((y_right_pred - y_right_test) ** 2)
                if (score_ori - score_tmp < self.min_error_red or score_tmp > score_best):
                    continue
                else:
                    score_best = score_tmp
                    feature_best = i
                    split_best = split_tmp
                    print("++Tmp++Best feature: %d & Best Split: %f\n" % (feature_best, split_best))
                    X_left_ret = X_left
                    y_left_ret = y_left
                    X_right_ret = X_right
                    y_right_ret = y_right
                    # plt.cla()
                    # plt.plot(X, y)
                    # plt.plot(X_left_test, y_left_pred, 'r-')
                    # plt.plot(X_right_test, y_right_pred, 'r-')
                    # plt.pause(0.1)
        if (score_best == score_ori):
            return None, None, None
        # print("One TreeNode: (%d, %lf)" % (feature_best, split_best))
        return feature_best, split_best, [X_left_ret, X_right_ret, y_left_ret, y_right_ret]

class RegTree:
    def __init__(self, min_leaf_size=30, min_error_red=0):
        self.leaves = []
        if (min_leaf_size < 30):
            self.min_leaf_size = 30
        else:
            self.min_leaf_size = min_leaf_size
        self.min_error_red = min_error_red

    def train(self, X, y):
        print(skl.__file__)
        self.root = self.creat_tree(X, y)

    def creat_tree(self, X, y):
        splitter = Splitter(self.min_leaf_size, self.min_error_red)
        # 通过数据划分创建树节点
        feature_split, split_best, data_split = splitter.split(X, y)
        if feature_split == None:
            # model = LinearRegression(positive=True)
            model = nnLinearSolver()
            model.fit(X, y)
            # leaf_tmp = LeafNode(len(self.leaves), model.coef_, model.intercept_)
            leaf_tmp = LeafNode(len(self.leaves), model.weights, model.bias)
            self.leaves.append(leaf_tmp)
            return leaf_tmp
        else:
            node_tmp = TreeNode(feature_split, split_best)
            node_left = self.creat_tree(data_split[0], data_split[2])
            node_right = self.creat_tree(data_split[1], data_split[3])
            node_tmp.add_children(node_left, node_right)
            return node_tmp

    def whichLeaf(self, node, x):
        if isinstance(node, LeafNode):
            print("Weights:", node.weight, "Bias:", node.bias)
            return np.sum(x * node.weight) + node.bias
        elif isinstance(node, TreeNode):
            fea_which = node.split_feature
            if x[fea_which] < node.split_value:
                ret = self.whichLeaf(node.left_node, x)
            else:
                ret = self.whichLeaf(node.right_node, x)
        return ret

    def leafWB(self, node, x):
        if isinstance(node, LeafNode):
            return node.weight, node.bias
        elif isinstance(node, TreeNode):
            fea_which = node.split_feature
            if x[fea_which] < node.split_value:
                ret = self.leafWB(node.left_node, x)
            else:
                ret = self.leafWB(node.right_node, x)
        return ret

    def predict(self, X):
        len_y = X.shape[0]
        y = np.zeros(len_y, dtype=X.dtype)
        for i in range(len_y):
            y[i] = self.whichLeaf(self.root, X[i])
        return y

    def pre_pred(self, X):
        len_y = X.shape[0]
        w = []
        b = []
        # print(X)
        for i in range(len_y):
            tmp_w, tmp_b = self.leafWB(self.root, X[i])
            w.append(tmp_w)
            b.append(tmp_b)
        return np.array(w), np.array(b)

def predict_wrapper(model, X_status, Xs, types, cpu_ratios, feanum_onetype, multi=False):
    if len(Xs) != len(types) != len(cpu_ratios):
        raise AttributeError("List length isn't match!")
    # 取出模型权重
    W, B = model.pre_pred(X_status)
    ret_list = []
    for i in range(len(types)):
        w_tmp = W[:, feanum_onetype * (int(types[i])-1): feanum_onetype * int(types[i])]
        ret_list.append((np.sum(Xs[i] * w_tmp) + cpu_ratios[i] * B[i]).tolist() if multi
                        else (np.sum(Xs[i] * w_tmp) + cpu_ratios[i] * B).tolist())
        # ret_list.append((np.sum(Xs[i] * w_tmp) + cpu_ratios[i] * B[i]).tolist())
    return ret_list