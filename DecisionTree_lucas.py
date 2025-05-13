import numpy as np
import multiprocessing

class DecisionTree:
        max_depth: int
        feature_index: int
        threshold: float
        value: int
        left: 'DecisionTree'
        right: 'DecisionTree'

        def __init__(self, max_depth:int):
            self.max_depth = max_depth
            self.feature_index = None
            self.threshold = None
            self.value = None
            self.left = None
            self.right = None

        def entropy(Y:np.ndarray)->float :
            Y_unique, counts = np.unique(Y, return_counts=True)
            proportions = counts/len(Y)
            return -np.sum(proportions*np.log2(proportions))
        
        def best_split(X:np.ndarray, y:np.ndarray, feature_index: int) -> (float, float):
            gain = 0
            threshold = -1
            l = len(y)
            for theta in np.unique(X[:, feature_index]):
                y_moins = y[X[:, feature_index] <= theta]
                y_plus = y[X[:, feature_index] > theta]
                y_moins_entropy = DecisionTree.entropy(y_moins)
                y_plus_entropy = DecisionTree.entropy(y_plus)
                new_gain = DecisionTree.entropy(y) - (len(y_plus)/l)*y_plus_entropy - (len(y_moins)/l)*y_moins_entropy
                if new_gain > gain or threshold == -1:
                    gain = new_gain
                    threshold = theta
            return (gain, threshold)
                 


        def best_split_pair(X:np.ndarray, y:np.ndarray) -> (float, int, float):
            gain = 0
            for s in range(X.shape[1]):
                new_gain = DecisionTree.best_split(X, y, s)[0]
                if new_gain > gain:
                    gain = new_gain
                    feature_index = s
                    threshold = DecisionTree.best_split(X, y, s)[1]
            return (gain, feature_index, threshold)
        
        def fit(self, X:np.ndarray, y:np.ndarray, depth:int=0)->None :
            if depth == self.max_depth or len(np.unique(y)) == 1:
                self.value = np.argmax(np.bincount(y))
                return
            else :
                self.feature_index, self.threshold = DecisionTree.best_split_pair(X, y)[1:]
                self.left = DecisionTree(self.max_depth)
                self.right = DecisionTree(self.max_depth)
                X_moins = X[X[:, self.feature_index] <= self.threshold]
                y_moins = y[X[:, self.feature_index] <= self.threshold]
                X_plus = X[X[:, self.feature_index] > self.threshold]
                y_plus = y[X[:, self.feature_index] > self.threshold]
                self.left.fit(X_moins, y_moins, depth+1)
                self.right.fit(X_plus, y_plus, depth+1)

        def predict(self, X:np.ndarray) -> int:
            if self.value is not None:
                return self.value
            else:
                if X[self.feature_index] <= self.threshold:
                    return self.left.predict(X)
                else:
                    return self.right.predict(X)


def train_single_tree(args):        #fonction necessaire pour le multiprocessing
    tree, X_sample, y_sample = args
    tree.fit(X_sample, y_sample)
    return tree
                
class RandomForest:
    """
    Attributes:
    - trees: np.array
    """

    def __init__(self, nbtrees=1, max_depth=1):
            self.trees = np.array([DecisionTree(max_depth) for _ in range(nbtrees)])

    def fit_single_tree(self, X: np.array, y: np.array, ratio=0.3) -> None:
            """Build the decision trees in the `trees` array,
            each using a proportion `ratio` of the data.
            """
            for tree in self.trees:
                # On choisit uniformément un sous-ensemble de données
                indices = np.random.choice(len(X), int(len(X) * ratio), replace=True)
                X_sample = X[indices]
                y_sample = y[indices]
                tree.fit(X_sample, y_sample)


    def fit(self, X: np.ndarray, y: np.ndarray, ratio=0.3) -> None:
        args = []
        for tree in self.trees:
            indices = np.random.choice(len(X), int(len(X) * ratio), replace=True)
            X_sample = X[indices]
            y_sample = y[indices]
            args.append((tree, X_sample, y_sample))

        with multiprocessing.Pool() as pool:
            self.trees = np.array(pool.map(train_single_tree, args))

    def predict(self, X:np.ndarray) -> int:
            predictions = np.array([tree.predict(X) for tree in self.trees])
            return np.argmax(np.bincount(predictions))
             
        
    def my_MNIST_Fashion_parameters()->(int, int, float):# Modify RHS in the next line:
        nb_trees, max_depth, ratio = 10,20,0.35
        return (nb_trees, max_depth, ratio)