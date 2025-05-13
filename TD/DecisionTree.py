import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split

class DecisionTree:
        max_depth: int
        feature_index: int
        threshold: float
        value: int
        left: 'DecisionTree'
        right: 'DecisionTree'

        def __init__(self, max_depth: int = 10):
            self.max_depth = max_depth
            self.feature_index = None
            self.threshold = None
            self.value = None
            self.left = None
            self.right = None
        
        @staticmethod
        def entropy(y:np.array)->float:
            """
            Nous calculons l'entropie de y
            """
            _, counts = np.unique(y, return_counts=True)
            probabilities = counts / len(y)
            return -np.sum(probabilities * np.log2(probabilities + 1e-10))  #On souhaite éviter de faire log(0). De tte facon si c'est 0 le tout fera 0

        @staticmethod
        def best_split(X:np.array, y:np.array, feature_index:int)->tuple:
            """
            Calcul de la meilleure séparation de X et y
            """
            #Initialisation des variables
            gain = 0
            threshold = 0
            cond = False    #Nous permet de mettre à jour le premier élément de la liste
            l = len(y)
            for theta in np.unique(X[:, feature_index]):
                #Calcul des composantes de l'entropie
                y_moins = y[X[:, feature_index] <= theta]
                y_plus = y[X[:, feature_index] > theta]
                y_moins_entropy = DecisionTree.entropy(y_moins)
                y_plus_entropy = DecisionTree.entropy(y_plus)
                new_gain = DecisionTree.entropy(y) - (len(y_plus)/l)*y_plus_entropy - (len(y_moins)/l)*y_moins_entropy

                #Mise à jour de la sépartion si le theta est meilleur
                if new_gain > gain or not cond:
                    gain = new_gain
                    cond = True
                    threshold = theta
            return (gain, threshold)

        
        def best_split_pair(X:np.ndarray, y:np.ndarray) -> (float, int, float):
            gain, feature_index, threshold = 0., 0, 0.
            cond = False
            for s in range(X.shape[1]):
                new_gain, new_threshold = DecisionTree.best_split(X, y, s)
                if new_gain > gain or not cond:
                    gain = new_gain
                    cond = True
                    threshold = new_threshold
                    feature_index = s
            return (gain, feature_index, threshold)

        def fit(self, X:np.ndarray, y:np.ndarray, depth:int=0)->None:
            """
            Nous allons créer l'arbre de décision
            """
            if depth == self.max_depth or len(np.unique(y)) == 1:
                self.value = np.argmax(np.bincount(y))
                self.left = None
                self.right = None
                return
            else:
                _, self.feature_index, self.threshold = DecisionTree.best_split_pair(X, y)
                self.left = DecisionTree(self.max_depth)
                self.right = DecisionTree(self.max_depth)
                X_moins = X[X[:, self.feature_index] <= self.threshold]
                y_moins = y[X[:, self.feature_index] <= self.threshold]
                X_plus = X[X[:, self.feature_index] > self.threshold]
                y_plus = y[X[:, self.feature_index] > self.threshold]
                #Nous allons créer les sous arbres
                self.left.fit(X_moins, y_moins, depth + 1)
                self.right.fit(X_plus, y_plus, depth + 1)
                return

def main():
    '''
    Initialisation
    '''
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    print(X.shape,y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=571)

    X_train=np.loadtxt("csv/fashion-mnist_2000_X_train.csv", delimiter=",", dtype=float)
    y_train=np.loadtxt("csv/fashion-mnist_2000_y_train.csv", delimiter=",", dtype=int)
    X_test=np.loadtxt("csv/fashion-mnist_500_X_test.csv", delimiter=",", dtype=float)
    y_test=np.loadtxt("csv/fashion-mnist_500_y_test.csv", delimiter=",", dtype=int)
    print(f"Shapes {X_train.shape, y_train.shape, X_test.shape, y_test.shape}") #Ici les Y sont des réels donc la shape est (2000, 1) et (500, 1)
    '''
    #Test Q2
    '''
    print(f"2) feature = 0 -> {DecisionTree.best_split(X,y,0)}")
    print(f"2) feature = 1 -> {DecisionTree.best_split(X,y,1)}")
    print(f"2) feature = 2 -> {DecisionTree.best_split(X,y,2)}")
    '''
    #Test Q3
    '''       
    def print_tree(node:DecisionTree, depth=0):
        if node.value is not None:
            print((' '*2*depth)+f"- class={node.value}")
        else:
            print((' '*2*depth)+f"- X[{node.feature_index}] <= {node.threshold}")
            print_tree(node.left, depth + 1)
            print_tree(node.right, depth + 1)

    mytree = DecisionTree(max_depth=3)
    mytree.fit(iris.data, iris.target)
    print_tree(mytree)

if __name__ == "__main__":
    main()
