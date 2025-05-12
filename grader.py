#! /usr/bin/env python3 
import sys
import time
from TD.DecisionTree import *
import numpy as np

import unittest


"""
Annotations used for the autograder.

[START-AUTOGRADER-ANNOTATION]
{
  "total" : 7,
  "names" : [
        "DecisionTree.py::test_entropy",
        "DecisionTree.py::test_splits",
        "DecisionTree.py::test_tree_fit",
        "DecisionTree.py::test_tree_predict",
        "DecisionTree.py::test_forest_fit",
        "DecisionTree.py::test_forest_predict",
        "DecisionTree.py::test_challenge"
        ],
  "points" : [10, 10, 16, 16, 16, 16, 16]
}
[END-AUTOGRADER-ANNOTATION]
"""



class Grader(unittest.TestCase):
    y = np.array([1, 2, 1, 0, 1, 2, 2, 2, 0, 2, 2, 0, 1, 2, 2, 1, 2, 2, 2, 0, 2, 2, 0, 0, 1, 2, 0, 0, 0, 2])
    X = np.array(
        [[6., 2.9, 4.5, 1.5], [5.6, 2.8, 4.9, 2.], [5.7, 2.8, 4.1, 1.3], [5.5, 3.5, 1.3, 0.2], [5.8, 2.6, 4., 1.2],
         [7.3, 2.9, 6.3, 1.8], [7.9, 3.8, 6.4, 2.], [6.7, 3.3, 5.7, 2.5], [5.4, 3.7, 1.5, 0.2], [6.2, 2.8, 4.8, 1.8],
         [6.1, 3., 4.9, 1.8], [5., 3.6, 1.4, 0.2], [6.1, 2.8, 4.7, 1.2], [6.3, 2.7, 4.9, 1.8], [6.3, 3.4, 5.6, 2.4],
         [5.9, 3.2, 4.8, 1.8], [6., 2.2, 5., 1.5], [6.3, 3.3, 6., 2.5], [7.7, 3.8, 6.7, 2.2], [5., 3.5, 1.6, 0.6],
         [6.7, 3., 5.2, 2.3], [6.4, 2.7, 5.3, 1.9], [5.4, 3.4, 1.5, 0.4], [4.8, 3., 1.4, 0.3], [6.6, 2.9, 4.6, 1.3],
         [7.7, 3., 6.1, 2.3], [5.1, 3.7, 1.5, 0.4], [5.8, 4., 1.2, 0.2], [5.7, 4.4, 1.5, 1.9], [5.8, 2.7, 5.1, 1.9]])

    def test_entropy(self):
        err_msg="Wrong computation of entropy"
        self.assertAlmostEqual(DecisionTree.entropy(np.array([1, 4, 4, 1])),1.0,msg=err_msg)
        self.assertAlmostEqual(DecisionTree.entropy(np.array([1, 1,1,1,1,1, 1])),0.0,msg=err_msg)
        self.assertAlmostEqual(DecisionTree.entropy(np.array([-10.])),0.0,msg=err_msg)
        self.assertAlmostEqual(DecisionTree.entropy(np.array([1, 1, 5, 5])),1.0,msg=err_msg)
        self.assertAlmostEqual(DecisionTree.entropy(np.array([1, 4, 4, 1,2,2,3,3,6])),2.281036112, msg=err_msg )

    def test_splits(self):
        X,y = self.X,self.y
        err_msg="Wrong computation of best_gain in best_split"
        self.assertAlmostEqual(DecisionTree.best_split(X,y,0)[0]/0.5545155955321148,1.0,msg=err_msg)
        self.assertAlmostEqual(DecisionTree.best_split(X,y,1)[0]/0.43613669571753055,1.0,msg=err_msg)
        self.assertAlmostEqual(DecisionTree.best_split(X,y,2)[0]/0.8812908992306927,1.0,msg=err_msg)
        self.assertAlmostEqual(DecisionTree.best_split(X,y,3)[0]/0.6856632415084736,1.0,msg=err_msg)
        err_msg="Wrong computation of best_threshold in best_split"
        self.assertAlmostEqual(DecisionTree.best_split(X,y,0)[1]/5.5,1.0,msg=err_msg)
        self.assertAlmostEqual(DecisionTree.best_split(X,y,1)[1]/3.3,1.0,msg=err_msg)
        self.assertAlmostEqual(DecisionTree.best_split(X,y,2)[1]/1.6,1.0,msg=err_msg)
        self.assertAlmostEqual(DecisionTree.best_split(X,y,3)[1]/0.6,1.0,msg=err_msg)

        err_msg="Wrong computation of best_gain in best_split_pair"
        self.assertAlmostEqual(DecisionTree.best_split_pair(X,y)[0]/0.8812908992306927,1.0,msg=err_msg)
        err_msg="Wrong computation of best_feature in best_split_pair"
        self.assertAlmostEqual(DecisionTree.best_split_pair(X,y)[1]/2.,1.0,msg=err_msg)
        err_msg="Wrong computation of best_fthreshold in best_split_pair"
        self.assertAlmostEqual(DecisionTree.best_split_pair(X,y)[2]/1.6,1.0,msg=err_msg)

    def tree_size(self,tree:'DecisionTree'):
        s=1
        if tree.left!=None:
            s=s+self.tree_size(tree.left)
        if tree.right!=None:
            s=s+self.tree_size(tree.right)
        return s

    def tree_contour(self,tree:'DecisionTree'):
        if tree.value!=None:
            return [tree.value,tree.threshold, tree.feature_index]
        else:
            return [tree.value,tree.threshold, tree.feature_index]+self.tree_contour(tree.left)+self.tree_contour(tree.right)


    def test_tree_fit(self):
        X,y = self.X,self.y
        mytree = DecisionTree(max_depth=2)
        mytree.fit(X,y)
        err_msg = "Wrong computation of the tree..."
        contour=self.tree_contour(mytree)
        sol=[None, np.float64(1.6), 2, np.int64(0), None, None, None, np.float64(4.8), 2, np.int64(1), None, None, np.int64(2), None, None]
        for i in range(len(sol)):
            if ((contour[i]==None) or (sol[i]==None)):
                self.assertTrue((contour[i]==None) and (sol[i]==None), msg=err_msg)
            else:
                self.assertAlmostEqual(contour[i],sol[i], msg=err_msg)

    def test_tree_predict(self):
        X, y = self.X, self.y
        mytree = DecisionTree(max_depth=4)
        mytree.fit(X, y)
        err_msg = "Wrong prediction from the tree.)"
        self.assertEqual(mytree.predict([1., 1., 2., 1.7]),1, msg=err_msg)
        self.assertEqual(mytree.predict([0., 0., 0., 0.]),0, msg=err_msg)
        self.assertEqual(mytree.predict([1., 1., -2., 1.7]),0, msg=err_msg)
        self.assertEqual(mytree.predict([6.15, 13., 3, 1.7,]),2, msg=err_msg)
        self.assertEqual(mytree.predict([6.21, 13., 3., 1.,4]),1, msg=err_msg)

    def test_forest_fit(self):
        X, y = self.X, self.y
        fo = RandomForest(100, 15)
        fo.fit(X, y, 0.3)
        err_msg = "Your forests contains only trees of exactly the same size... this is almost impossible!)"
        self.assertTrue(np.unique([self.tree_size(tree) for tree in fo.trees]).shape[0]>1, msg=err_msg)

    def test_forest_predict(self):
        X, y = self.X, self.y
        fo = RandomForest(300, 8)
        err_msg = "Your forest seems to make very deterministic predictions!)"
        l=[]
        fo.fit(X, y, 0.5)
        for tree in fo.trees:
            l=l+[tree.predict([6.21, 13., 3., 1.,4])]
        u,c = np.unique(l, return_counts=True)
        #Explanation for the test: typical prediction counts with these parameters should be
        # around 60,150,60 (experimentally) so below 10 is impossible
        # (this tests that there is randomness among the trees)
        self.assertTrue(c[0]>10 and c[1]>10 and c[2]>10, msg=err_msg)
        err_msg = "Your forest should not make this (very unlikely) prediction"
        self.assertEqual(fo.predict([6.21, 13., 3., 1., 4]), 1,  msg=err_msg)

    def test_challenge(self):
        X_train = np.loadtxt("csv/fashion-mnist_2000_X_train.csv", delimiter=",", dtype=float)
        y_train = np.loadtxt("csv/fashion-mnist_2000_y_train.csv", delimiter=",", dtype=int)
        X_test = np.loadtxt("csv/fashion-mnist_500_X_test.csv", delimiter=",", dtype=float)
        y_test = np.loadtxt("csv/fashion-mnist_500_y_test.csv", delimiter=",", dtype=int)
       
        ts=time.time()
        nb_trees, max_depth, ratio = RandomForest.my_MNIST_Fashion_parameters()

        fo = RandomForest(nb_trees, max_depth)
        fo.fit(X_train, y_train, ratio)
        accuracy=1. - np.sum(np.int64(([fo.predict(x) for x in X_test] - y_test) != 0)) / len(X_test)
        print(
            f"(nb_trees={nb_trees}, depth={max_depth}, accuracy={accuracy:.3f},ratio={ratio},time={np.int64(time.time() - ts)})",
            flush=True)
        match accuracy:
            case c if c < 0.90:
                print("Grade: D")
            case c if 0.90 <= c < 0.92:
                print("Grade: C")
            case c if 0.92 <= c < 0.93:
                print("Grade: B")
            case c if 0.93 <= c < 0.945:
                print("Grade: A")
            case c if c >= 0.945:
                print("Grade: A+")


            


def print_help():
    print(
        "./grader script. Usage: ./grader.py test_number, e.g., ./grader.py 1 for the 1st exercise."
    )
    print("N.B.: ./grader.py 0 runs all tests.")
    print(f"You provided {sys.argv}.")
    exit(1)

def suite(test_nb):
    suite = unittest.TestSuite()
    test_name = [
        "test_entropy",
        "test_splits",
        "test_tree_fit",
        "test_tree_predict",
        "test_forest_fit",
        "test_forest_predict",
        "test_challenge",
    ]

    if test_nb > 0:
        suite.addTest(Grader(test_name[test_nb - 1]))
    else:
        for name in test_name:
            suite.addTest(Grader(name))

    return suite


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print_help()
    try:
        test_nb = int(sys.argv[1])
    except ValueError as e:
        print(
            f"You probably didn't pass an int to ./grader.py: passed {sys.argv[1]}; error {e}"
        )
        exit(1)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite(test_nb))
