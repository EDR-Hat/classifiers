import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.datasets import load_iris, load_digits
iris = load_digits()

X_train, X_test,Y_train, Y_test = train_test_split(iris.data, iris.target, shuffle=True)

from sklearn.neighbors import KNeighborsClassifier as knc
neigh = knc(n_neighbors=5, weights=custom)
cross_val_score(neigh, X_train, Y_train, cv=5)

def kneighbor(neighbor, algo, weight):
  mod = knc(n_neighbors=neighbor, algorithm=algo, weights=weight)
  return mod

param_grid = {
    "n_neighbors" : [2, 3, 4, 5, 6, 7, 8, 9, 10],
    "algorithm" : ["auto", "ball_tree", "kd_tree", "brute"],
    "weights" : ["uniform", "distance"],
    "p" : [1, 2]
}

kneigh_base = kneighbor(2, "auto", "uniform")
gs = GridSearchCV(knc(), param_grid, scoring="accuracy", cv=5)

gs.fit(X_train, Y_train)

results = pd.DataFrame(gs.cv_results_["params"])
results["mean score"] = gs.cv_results_["mean_test_score"]

# this was really similar to the lab; however I decided to add more granularity to the test
# turns out that the neighbors is more optimal at 4 neighbors with distance.
# i attempted the z-score function as a custom one just to try it out. turns out to
# perform terribly, probably due to negative numbers, leanding to mean scores of NaN
# it looks like the best parameters are 4 neighbors and euclidean distance for the weights
results.sort_values(by="mean score", ascending=False)

#part 2

import tensorflow as tf
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.reshape(60000, 28*28)
x_test = x_test.reshape(10000, 28*28)

from sklearn.ensemble import RandomForestClassifier

#trying default settings random forest
rf = RandomForestClassifier()
model_rf = rf.fit(x_train, y_train)

cross_val_score(model_rf, x_test, y_test, cv=5)
#nearest neighbor
neigh_model = knc()
cross_val_score(knc(), x_test, y_test, cv=5)

from sklearn.svm import SVC
svc_model = SVC(probability=True)
cross_val_score(svc_model, x_test, y_test, cv=5)

from sklearn.ensemble import VotingClassifier

softCL = VotingClassifier(
    estimators=[
                ('svc', svc_model), ('knc', neigh_model), ('rf', model_rf)
    ], voting='soft')
softCL = softCL.fit(x_train, y_train)

cross_val_score(softCL, x_test, y_test, cv=5)

# The soft voting classifier did better on average
# than the nearest neighbor and random forest classifiers
# The SVC was performed the highest on average, getting
# to 97% accuracy while also dipping fairly low.
# I think the soft voting classifier attempts to bridge
# the gaps that these classifiers have, for instance
# SVC does well on data in hard lined clusters but
# isn't as accurate on a bleeding edge. I feel like this
# normalized some of the variance of SVC but not by that
# many percentage points. Using a voting classifier is
# probably best if these classifiers are put through
# some sort of hyper parameter optimization before
# being added to the Voting Classifier.

# Also I hadn't tried out SVC before and was surprised
# just how long it took to run. I was finishing up
# the last part here and it took a very long time
# for anything involving the SVC to fit, run or predict.
