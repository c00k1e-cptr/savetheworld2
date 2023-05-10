# adapted from https://github.com/beanbeah/ML/blob/main/sklearn-ml-bruteforce.py <3

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import PassiveAggressiveClassifier, RidgeClassifier, SGDClassifier
from sklearn.naive_bayes import BernoulliNB, CategoricalNB, ComplementNB, GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, GradientBoostingRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.linear_model import TweedieRegressor, BayesianRidge, ARDRegression,  LinearRegression, Ridge, RidgeCV, SGDRegressor, ElasticNet, QuantileRegressor, RANSACRegressor, PoissonRegressor, PassiveAggressiveRegressor, OrthogonalMatchingPursuit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.cross_decomposition import PLSRegression

classifiers = {
    "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=0),
    "Bagging": BaggingClassifier(n_estimators=10, random_state=0),
    "ExtraTrees (Gini)": ExtraTreesClassifier(criterion="gini", n_estimators=100, random_state=0),
    "ExtraTrees (Entropy)": ExtraTreesClassifier(criterion="entropy", n_estimators=100, random_state=0),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0),
    "RandomForest": RandomForestClassifier(max_depth=2, random_state=0),
    "HistGradientBoosting": HistGradientBoostingClassifier(),
    "PassiveAggressive": PassiveAggressiveClassifier(max_iter=1000, random_state=0, tol=1e-3),
    "Ridge": RidgeClassifier(),
    "SGDClassifier": make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3)),
    "BernoulliNaiveBayes": BernoulliNB(),
    "CategoricalNaiveBayes": CategoricalNB(),
    "ComplementNaiveBayes": ComplementNB(),
    "GaussianNaiveBayes": GaussianNB(),
    "MultinomialNaiveBayes": MultinomialNB(),
    "KNeighbors": KNeighborsClassifier(n_neighbors=3),
    "NearestCentroid": NearestCentroid(),
    "DecisionTree": DecisionTreeClassifier(random_state=0),
    "ExtraTree": ExtraTreeClassifier(random_state=0)
}

regressors = {
    "AdaBoost (square)": AdaBoostRegressor(random_state=0, n_estimators=100, loss="square"),
    "AdaBoost (linear)": AdaBoostRegressor(random_state=0, n_estimators=100, loss="linear"),
    "Adaboost (exponential)": AdaBoostRegressor(random_state=0, n_estimators=100, loss="exponential"),
    "Bagging": BaggingRegressor(n_estimators=10, random_state=0),
    "GradientBoosting (huber)": GradientBoostingRegressor(random_state=0, loss="huber"),
    "GradientBoosting (sq err)": GradientBoostingRegressor(random_state=0, loss="squared_error"),
    "GradientBoosting (abs err)": GradientBoostingRegressor(random_state=0, loss="absolute_error"),
    "Random Forest (sq err)": RandomForestRegressor(max_depth=2, random_state=0, criterion="squared_error"),
    "Random Forest (poisson)": RandomForestRegressor(max_depth=2, random_state=0, criterion="poisson"),
    "HistGradientBoosting (sq err)": HistGradientBoostingRegressor(loss="squared_error"),
    "HistGradientBoosting (abs err)": HistGradientBoostingRegressor(loss="absolute_error"),
    "HistGradientBoosting (poisson)": HistGradientBoostingRegressor(loss="poisson"),
    "Linear": LinearRegression(),
    "Ridge (Linear)": Ridge(),
    "RidgeCV": RidgeCV(),
    "SGDRegressor (elasticnet)": make_pipeline(StandardScaler(), SGDRegressor(max_iter=1000, tol=1e-3, penalty="elasticnet")),
    "SGDRegressor (l2)": make_pipeline(StandardScaler(), SGDRegressor(max_iter=1000, tol=1e-3, penalty="l2")),
    "SGDRegressor (l1)": make_pipeline(StandardScaler(), SGDRegressor(max_iter=1000, tol=1e-3, penalty="l1")),
    "Elastic Net (random)": ElasticNet(random_state=0, selection="random"),
    "Elastic Net (cyclic)": ElasticNet(random_state=0, selection="cyclic"),
    "ARD": ARDRegression(),
    "BayesianRidge": BayesianRidge(),
    "Quantile (highs-ipm)": QuantileRegressor(quantile=0.8, solver="highs-ipm"),
    "RANSAC": RANSACRegressor(random_state=0),
    "PoissonRegressor": PoissonRegressor(),
    "TweedieRegressor (log)": TweedieRegressor(link="log"),
    "PassiveAggressiveRegressor (epsilon_insensitive)":  PassiveAggressiveRegressor(max_iter=100, random_state=0, tol=1e-3, loss="epsilon_insensitive"),
    "PassiveAggressiveRegressor (squared_epsilon_insensitive)":  PassiveAggressiveRegressor(max_iter=100, random_state=0, tol=1e-3, loss="squared_epsilon_insensitive"),
    "KNeighbors": KNeighborsRegressor(n_neighbors=3),
    "MLP": MLPRegressor(random_state=1, max_iter=500),
    "DecisionTree": DecisionTreeRegressor(random_state=0),
    "Extra Tree": ExtraTreeRegressor(random_state=0),
    "Linear SVR (epsilon_insensitive)": make_pipeline(StandardScaler(), LinearSVR(random_state=0, tol=1e-5, loss="epsilon_insensitive")),
    "Linear SVR (squared_epsilon_insensitive)": make_pipeline(StandardScaler(), LinearSVR(random_state=0, tol=1e-5, loss="squared_epsilon_insensitive")),
    "PLS": PLSRegression(n_components=2),
    "OrthogonalMatchingPursuit": OrthogonalMatchingPursuit(),
}


def test_classifier(classifier_type, X_train, X_test, y_train, y_test):
    print("Classifier", classifier_type, "training...")
    clf = classifiers[classifier_type]
    clf.fit(X_train, y_train)
    y_predicted = clf.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_predicted) * 100
    # print("Classifier", classifier_type, "completed with accuracy", accuracy, '%')
    pickle.dump(
        clf, open(f'trainedmodels/classifiers/{classifier_type}.pkl', 'wb'))
    # to read the model use below code
    # with open('filename.pkl', 'rb') as f:
    #     clf = pickle.load(f)
    return [classifier_type, accuracy]


def test_classifiers(X_train, X_test, y_train, y_test):
    result_queue = []
    multiple_results = [
        (test_classifier(key, X_train, X_test, y_train, y_test)) for key in classifiers]
    for res in multiple_results:
        if res:
            try:
                tmp = res[0]
                if tmp is not None:
                    result_queue.append(tmp)
            except TimeoutError:
                print("\nClassifier", res[1], "exceeded the time limit.")
            except MemoryError:
                print("\nClassifier", res[1], "exceeded the memory limit.")

    accuracy = {}
    for value in multiple_results:
        accuracy[value[0]] = value[1]
    accuracy = {k: v for k, v in sorted(
        accuracy.items(), key=lambda item: item[1], reverse=True)}

    returning = "---"*20
    returning += "\nResults (larger accuracy better): "

    i = 1
    for key in accuracy:
        returnin = str(i).zfill(2) + ' ' + key + ' ' + '{:.2f}'.format(accuracy[key]) + '%'
        returning += "\n" + returnin
        i += 1
    returning += "\n" + "---"*20
    return returning


def test_regressor(regressor_type, X_train, X_test, y_train, y_test):
    print("Regressor", regressor_type, "training..")
    reg = regressors[regressor_type]
    reg.fit(X_train, y_train)
    y_predicted = reg.predict(X_test)
    accuracy = metrics.mean_absolute_error(y_test, y_predicted) * 100
    pickle.dump(
        reg, open(f'trainedmodels/regressors/{regressor_type}.pkl', 'wb'))
    # print("\nRegressor", regressor_type, "accuracy is", accuracy)
    return [regressor_type, accuracy]


def test_regressors(X_train, X_test, y_train, y_test):
    result_queue = []
    multiple_results = [
        (test_regressor(key, X_train, X_test, y_train, y_test)) for key in regressors]
    for res in multiple_results:
        if res:
            try:
                tmp = res[0]
                if tmp is not None:
                    result_queue.append(tmp)
            except TimeoutError:
                print("\nClassifier", res[1], "exceeded the time limit.")
            except MemoryError:
                print("\nClassifier", res[1], "exceeded the memory limit.")

    mae = {}
    for value in multiple_results:
        mae[value[0]] = value[1]
    mae = {k: v for k, v in sorted(
        mae.items(), key=lambda item: item[1], reverse=False)}
    returning = "\n\n"
    returning += "---"*20
    returning += "\nResults (smaller error better): "
    i = 1
    for key in mae:
        returnin = "\n" + (str(i).zfill(2) + ' ' + key + ' ' +
                    '{:.2f}'.format(mae[key]) + '%')
        returning += returnin
        i += 1
    returning += "\n" + "---"*20
    return returning


def train_models():
    # get data
    data = pd.read_csv("diabetes_dataset.csv")

    # prepocess
    gendertypes = {'Male': 0, 'Female': 1, 'Other': 0.5}
    data.gender = [gendertypes[x] for x in data.gender]
    smokingtypes = {'No Info': 0.5, 'never': 0,
                    'current': 1, 'former': 0.25, 'not current': 0.75}
    data.smoking_history = [smokingtypes[x] for x in data.smoking_history]

    # <removed gender>,age,hypertension,heart_disease,<removed smoking_history>,bmi,HbA1c_level,blood_glucose_level,diabetes
    features = ['age', 'HbA1c_level', 'blood_glucose_level']

    X = data[features]
    y = data.diabetes

    plt.matshow(data.corr())
    plt.xticks(range(data.shape[1]), data.columns, fontsize=11, rotation=90)
    plt.yticks(range(data.shape[1]), data.columns, fontsize=11)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title('Correlation Matrix', fontsize=16)
    plt.show()

    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.1)

    output = ""
    output += test_classifiers(Xtrain, Xtest, ytrain, ytest)
    output += test_regressors(Xtrain, Xtest, ytrain, ytest)
    print(output)
    with open("trainedmodels/results.txt", "w") as f:
        f.write(output)

def predict(inputs):
    # age,  HbA1c_level,  blood_glucose_level
    with open('trainedmodels/regressors/GradientBoosting (abs err).pkl', 'rb') as f:
        clf = pickle.load(f)
    outputs = clf.predict(inputs)
    return outputs

# inputs = [[80, 7.0, 150]]
# print(predict(inputs))

# train_models()