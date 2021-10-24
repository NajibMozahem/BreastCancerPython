import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer/breast-cancer.data",
                 header=None)
# label columns
df.columns = ["class", "age", "menopause", "tumor_size",
                    "inv_nodes", "node_caps", "deg_malig",
                    "breast", "breast_quad", "irradiat"]
# check for missing values
df.isnull().sum()
# no missing values

# look at the types of the columns
df.info()

# let us look at the values of the object variables
df["class"].value_counts()
df["age"].value_counts()
df["tumor_size"].value_counts()
df["inv_nodes"].value_counts()
df["menopause"].value_counts()
df["node_caps"].value_counts()
df["breast"].value_counts()
df["irradiat"].value_counts()
df["breast_quad"].value_counts()
# we note that in somce cases there are question marks. we should treat them like missing values
df.replace("?", np.nan, inplace=True)
df.isnull().sum()
# now we have nan values. drop these observations
df.dropna(inplace=True)

# many columns are objects. in fact, these objects are better thought of as categorical variables
df[df.select_dtypes("object").columns] = df.select_dtypes("object").astype("category")
# many of the categories have a natural order. We reorder so that this natural order is maintained
# the three columns are age, tumor_size, and inv_nodes

fig, ax = plt.subplots(2, 3)

long = df[["class", "age"]].groupby(["age", "class"]).size().reset_index(name="count").pivot(index="age", columns="class", values="count")
long["total"] = long.sum(axis=1)
long["no-recurrence-events"] = long["no-recurrence-events"] / long["total"]
long["recurrence-events"] = long["recurrence-events"] / long["total"]
long.drop("total", axis=1, inplace=True)
long.plot(kind="bar", stacked=True, ax=ax[0,0])
ax[0,0].set_xlabel("age")
ax[0,0].get_legend().remove()

long = df[["class", "menopause"]].groupby(["menopause", "class"]).size().reset_index(name="count").pivot(index="menopause", columns="class", values="count")
long["total"] = long.sum(axis=1)
long["no-recurrence-events"] = long["no-recurrence-events"] / long["total"]
long["recurrence-events"] = long["recurrence-events"] / long["total"]
long.drop("total", axis=1, inplace=True)
long.plot(kind="bar", stacked=True, rot=0, ax=ax[0,1])
ax[0,1].set_xlabel("menopause")
ax[0,1].get_legend().remove()

long = df[["class", "tumor_size"]].groupby(["tumor_size", "class"]).size().reset_index(name="count").pivot(index="tumor_size", columns="class", values="count")
long["total"] = long.sum(axis=1)
long["no-recurrence-events"] = long["no-recurrence-events"] / long["total"]
long["recurrence-events"] = long["recurrence-events"] / long["total"]
long.drop("total", axis=1, inplace=True)
order = ["0-4", "5-9", "10-14", "15-19", "20-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54"]
long.loc[order].plot(kind="bar", stacked=True, ax=ax[0,2])
ax[0,2].set_xlabel("tumor size")
ax[0,2].get_legend().remove()

long = df[["class", "breast"]].groupby(["breast", "class"]).size().reset_index(name="count").pivot(index="breast", columns="class", values="count")
long["total"] = long.sum(axis=1)
long["no-recurrence-events"] = long["no-recurrence-events"] / long["total"]
long["recurrence-events"] = long["recurrence-events"] / long["total"]
long.drop("total", axis=1, inplace=True)
long.plot(kind="bar", stacked=True, rot=0, ax=ax[1,0])
ax[1,0].set_xlabel("breast")
ax[1,0].get_legend().remove()

long = df[["class", "breast_quad"]].groupby(["breast_quad", "class"]).size().reset_index(name="count").pivot(index="breast_quad", columns="class", values="count")
long["total"] = long.sum(axis=1)
long["no-recurrence-events"] = long["no-recurrence-events"] / long["total"]
long["recurrence-events"] = long["recurrence-events"] / long["total"]
long.drop("total", axis=1, inplace=True)
long.plot(kind="bar", stacked=True, ax=ax[1,1])
ax[1,1].set_xlabel("breast quad")
ax[1,1].get_legend().remove()

long = df[["class", "deg_malig"]].groupby(["deg_malig", "class"]).size().reset_index(name="count").pivot(index="deg_malig", columns="class", values="count")
long["total"] = long.sum(axis=1)
long["no-recurrence-events"] = long["no-recurrence-events"] / long["total"]
long["recurrence-events"] = long["recurrence-events"] / long["total"]
long.drop("total", axis=1, inplace=True)
long.plot(kind="bar", stacked=True, rot=0, ax=ax[1,2])
ax[1,2].set_xlabel("deg_malig")
ax[1,2].get_legend().remove()
handles, labels = ax[1,2].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2)


lb_age = LabelEncoder()
lb_age.fit(["20-29", "30-39", "40-49", "50-59", "60-69", "70-79"])
df["age"] = lb_age.transform(df["age"])

lb_tumor = LabelEncoder()
lb_tumor.fit(["0-4", "5-9", "10-14", "15-19", "20-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54"])
df["tumor_size"] = lb_tumor.transform(df["tumor_size"])

lb_nodes = LabelEncoder()
lb_nodes.fit(["0-2", "3-5", "6-8", "9-11", "12-14", "15-17", "24-26"])
df["inv_nodes"] = lb_nodes.transform(df["inv_nodes"])

# now we look at the other columns to make sure that everything looks alright
df["class"].value_counts()
# let us convert this to factors where 1 means recurrence and 0 means no-recurrence
lb_class = LabelEncoder()
lb_class.fit(["no-recurrence-events", "recurrence-events"])
df["class"] = lb_class.transform(df["class"])

df["menopause"].value_counts()
# Since there are three classes and order is not important, we will use hot coding
hot = pd.get_dummies(df["menopause"], prefix="menopause")
# merge into data set
df = df.join(hot)

df["node_caps"].value_counts()
# now we just convert these columns to factors without thinking of the order since there is no order.
# use binary since there are only two possible values
df["node_caps"] = df["node_caps"].cat.codes
df["breast"].value_counts()
df["breast"] = df["breast"].cat.codes
df["irradiat"].value_counts()
df["irradiat"] = df["irradiat"].cat.codes
df["breast_quad"].value_counts()
# use hot coding since there are more than two classes
hot = pd.get_dummies(df["breast_quad"], prefix="breast_quad")
df = df.join(hot)

# split data set
x = df[["age", "tumor_size", "inv_nodes", "node_caps", "deg_malig", "menopause_ge40", "menopause_lt40",
        "menopause_premeno", "breast_quad_central", "breast_quad_left_low", "breast_quad_left_up",
        "breast_quad_right_low", "breast_quad_right_up", "irradiat"]]
y = df["class"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=8)

# create data frame to save results
results = pd.DataFrame({"model": object(), "accuracy": float(), "f1": float()}, index=[])

#knn
#define the parameters
leaf_size = list(range(1, 10))
n_neighbors = list(range(1, 10))
p = [1,2]
parameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)
neigh = KNeighborsClassifier()
#define the grid search
neigh_grid = GridSearchCV(neigh, parameters, cv=10, scoring="f1")
# find the best fit model
neigh_best = neigh_grid.fit(x_train, y_train)
# get the parameters of the best fit model
parameters_best = neigh_best.best_estimator_.get_params()
# fit the best fit model
neigh = KNeighborsClassifier(n_neighbors=parameters_best["n_neighbors"],
                             leaf_size=parameters_best["leaf_size"],
                             p=parameters_best["p"])
neigh.fit(x_train, y_train)
# get the predicted values
yhat_neigh = neigh.predict(x_test)
results = results.append({"model": "knn", "accuracy": metrics.accuracy_score(y_test, yhat_neigh), "f1": metrics.f1_score(y_test, yhat_neigh)}, ignore_index=True)

# decision tree
# define the parameters
max_depth = list(range(1, 10))
min_samples_split = list(range(1, 10))
min_samples_leaf = list(range(1, 5))
parameters = dict(max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
tree = DecisionTreeClassifier()
# define the grid search
tree_grid = GridSearchCV(tree, parameters, cv=10, scoring="f1")
# find the best fit model
tree_best = tree_grid.fit(x_train, y_train)
# get the parameters of the best fit model
parameters_best = tree_best.best_estimator_.get_params()
# fit the best model
tree = DecisionTreeClassifier(max_depth=parameters_best["max_depth"],
                              min_samples_split=parameters_best["min_samples_split"],
                              min_samples_leaf=parameters_best["min_samples_leaf"])
tree.fit(x_train, y_train)
yhat_tree = tree.predict(x_test)
results = results.append({"model": "tree", "accuracy": metrics.accuracy_score(y_test, yhat_tree), "f1": metrics.f1_score(y_test, yhat_tree)}, ignore_index=True)

# logistic
# define parameters
c = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
parameters = dict(C=c)
lr = LogisticRegression(max_iter=500)
# define the grid search
log_grid = GridSearchCV(lr, parameters, cv=10, scoring="f1")
# get the best fit model
log_best = log_grid.fit(x_train, y_train)
# get the parameters of the best fit model
parameters_best = log_best.best_estimator_.get_params()
# fit the best model
lr = LogisticRegression(C=parameters_best["C"], max_iter=500)
lr.fit(x_train, y_train)
yhat_log = lr.predict(x_test)
results = results.append({"model": "logistic", "accuracy": metrics.accuracy_score(y_test, yhat_log), "f1": metrics.f1_score(y_test, yhat_log)}, ignore_index=True)

#SVM
# define the parameters
kernel = ['poly', 'rbf', 'sigmoid', 'linear']
c = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
parameters = dict(kernel=kernel, C=c)
sv =svm.SVC()
# define the grid search
sv_grid = GridSearchCV(sv, parameters, cv=10, scoring="f1")
# get the best fit model
sv_best = sv_grid.fit(x_train, y_train)
# get the parameters of the best fit model
parameters_best = sv_best.best_estimator_.get_params()
# fit the best model
sv = svm.SVC(C=parameters_best["C"], kernel=parameters_best["kernel"])
sv.fit(x_train, y_train)
yhat_svm = sv.predict(x_test)
results = results.append({"model": "SVM", "accuracy": metrics.accuracy_score(y_test, yhat_svm), "f1": metrics.f1_score(y_test, yhat_svm)}, ignore_index=True)

# Random forest
# define the parameters
max_depth = [int(x) for x in np.linspace(1, 50, 10)]
max_features = [int(x) for x in linspace(1, 10, 2)]
n_estimators = [int(x) for x in np.linspace(1, 50, 5)]
parameters = dict(max_depth=max_depth, max_features=max_features, n_estimators=n_estimators)
forest = RandomForestClassifier()
# define the grid search
forest_grid = GridSearchCV(forest, parameters, cv=10, scoring="f1")
# get the best fit model
forest_best = forest_grid.fit(x_train, y_train)
# get the parameters of the best model
parameters_best = forest_best.best_estimator_.get_params()
# fit the model with the best parameters
forest = RandomForestClassifier()
forest.train(x_train, y_train)
yhat_forest = forest.predict(x_test)
results = results.append({"model": "Random forest", "accuracy": metrics.accuracy_score(y_test, yhat_forest), "f1": metrics.f1_score(y_test, yhat_forest)}, ignore_index=True)



depth = 10
accuracy = np.zeros(10-1)
f1 = np.zeros(10-1)
for d in range(1, depth):
    forest = RandomForestClassifier(max_depth=d)
    forest.fit(x_train, y_train)
    yhat_forest = forest.predict(x_test)
    accuracy[d-1] = metrics.accuracy_score(y_test, yhat_forest)
    f1[d-1] = metrics.f1_score(y_test, yhat_forest)

plt.figure()
plt.plot(range(1, depth), accuracy, '-o', label="accuracy")
plt.plot(range(1, depth), f1, '-o', label="f1")
plt.title("Random forest")
plt.legend()
# Take the model with the highest f1 score
forest = RandomForestClassifier(max_depth=f1.argmax()+1)
forest.fit(x_train, y_train)
yhat_forest = forest.predict(x_test)
results = results.append({"model": "Random forest", "accuracy": metrics.accuracy_score(y_test, yhat_forest), "f1": metrics.f1_score(y_test, yhat_forest)}, ignore_index=True)



