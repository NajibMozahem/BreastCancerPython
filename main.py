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

# knn
k = 10
accuracy = np.zeros(k-1)
f1 = np.zeros(k-1)
for n in range(1, k):
    neigh = KNeighborsClassifier(n_neighbors=n)
    neigh.fit(x_train, y_train)
    yhat_neigh = neigh.predict(x_test)
    # do not include the neighbors = 1 because it would result in over fitting
    if n != 1:
        accuracy[n-1] = metrics.accuracy_score(y_test, yhat_neigh)
        f1[n-1] = metrics.f1_score(y_test, yhat_neigh)

plt.figure()
plt.plot(range(1, k), accuracy, '-o', label="accuracy")
plt.plot(range(1, k), f1, '-0', label="f1")
plt.title("KNN")
plt.legend()
neigh = KNeighborsClassifier(n_neighbors=f1.argmax()+1)
neigh.fit(x_train, y_train)
yhat_neigh = neigh.predict(x_test)
results = results.append({"model": "knn", "accuracy": metrics.accuracy_score(y_test, yhat_neigh), "f1": metrics.f1_score(y_test, yhat_neigh)}, ignore_index=True)

# decision tree
d = 10
accuracy = np.zeros(d-1)
f1 = np.zeros(d-1)
for depth in range(1, d):
    tree = DecisionTreeClassifier(criterion="entropy", max_depth=depth)
    tree.fit(x_train, y_train)
    yhat_tree = tree.predict(x_test)
    accuracy[depth-1] = metrics.accuracy_score(y_test, yhat_tree)
    f1[n - 1] = metrics.f1_score(y_test, yhat_tree)

plt.figure()
plt.plot(range(1, d), accuracy, '-o', label="accuracy")
plt.plot(range(1, d), f1, '-o', label="f1")
plt.title("Decision tree")
plt.legend()
# Take the model with the highest f1 score
tree = DecisionTreeClassifier(criterion="entropy", max_depth=f1.argmax()+1)
tree.fit(x_train, y_train)
yhat_tree = tree.predict(x_test)
results = results.append({"model": "tree", "accuracy": metrics.accuracy_score(y_test, yhat_tree), "f1": metrics.f1_score(y_test, yhat_tree)}, ignore_index=True)

# logistic
LR = LogisticRegression()
LR.fit(x_train, y_train)
yhat_logistic = LR.predict(x_test)
results = results.append({"model": "logistic", "accuracy": metrics.accuracy_score(y_test, yhat_logistic), "f1": metrics.f1_score(y_test, yhat_logistic)}, ignore_index=True)

#SVM
SV = svm.SVC()
SV.fit(x_train, y_train)
yhat_svm = SV.predict(x_test)
metrics.accuracy_score(y_test, yhat_svm)

# Random forest
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



