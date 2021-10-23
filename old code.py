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