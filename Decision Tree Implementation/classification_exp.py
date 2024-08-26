import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Tree.Base import DecisionTree
from metrics import precision as precision_metric, recall as recall_metric, accuracy as accuracy_metric
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, KFold
from scipy import stats

np.random.seed(42)

# Generate dataset
X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)

plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()

X_df = pd.DataFrame(X, columns=['Feature 1', 'Feature 2'])
y_series = pd.Series(y, name='Target')

X_train, X_test, y_train, y_test = train_test_split(X_df, y_series, test_size=0.3, random_state=42)

tree = DecisionTree(criterion='information_gain', max_depth=3)
tree.fit(X_train, y_train)

y_pred = tree.predict(X_test)

for cls in y_test.unique():
    precision_score = precision_metric(y_test, y_pred, cls)
    recall_score = recall_metric(y_test, y_pred, cls)
    print(f'Precision for class {cls}: {precision_score:.4f}')
    print(f'Recall for class {cls}: {recall_score:.4f}')

accuracy = accuracy_metric(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')

x_min, x_max = X_test['Feature 1'].min() - 1, X_test['Feature 1'].max() + 1
y_min, y_max = X_test['Feature 2'].min() - 1, X_test['Feature 2'].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

Z = tree.predict(pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=['Feature 1', 'Feature 2']))
Z = Z.values.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.8)
plt.scatter(X_test['Feature 1'], X_test['Feature 2'], c=y_test, edgecolors='k', marker='o')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Tree Decision Boundaries')
plt.show()


depths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)

best_depths = []
accuracies = []

for train_index, test_index in outer_cv.split(X_df):
    X_train, X_test = X_df.iloc[train_index], X_df.iloc[test_index]
    y_train, y_test = y_series.iloc[train_index], y_series.iloc[test_index]

    best_depth = None
    best_accuracy = -float('inf')

    inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)

    for depth in depths:
        accuracies_inner = []

        for inner_train_index, inner_test_index in inner_cv.split(X_train):
            X_inner_train, X_inner_test = X_train.iloc[inner_train_index], X_train.iloc[inner_test_index]
            y_inner_train, y_inner_test = y_train.iloc[inner_train_index], y_train.iloc[inner_test_index]

            tree = DecisionTree(criterion='information_gain', max_depth=depth)
            tree.fit(X_inner_train, y_inner_train)

            y_pred_inner = tree.predict(X_inner_test)
            accuracy_inner = accuracy_metric(y_inner_test, y_pred_inner)
            accuracies_inner.append(accuracy_inner)

        mean_accuracy = np.mean(accuracies_inner)

        if mean_accuracy > best_accuracy:
            best_accuracy = mean_accuracy
            best_depth = depth

    final_tree = DecisionTree(criterion='information_gain', max_depth=best_depth)
    final_tree.fit(X_train, y_train)

    y_pred = final_tree.predict(X_test)
    accuracy = accuracy_metric(y_test, y_pred)
    accuracies.append(accuracy)
    best_depths.append(best_depth)
    
print("\n")
print(f'Best depths from each outer fold: {stats.mode(best_depths)[0]}')
print(f'Average accuracy across outer folds: {np.mean(accuracies)}')
