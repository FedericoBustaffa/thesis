import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

X, y = datasets.make_classification(
    n_samples=100, n_features=2, n_redundant=0, n_classes=2
)

X = np.array(X).T
y = np.array(y)


plt.figure(figsize=(16, 9))
plt.set_cmap(plt.get_cmap("bwr"))
plt.scatter(x=X[0], y=X[1], c=y)
plt.show()
