from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from linear_regression import LinearRegression
import numpy as np

np.set_printoptions(suppress=True)

x, y = make_regression(n_samples=10000, n_features=20, n_informative=15, noise=2, random_state=42)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print(model.accuracy(y_test, y_pred))
