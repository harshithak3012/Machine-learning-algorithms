import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class Linear_Regression():
    def __init__(self, learning_rate, no_of_iterations):
        self.learning_rate = learning_rate
        self.no_of_iterations = no_of_iterations

    def fit(self, X, Y):
        self.m, self.n = X.shape
        self.w = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.Y = Y
        for i in range(self.no_of_iterations):
            self.update_weights()

    def update_weights(self):
        Y_predict = self.predict(self.X)
        dw = -(2 * (self.X.T).dot(self.Y - Y_predict)) / self.m
        db = -2 * np.sum(self.Y - Y_predict) / self.m
        self.w = self.w - self.learning_rate * dw
        self.b = self.b - self.learning_rate * db

    def predict(self, X):
        return X.dot(self.w) + self.b

salary_data = pd.read_csv("D:\Harshitha docs\machine learning\salary_data.csv")
X = salary_data.iloc[:, :-1].values
Y = salary_data.iloc[:, 1].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=2)

model = Linear_Regression(learning_rate=0.02, no_of_iterations=1000)
model.fit(X_train, Y_train)

print('weight = ', model.w[0])
print('bias = ', model.b)

test_data_predict = model.predict(X_test)
print(test_data_predict)

plt.scatter(X_test, Y_test, color='red')
plt.plot(X_test, test_data_predict, color='blue')
plt.xlabel('Work Experience')
plt.ylabel('Salary')
plt.title('Salary vs Experience')
plt.show()
