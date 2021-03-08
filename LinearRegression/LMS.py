import numpy as np
import random
import matplotlib.pyplot as plt


"""
Batch gradient descent:
X, y: input data and label
l_rate: learning rate
ep: converge tolerance
max_iter: maximum iteration
"""
def batch_gradient_descent(X, y, l_rate, ep=1e-06, max_iter=10000):
    w = np.zeros(X.shape[1])
    costs = []

    for t in range(max_iter):
        gradient = np.zeros(X.shape[1])

        for i in range(X.shape[0]):
            gradient += calc_gradient(w, X[i], y[i])

        w -= l_rate * gradient
        # w /= np.linalg.norm(w)
        cost = calc_cost(w, X, y)
        costs.append(cost)

        if np.linalg.norm(l_rate * gradient) < ep:
            print('GD converged on iter ', t, 'cost: ', cost)
            break

    return w, costs


"""
Stochastic gradient descent:
X, y: input data and label
l_rate: learning rate
ep: converge tolerance
max_iter: maximum iteration
"""
def stochastic_gradient_descent(X, y, l_rate, ep=1e-06, max_iter=100000):
    w = np.zeros(X.shape[1])
    costs = []

    for t in range(max_iter):
        i = random.randrange(X.shape[0])

        gradient = calc_gradient(w, X[i], y[i])

        w -= l_rate * gradient
        cost = calc_cost(w, X, y)
        costs.append(cost)

        if np.linalg.norm(l_rate * gradient) < ep:
            print('SGD converged on iter ', t, 'cost: ', cost)
            break

    return w, costs


"""
helper function to calculate cost
"""
def calc_cost(w, X, y):
    cost = 0
    for i in range(len(y)):
        cost += (y[i] - np.dot(w, X[i]))**2
    cost /=2
    return cost


"""
helper function to calculate gradient
"""
def calc_gradient(w, X, y):
    return -(y - np.dot(w, X)) * X


"""
to plot cost each iteration
"""
def plot_cost(costs, title):
    trials = [i+1 for i in range(len(costs))]
    plt.plot(trials, costs)
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.show()


