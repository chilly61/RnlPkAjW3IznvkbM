import numpy as np


class Logistics:
    def __init__(self, learning_rate=0.01, epochs=1000, Regularization=None, lambda_val=0.1):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.regularization = Regularization  # optional
        self.lambda_val = lambda_val  # optional if regularization is True
        self.weights = []
        self.loss_history = []  # optional and recommended
        self.bias = 0

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def hypothesis(self, X):
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)

    def compute_loss(self, y_true, y_pred):
        """entropy loss"""
        m = len(y_true)
        # avoid log(0) by clipping y_pred to a small value
        y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
        loss = -np.mean(y_true * np.log(y_pred) +
                        (1 - y_true) * np.log(1 - y_pred))
        return loss

    def fit(self, X, y):
        """train the model using gradient descent"""
        X = np.array(X)
        y = np.array(y)
        m, n = X.shape

        # initialize weights randomly
        rng = np.random.RandomState(42)
        self.weights = rng.rand(n) * 0.01
        self.bias = 0.0

        for epoch in range(self.epochs):
            # forward pass
            y_pred = self.hypothesis(X)

            # compute loss
            dw = (1 / m) * np.dot(X.T, (y_pred - y))
            db = (1 / m) * np.sum(y_pred - y)

            #  update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # compute loss
            loss = self.compute_loss(y, y_pred)
            self.loss_history.append(loss)

            # print progress
            # if epoch % 500 == 0:
            # print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict_proba(self, X):
        """return probability of class 1"""
        X = np.array(X)
        return self.hypothesis(X)

    def predict(self, X, threshold=0.5):
        """return class label based on threshold"""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def add_polynomial_features(X):
        """
        add polynomial features to X
        """
        X = np.array(X)
        n_samples, n_features = X.shape

        # 计算新特征数量
        # 原始特征(n) + 平方项(n) + 交互项(n*(n-1)//2)
        n_new_features = n_features + n_features + \
            (n_features * (n_features - 1)) // 2
        X_poly = np.zeros((n_samples, n_new_features))

        current_col = 0

        # 1. 添加原始特征
        X_poly[:, current_col:current_col + n_features] = X
        current_col += n_features

        # 2. 添加平方项
        for i in range(n_features):
            X_poly[:, current_col] = X[:, i] ** 2
            current_col += 1

        # 3. 添加交互项 (X_i * X_j, where i < j)
        for i in range(n_features):
            for j in range(i + 1, n_features):
                X_poly[:, current_col] = X[:, i] * X[:, j]
                current_col += 1

        return X_poly
