import csv
import random


class DataLoader:
    @staticmethod
    def load_csv(filename, target_column, has_header=True):
        """
        Load data from CSV file
        Expected format: Y,X1,X2,X3,X4,X5,X6, target_column is the first column, so set it to 0.
        """
        X = []
        y = []

        with open(filename, 'r') as f:
            reader = csv.reader(f)
            if has_header:
                next(reader)  # Skip header

            for row_num, row in enumerate(reader):
                if not row:  # skip empty rows
                    continue

                y.append(int(row[target_column]))
                features = [float(row[i])
                            for i in range(len(row)) if i != target_column]
                X.append(features)

        return X, y

    @staticmethod
    def train_test_split(X, y, test_size=0.2, random_state=None):
        """Split data into training and testing sets"""
        if random_state is not None:
            random.seed(random_state)

        n_total = len(X)
        n_test = int(n_total * test_size)

        indices = list(range(n_total))
        random.shuffle(indices)

        test_indices = indices[:n_test]
        train_indices = indices[n_test:]

        X_train = [X[i] for i in train_indices]
        X_test = [X[i] for i in test_indices]
        y_train = [y[i] for i in train_indices]
        y_test = [y[i] for i in test_indices]

        return X_train, X_test, y_train, y_test
