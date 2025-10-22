from Data_Loader import DataLoader
from Logistics import Logistics
import numpy as np


def main():
    # 1. Load Data
    X, y = DataLoader.load_csv(
        'ACME-HappinessSurvey2020.csv', target_column=0, has_header=True)
    X = np.array(X)
    y = np.array(y)

    X_standardized = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    X_poly = Logistics.add_polynomial_features(X_standardized)

    learning_rates = [0.004]
    for lr in learning_rates:
        print(f"\ncurrent learning rate: {lr}")
        print("-" * 30)
        # 2. Create Model
        model = Logistics(learning_rate=lr, epochs=10000)

        # 3. Train Model
        model.fit(X_poly, y)

        # 4. Predict
        predictions = model.predict(X_poly)

        # 5. Evaluate
        accuracy = sum(1 for i in range(len(y))
                       if y[i] == predictions[i]) / len(y)
        print(f"accuracy: {accuracy:.4f}")
        print(f"final loss: {model.loss_history[-1]:.4f}")


if __name__ == "__main__":
    main()
