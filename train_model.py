import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from xgboost import XGBRegressor


INPUT_FILE = "training_dataset.csv"
MODEL_FILE = "rainfall_model.pkl"


def load_data():

    df = pd.read_csv(INPUT_FILE)

    drop_columns = [
        "timestamp",
        "river",
        "zone",
        "rainfall_next_hour",
        "rainfall_class"
    ]

    X = df.drop(columns=drop_columns)

    X = pd.get_dummies(
        X,
        columns=[],
        drop_first=True
    )

    y = df["rainfall_next_hour"]

    return X, y


def train_model(X_train, y_train):

    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model


def evaluate_model(
    model,
    X_test,
    y_test
):

    predictions = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    mse = mean_squared_error(
        y_test,
        predictions
    )

    rmse = mse ** 0.5

    r2 = r2_score(
        y_test,
        predictions
    )

    print("\nModel Performance")
    print("-" * 40)

    print(f"MAE  : {mae:.3f}")
    print(f"MSE  : {mse:.3f}")
    print(f"RMSE : {rmse:.3f}")
    print(f"R²   : {r2:.3f}")


def save_model(model):

    joblib.dump(
        model,
        MODEL_FILE
    )

    print(
        f"\nModel saved as {MODEL_FILE}"
    )


def main():

    print("Loading dataset...")

    X, y = load_data()

    print(
        f"Dataset size: {len(X)} rows"
    )

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )
    )

    print("Training model...")

    model = train_model(
        X_train,
        y_train
    )

    evaluate_model(
        model,
        X_test,
        y_test
    )

    save_model(model)


if __name__ == "__main__":
    main()
