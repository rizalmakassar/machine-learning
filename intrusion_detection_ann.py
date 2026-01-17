"""Simple intrusion detection model using an artificial neural network.

This script loads the KDD Cup 99 dataset via scikit-learn, preprocesses the
features, trains a small neural network using TensorFlow/Keras, and prints a
classification report on a hold-out test set.

The dataset is converted into a binary classification problem where ``normal``
traffic is labeled ``0`` and everything else is labeled ``1`` (intrusion).
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_kddcup99
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense


def load_data():
    """Load and return the KDD Cup 99 subset as a DataFrame."""
    dataset = fetch_kddcup99(subset="SA", percent10=True, shuffle=True)

    # Decode bytes to strings and build DataFrame
    X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    X = X.applymap(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)

    y = pd.Series(dataset.target)
    y = y.apply(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)
    y = y.apply(lambda lbl: 0 if lbl == "normal." else 1)
    return X, y


def build_preprocessor(cat_features, num_features):
    """Return a ColumnTransformer for preprocessing features."""
    cat_transformer = OneHotEncoder(handle_unknown="ignore")
    num_transformer = StandardScaler()

    return ColumnTransformer(
        [
            ("cat", cat_transformer, cat_features),
            ("num", num_transformer, num_features),
        ]
    )


def build_model(input_dim: int) -> Sequential:
    """Return a compiled Keras model."""
    model = Sequential(
        [
            Dense(64, activation="relu", input_dim=input_dim),
            Dense(32, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def main():
    X, y = load_data()
    cat_features = X.select_dtypes(include="object").columns
    num_features = X.select_dtypes(exclude="object").columns

    preprocessor = build_preprocessor(cat_features, num_features)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    model = build_model(X_train_processed.shape[1])

    model.fit(
        X_train_processed,
        y_train,
        batch_size=128,
        epochs=10,
        validation_split=0.1,
        verbose=1,
    )

    predictions = (model.predict(X_test_processed) > 0.5).astype("int32").flatten()
    print(classification_report(y_test, predictions))


if __name__ == "__main__":
    main()
