from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from typing import Optional, Tuple

def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 100,
    eval_set: Optional[Tuple[pd.DataFrame, pd.Series]] = None):
    
    """Train a classifier on X_train/y_train and return (model, feature_columns).
    Optionally evaluate on eval_set=(X_val, y_val)"""
    
    X_res, y_res = X_train, y_train

    model = RandomForestClassifier(
        random_state=random_state, 
        class_weight="balanced",
        n_estimators=150,
        max_depth=6,
        min_samples_leaf=3)
    
    model.fit(X_res, y_res)

    # Optional evaluation on a held-out set (no leakage)
    if eval_set is not None:
        X_val, y_val = eval_set
        y_pred = model.predict(X_val)
        print("\nConfusion Matrix (validation):")
        print(confusion_matrix(y_val, y_pred))
        print("\nClassification Report (validation):")
        print(classification_report(y_val, y_pred, zero_division=0))

    return model, list(X_train.columns)
