import os
import glob
import zipfile
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

RANDOM_STATE = 42


def read_parts_from_extracted_dir(extract_dir: str) -> pd.DataFrame:
    files = glob.glob(os.path.join(extract_dir, "**", "data_part_*.*"), recursive=True)
    if not files:
        raise FileNotFoundError

    dfs = []
    for f in sorted(files):
        ext = os.path.splitext(f)[1].lower()

        if ext == ".csv":
            df = pd.read_csv(f)
        elif ext in (".xlsx", ".xls"):
            df = pd.read_excel(f)
        elif ext == ".parquet":
            try:
                df = pd.read_parquet(f)
            except Exception:
                import pyarrow.parquet as pq

                df = pq.read_table(f).to_pandas()
        else:
            continue

        dfs.append(df)

    train = pd.concat(dfs, ignore_index=True, sort=False)
    return train


def make_submission_files(pred: np.ndarray, out_csv: str = "submission.csv", out_zip: str = "submission.zip") -> None:
    sub = pd.DataFrame({"proba": pred.astype(float)})
    sub.to_csv(out_csv, index=False)

    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(out_csv, arcname="submission.csv")
        z.write(out_csv, arcname="submisson.csv")


def main():
    zip_path = "train_data.zip"
    test_path = "X_test.csv"

    extract_dir = "train_data_extracted"
    os.makedirs(extract_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)

    train = read_parts_from_extracted_dir(extract_dir)
    test = pd.read_csv(test_path)

    target_col = "Увольнение"

    X = train.drop(columns=[target_col])
    y = train[target_col].astype(int)

    common_cols = [c for c in X.columns if c in test.columns]
    X = X[common_cols]
    test = test[common_cols]

    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    for c in cat_cols:
        X[c] = X[c].astype("category")
        test[c] = test[c].astype("category")

    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    oof = np.zeros(len(X))
    test_pred = np.zeros(len(test))

    params = dict(
        n_estimators=5000,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        min_child_samples=20,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    for tr_idx, va_idx in skf.split(X, y):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        model = lgb.LGBMClassifier(**params)

        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(200, verbose=False)],
        )

        oof[va_idx] = model.predict_proba(X_va)[:, 1]
        test_pred += model.predict_proba(test)[:, 1] / skf.n_splits

    make_submission_files(test_pred, out_csv="submission.csv", out_zip="submission.zip")


if __name__ == "__main__":
    main()
