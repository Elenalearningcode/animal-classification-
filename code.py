"""
Mini project: Phân loại động vật dựa trên đặc điểm sinh học
Dữ liệu: zoo.csv + class.csv (Zoo dataset)
Chạy:    python zoo_classification.py --data_dir .
"""
import argparse
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, f1_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42


# ----------------------------------------------------------------------------
# 1. LOAD & MERGE
# ----------------------------------------------------------------------------
def load_data(data_dir: Path) -> pd.DataFrame:
    zoo = pd.read_csv(data_dir / "zoo.csv")
    cls = pd.read_csv(data_dir / "class.csv")

    # chuẩn hoá tên cột cho đỡ lỗi hoa/thường, khoảng trắng
    zoo.columns = zoo.columns.str.strip().str.lower()
    cls.columns = cls.columns.str.strip().str.lower()

    # zoo.class_type  <->  class.class_number
    cls = cls[["class_number", "class_type"]].rename(
        columns={"class_type": "class_name"})
    df = zoo.merge(cls, left_on="class_type", right_on="class_number", how="left")
    df = df.drop(columns="class_number")
    return df


# ----------------------------------------------------------------------------
# 2. EDA + LÀM SẠCH
# ----------------------------------------------------------------------------
def explore(df: pd.DataFrame, out_dir: Path) -> None:
    print("=" * 60, "\nKÍCH THƯỚC:", df.shape)
    print("\nKIỂU DỮ LIỆU:\n", df.dtypes)
    print("\nGIÁ TRỊ THIẾU:\n", df.isna().sum()[lambda s: s > 0])
    print("Số dòng trùng:", df.duplicated().sum())
    print("\nPHÂN BỐ LỚP:\n", df["class_name"].value_counts())

    fig, ax = plt.subplots(figsize=(7, 4))
    df["class_name"].value_counts().plot.bar(ax=ax, color="steelblue")
    ax.set_title("Phân bố số lượng loài theo lớp")
    ax.set_ylabel("Số loài")
    plt.tight_layout()
    fig.savefig(out_dir / "class_distribution.png", dpi=120)
    plt.close(fig)


def preprocess(df: pd.DataFrame):
    df = df.copy()
    # bỏ dòng thiếu nhãn (nếu merge lỗi) và dòng trùng
    df = df.dropna(subset=["class_name"]).drop_duplicates()

    names = df["animal_name"]
    y = df["class_name"]
    X = df.drop(columns=["animal_name", "class_type", "class_name"])

    # 15 cột nhị phân 0/1 + 'legs' (số chân: 0,2,4,5,6,8)
    # one-hot cho 'legs' vì nó mang tính phân loại hơn là thứ tự
    X = pd.get_dummies(X, columns=["legs"], prefix="legs", dtype=int)
    return X, y, names


# ----------------------------------------------------------------------------
# 3. HUẤN LUYỆN & SO SÁNH MÔ HÌNH
# ----------------------------------------------------------------------------
def build_models() -> dict:
    return {
        "LogisticRegression": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))]),
        "KNN": Pipeline([
            ("scale", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=3))]),
        "DecisionTree": DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(n_estimators=200,
                                               random_state=RANDOM_STATE),
    }


def main(data_dir: str) -> None:
    data_dir = Path(data_dir)
    out_dir = data_dir / "outputs"
    out_dir.mkdir(exist_ok=True)

    df = load_data(data_dir)
    explore(df, out_dir)
    X, y, names = preprocess(df)
    print("\nSố đặc trưng sau xử lý:", X.shape[1])

    # chia train/test có phân tầng (giữ tỉ lệ các lớp)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    print(f"Train: {len(X_tr)} | Test: {len(X_te)}")

    # cross-validation (3 fold vì sau khi chia train, lớp nhỏ nhất chỉ còn ~3 loài)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    models = build_models()
    results = []
    print("\n" + "=" * 60 + "\nSO SÁNH MÔ HÌNH")
    for name, model in models.items():
        cv_scores = cross_val_score(model, X_tr, y_tr, cv=cv, scoring="accuracy")
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        results.append({
            "model": name,
            "cv_acc_mean": cv_scores.mean(),
            "cv_acc_std": cv_scores.std(),
            "test_acc": accuracy_score(y_te, pred),
            "test_f1_macro": f1_score(y_te, pred, average="macro"),
        })
    res = pd.DataFrame(results).sort_values("test_acc", ascending=False)
    print(res.round(3).to_string(index=False))
    res.to_csv(out_dir / "model_comparison.csv", index=False)

    # chọn mô hình tốt nhất theo CV (tránh chọn theo test set)
    best_name = res.sort_values("cv_acc_mean", ascending=False).iloc[0]["model"]
    best = models[best_name]
    pred = best.predict(X_te)
    print(f"\nMô hình được chọn: {best_name}")
    print(classification_report(y_te, pred, zero_division=0))

    # confusion matrix
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_te, pred, ax=ax, xticks_rotation=45, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion matrix - {best_name}")
    plt.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=120)
    plt.close(fig)

    # độ quan trọng đặc trưng (Random Forest)
    rf = models["RandomForest"]
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values()
    fig, ax = plt.subplots(figsize=(7, 6))
    imp.tail(12).plot.barh(ax=ax, color="darkorange")
    ax.set_title("Top đặc trưng quan trọng (Random Forest)")
    plt.tight_layout()
    fig.savefig(out_dir / "feature_importance.png", dpi=120)
    plt.close(fig)

    # liệt kê các loài bị dự đoán sai
    wrong = X_te.index[pred != y_te.values]
    if len(wrong):
        err = pd.DataFrame({"animal": names.loc[wrong],
                            "thực tế": y_te.loc[wrong],
                            "dự đoán": pred[pred != y_te.values]})
        print("\nCác loài dự đoán sai:\n", err.to_string(index=False))

    # lưu mô hình + danh sách cột để dùng lại
    joblib.dump({"model": best, "columns": list(X.columns)},
                out_dir / "best_model.joblib")

    # demo dự đoán 1 con vật mới (ví dụ: một loài chim)
    demo = {c: 0 for c in X.columns}
    demo.update({"feathers": 1, "eggs": 1, "backbone": 1, "breathes": 1,
                 "airborne": 1, "tail": 1})
    if "legs_2" in demo:
        demo["legs_2"] = 1
    print("\nDemo - đặc điểm: có lông vũ, đẻ trứng, biết bay, 2 chân")
    print("=> Dự đoán:", best.predict(pd.DataFrame([demo])[X.columns])[0])
    print(f"\nKết quả đã lưu trong: {out_dir.resolve()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=".", help="Thư mục chứa zoo.csv và class.csv")
    main(ap.parse_args().data_dir)