import json, time
import pandas as pd, lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score

t0 = time.time()
df = pd.read_csv("/home/ubuntu/ml-benchmark/creditcard.csv")
load_time = time.time() - t0
print(f"Loaded {df.shape} in {load_time:.2f}s")

X, y = df.drop(columns=["Class"]), df["Class"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_tr, X_val, y_tr, y_val = train_test_split(X_tr, y_tr, test_size=0.1, stratify=y_tr, random_state=42)

model = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.05, num_leaves=31, n_jobs=-1, verbose=-1, metric="auc", min_child_samples=50, reg_lambda=1.0, subsample=0.8, subsample_freq=1, colsample_bytree=0.8)
t0 = time.time()
model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric="auc",
          callbacks=[lgb.early_stopping(100, first_metric_only=True), lgb.log_evaluation(100)])
train_time = time.time() - t0

proba = model.predict_proba(X_te)[:, 1]
pred = (proba >= 0.5).astype(int)

row = X_te.iloc[[0]]
for _ in range(10): model.predict_proba(row)
t0 = time.perf_counter()
for _ in range(100): model.predict_proba(row)
latency_ms = (time.perf_counter() - t0) / 100 * 1000

batch = X_te.iloc[:1000]
t0 = time.perf_counter()
model.predict_proba(batch)
batch_s = time.perf_counter() - t0

result = {
    "instance": "t3.medium (2 vCPU, 4GB RAM)",
    "data_load_time_s": round(load_time, 3),
    "training_time_s": round(train_time, 3),
    "best_iteration": int(model.best_iteration_ or model.n_estimators),
    "auc_roc": round(roc_auc_score(y_te, proba), 6),
    "accuracy": round(accuracy_score(y_te, pred), 6),
    "f1_score": round(f1_score(y_te, pred), 6),
    "precision": round(precision_score(y_te, pred), 6),
    "recall": round(recall_score(y_te, pred), 6),
    "inference_latency_1row_ms": round(latency_ms, 3),
    "inference_1000rows_s": round(batch_s, 4),
    "throughput_rows_per_s": round(1000 / batch_s, 1),
}
print(json.dumps(result, indent=2))
with open("/home/ubuntu/ml-benchmark/benchmark_result.json", "w") as f:
    json.dump(result, f, indent=2)
print("Saved benchmark_result.json")
