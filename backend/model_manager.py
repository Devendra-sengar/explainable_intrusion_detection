import os
import joblib
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import logging

LOG = logging.getLogger(__name__)

class ModelManager:
    def __init__(self, model_path: str, data_path: Optional[str] = None, batch_threshold: int = 50):
        self.model_path = model_path
        self.data_path = data_path
        self.batch_threshold = batch_threshold
        self.adaptive_queue: List[Dict] = []
        self.last_metrics = {}
        self._model = None
        self._label_encoder = None
        self.feature_names = None

        if os.path.exists(self.model_path):
            self._load_model()
            LOG.info(f"Loaded model from {self.model_path}")
        else:
            if data_path and os.path.exists(data_path):
                LOG.info("Model not found; training initial model from provided data path")
                self._train_initial_model()
            else:
                raise FileNotFoundError(f"Model not found at {self.model_path} and no data available to train initial model.")

    def _load_model(self):
        data = joblib.load(self.model_path)
        self.pipeline = data["pipeline"]
        self._label_encoder = data["label_encoder"]
        self.feature_names = data.get("feature_names")
        self.last_metrics = data.get("last_metrics", {})

    def _save_model(self):
        data = {
            "pipeline": self.pipeline,
            "label_encoder": self._label_encoder,
            "feature_names": self.feature_names,
            "last_metrics": self.last_metrics,
        }
        joblib.dump(data, self.model_path)
        LOG.info(f"Saved updated model to {self.model_path}")

    def _detect_label_column(self, df: pd.DataFrame) -> str:
        candidates = ["Label", "label", "Class", "class", "attack", "Attack", "true_label"]
        for c in candidates:
            if c in df.columns:
                return c
        return df.columns[-1]

    def _train_initial_model(self):
        df = pd.read_csv(self.data_path)
        label_col = self._detect_label_column(df)
        y = df[label_col]
        X = df.drop(columns=[label_col])
        X = X.select_dtypes(include=[np.number])
        self.feature_names = list(X.columns)

        self._label_encoder = LabelEncoder()
        y_enc = self._label_encoder.fit_transform(y.astype(str))

        xgb = XGBClassifier(
            use_label_encoder=False,
            eval_metric='mlogloss',
            n_estimators=100
        )
        self.pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", xgb),
        ])

        self.pipeline.fit(X.values, y_enc)
        
        # Compute initial metrics
        X_train, X_test, y_train, y_test = train_test_split(
            X.values, y_enc, test_size=0.2, random_state=42, stratify=y_enc
        )
        y_pred = self.pipeline.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred).tolist()
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted', zero_division=0
        )
        self.last_metrics = {
            "accuracy": acc,
            "confusion_matrix": cm,
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
        }
        self._save_model()

    def predict(self, sample: Any) -> Dict[str, Any]:
        if isinstance(sample, dict):
            if not self.feature_names:
                raise ValueError("Model has no feature names metadata; cannot accept dict samples")
            x = [sample.get(fn, np.nan) for fn in self.feature_names]
        elif isinstance(sample, (list, tuple, np.ndarray)):
            x = list(sample)
        else:
            raise ValueError("Sample must be a dict or list of numeric feature values")

        X = np.array(x, dtype=float).reshape(1, -1)
        probs = self.pipeline.predict_proba(X)[0]
        idx = int(np.argmax(probs))
        class_label = self._label_encoder.inverse_transform([idx])[0]
        confidence = float(probs[idx])
        
        return {
            "predicted_class": idx,
            "class_name": str(class_label),
            "confidence": confidence
        }

    def add_adaptive_batch(self, samples: List[Dict], force_trigger: bool = False) -> Dict:
        added = 0
        for s in samples:
            try:
                pred = self.predict(s["features"])
                pred_name = pred["class_name"]
                true_name = s["true_label"]
                if str(pred_name) != str(true_name):
                    self.adaptive_queue.append(s)
                    added += 1
            except Exception as e:
                LOG.warning(f"Skipping sample due to error: {e}")

        triggered = False
        update_info = {}
        if force_trigger or len(self.adaptive_queue) >= self.batch_threshold:
            triggered = True
            update_info = self._perform_adaptive_update()

        return {
            "queued_samples": len(self.adaptive_queue),
            "added_from_request": added,
            "adaptive_update_triggered": triggered,
            **update_info,
        }

    def _perform_adaptive_update(self) -> Dict:
        if len(self.adaptive_queue) == 0:
            return {"message": "No queued samples to update."}

        # Prepare new data
        df_new = pd.DataFrame([{
            **({k: v for k, v in (s["features"].items())} if isinstance(s["features"], dict)
               else {i: v for i, v in enumerate(s["features"])}),
            "__label__": s["true_label"]
        } for s in self.adaptive_queue])

        # Ensure columns align to feature_names
        if self.feature_names is not None:
            if set(self.feature_names).issubset(set(df_new.columns)):
                X_new = df_new[self.feature_names].select_dtypes(include=[np.number])
            else:
                X_new = df_new.drop(columns=["__label__"]).select_dtypes(include=[np.number])
        else:
            X_new = df_new.drop(columns=["__label__"]).select_dtypes(include=[np.number])

        y_new = df_new["__label__"].values

        # If some classes missing in this batch, try to augment from original dataset
        missing_classes = set(self._label_encoder.classes_) - set(map(str, y_new))
        augment_rows = []
        if missing_classes and self.data_path and os.path.exists(self.data_path):
            df_full = pd.read_csv(self.data_path)
            label_col = self._detect_label_column(df_full)
            for mc in missing_classes:
                sub = df_full[df_full[label_col].astype(str) == str(mc)]
                if len(sub) > 0:
                    augment_rows.append(sub.sample(n=1, random_state=42))

        if augment_rows:
            df_aug = pd.concat(augment_rows, ignore_index=True)
            y_aug = df_aug[label_col].values
            X_aug = df_aug.drop(columns=[label_col]).select_dtypes(include=[np.number])
            if self.feature_names and list(X_aug.columns) != self.feature_names:
                X_aug = X_aug.reindex(columns=self.feature_names, fill_value=0)
            X_comb = pd.concat([pd.DataFrame(X_new, columns=X_new.columns), X_aug], ignore_index=True).fillna(0)
            y_comb = np.concatenate([y_new, y_aug])
        else:
            X_comb = pd.DataFrame(X_new, columns=X_new.columns).fillna(0)
            y_comb = y_new

        # Limit training size to avoid overfitting
        max_samples = 2000
        if len(y_comb) > max_samples and self.data_path and os.path.exists(self.data_path):
            df_all = pd.read_csv(self.data_path)
            label_col = self._detect_label_column(df_all)
            X_base = df_all.drop(columns=[label_col]).select_dtypes(include=[np.number])
            y_base = df_all[label_col]
            df_base = pd.concat([X_base, y_base], axis=1)
            df_sampled = df_base.groupby(label_col, group_keys=False).apply(
                lambda x: x.sample(
                    min(len(x), max_samples // max(1, len(self._label_encoder.classes_))),
                    random_state=42
                )
            )
            y_base_s = df_sampled[label_col].values
            X_base_s = df_sampled.drop(columns=[label_col])
            X_total = pd.concat([X_base_s.reset_index(drop=True), X_comb.reset_index(drop=True)], ignore_index=True).fillna(0)
            y_total = np.concatenate([y_base_s, y_comb])
        else:
            X_total = X_comb
            y_total = y_comb

        # Encode labels using existing encoder
        y_total_enc = self._label_encoder.transform(list(map(str, y_total)))

        # Compute metrics before retrain
        X_train, X_test, y_train, y_test = train_test_split(
            X_total.values, y_total_enc, test_size=0.2, random_state=42, stratify=y_total_enc
        )
        old_preds = self.pipeline.predict(X_test)
        old_acc = float(accuracy_score(y_test, old_preds))

        # Retrain pipeline
        xgb = XGBClassifier(
            use_label_encoder=False,
            eval_metric='mlogloss',
            n_estimators=100
        )
        new_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", xgb),
        ])
        new_pipeline.fit(X_train, y_train)

        new_preds = new_pipeline.predict(X_test)
        new_acc = float(accuracy_score(y_test, new_preds))
        cm = confusion_matrix(y_test, new_preds).tolist()
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, new_preds, average='weighted', zero_division=0
        )

        # Replace pipeline with new one and save
        self.pipeline = new_pipeline
        self.last_metrics = {
            "accuracy": new_acc,
            "confusion_matrix": cm,
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "accuracy_before_update": old_acc,
            "samples_used_for_update": int(len(y_total)),
        }
        self._save_model()

        # Clear queue
        num_corrected = len(self.adaptive_queue)
        self.adaptive_queue = []

        LOG.info(f"Adaptive update performed: corrected_samples={num_corrected}, accuracy_before={old_acc}, accuracy_after={new_acc}")

        return {
            "corrected_samples": num_corrected,
            "accuracy_before": old_acc,
            "accuracy_after": new_acc,
            "confusion_matrix": cm,
        }

    def get_metrics(self) -> Dict:
        return self.last_metrics or {}