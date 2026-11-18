"""Data loading, feature selection, and model training module."""
from __future__ import annotations
import os
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score

# Silence loky core counting warning on Windows
os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 4)


class ExperimentSetup:
    """Encapsulates dataset split, feature selection, trained model, background set, and explicands."""
    def __init__(self, d: int = 12, n_explicands: int = 10):
        self.d = d
        self.n_explicands = n_explicands
        
        # Load dataset
        X, y = load_breast_cancer(return_X_y=True)
        self.feature_names_all = load_breast_cancer().feature_names
        
        # 70/30 stratified train/test split
        self.X_train_raw, self.X_test_raw, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.30, random_state=0, stratify=y
        )
        
        # Mutual information on train split only
        self.mi_scores = mutual_info_classif(self.X_train_raw, self.y_train, random_state=0)
        self.top_d_indices = np.argsort(self.mi_scores)[::-1][:d]
        self.feature_names = [self.feature_names_all[i] for i in self.top_d_indices]
        
        # Subset top-d features
        self.X_train = self.X_train_raw[:, self.top_d_indices]
        self.X_test = self.X_test_raw[:, self.top_d_indices]
        
        # Train HistGradientBoostingClassifier
        self.model = HistGradientBoostingClassifier(max_iter=200, max_depth=4, random_state=0)
        self.model.fit(self.X_train, self.y_train)
        
        # Evaluate test accuracy
        y_pred = self.model.predict(self.X_test)
        self.test_accuracy = accuracy_score(self.y_test, y_pred)
        
        # Background: 100 training points chosen once with default_rng(0)
        bg_rng = np.random.default_rng(0)
        self.bg_indices = bg_rng.choice(len(self.X_train), size=100, replace=False)
        self.background = self.X_train[self.bg_indices]
        
        # Explicands: n test instances chosen once with default_rng(0)
        test_rng = np.random.default_rng(0)
        self.explicand_indices = test_rng.choice(len(self.X_test), size=n_explicands, replace=False)
        self.explicands = self.X_test[self.explicand_indices]
        self.explicand_test_indices = self.explicand_indices
