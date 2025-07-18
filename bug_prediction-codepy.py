# 1. Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    roc_auc_score, precision_score, recall_score, f1_score,
    precision_recall_curve, roc_curve
)
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, HistGradientBoostingClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from imblearn.over_sampling import SMOTE

# 2. Load dataset
file_path = "/content/csv_result-kc2 (1) (1).csv"  # Update if needed
df = pd.read_csv(file_path)

# 3. Preprocessing
df.rename(columns={"problems": "is_defective"}, inplace=True)
df['is_defective'] = df['is_defective'].map({'no': 0, 'yes': 1})
if 'id' in df.columns:
    df.drop(columns=['id'], inplace=True)
df.fillna(0, inplace=True)

# 4. Drop highly correlated features
drop_cols = ['total_Opnd', 'n', 'branchCount']
df.drop(columns=drop_cols, inplace=True)

# 5. Feature-target split
X = df.drop('is_defective', axis=1)
y = df['is_defective']

# 6. Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 7. SMOTE for class balance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# 8. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

# 9. Baseline model
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train, y_train)
y_dummy = dummy.predict(X_test)
print("Baseline Accuracy:", accuracy_score(y_test, y_dummy))

# 10. Define classifiers
rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
lgbm = LGBMClassifier(n_estimators=100, num_leaves=31, max_depth=5, learning_rate=0.1, random_state=42)
hgb = HistGradientBoostingClassifier(random_state=42)

# 11. Define stacking model
stacking_clf = StackingClassifier(
    estimators=[('rf', rf), ('lgbm', lgbm), ('hgb', hgb)],
    final_estimator=LogisticRegression(),
    cv=StratifiedKFold(n_splits=5)
)

# 12. Cross-validation F1 score
cv_scores = cross_val_score(stacking_clf, X_resampled, y_resampled, cv=5, scoring='f1')
print("Mean CV F1-Score:", np.mean(cv_scores))

# 13. Train stacking classifier
stacking_clf.fit(X_train, y_train)

# 14. Prediction
y_pred = stacking_clf.predict(X_test)
y_prob = stacking_clf.predict_proba(X_test)[:, 1]

# 15. Evaluation
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))

# 16. Feature importance (RF)
rf.fit(X_train, y_train)
rf_importances = rf.feature_importances_
features = X.columns
plt.figure(figsize=(10, 6))
plt.barh(features, rf_importances)
plt.xlabel("Importance")
plt.title("Feature Importance (Random Forest)")
plt.tight_layout()
plt.show()

# 17. Feature importance (LGBM)
lgbm.fit(X_train, y_train)
lgbm_importances = lgbm.feature_importances_
plt.figure(figsize=(10, 6))
plt.barh(features, lgbm_importances)
plt.xlabel("Importance")
plt.title("Feature Importance (LightGBM)")
plt.tight_layout()
plt.show()

# 18. Precision-Recall Curve
precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)
plt.plot(recall_curve, precision_curve, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.grid(True)
plt.show()

# 19. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc_score(y_test, y_prob):.2f})')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.grid(True)
plt.show()
