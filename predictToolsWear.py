import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, cross_val_score
import matplotlib.pyplot as plt
import joblib
from flask import Flask, request, jsonify
import numpy as np
from ucimlrepo import fetch_ucirepo

# Fetch the dataset
dataset = fetch_ucirepo(id=601)
X = dataset.data.features
y = dataset.data.targets['Machine failure']

# Dataset metadata and variables information
print(dataset.metadata)
print(dataset.variables)

print(X.isnull().sum())
X = X.dropna()  # Drop missing values if any

# Scale numerical features and split the dataset
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
logistic_model = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42)

# Train the model
logistic_model.fit(X_train, y_train)
y_pred = logistic_model.predict(X_test)

print('Accuracy:', accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:\n', cm)

# ROC Curve
y_pred_proba = logistic_model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

param_grid = {
    'C': [0.1, 1, 10, 100],
    'solver': ['lbfgs', 'liblinear']
}

# Perform Grid Search
grid_search = GridSearchCV(estimator=logistic_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print('Best Parameters:', best_params)

# Cross-validate the best model
scores = cross_val_score(best_model, X_scaled, y, cv=5)
print('Cross-validation scores:', scores)
print('Mean cross-validation score:', scores.mean())

joblib.dump(best_model, 'logistic_model.pkl')

# Flask
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = np.array(data['features']).reshape(1, -1)
    prediction = best_model.predict(features)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(port=5000, debug=True)
