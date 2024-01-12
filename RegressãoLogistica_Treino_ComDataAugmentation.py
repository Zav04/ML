import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from joblib import dump
from ConfusionMatrix import plot_confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


data = pd.read_excel('ML.xlsx')


class_balance = data['Abandono'].value_counts(normalize=True)
print("Balanceamento das classes:\n", class_balance)


stats_summary = data.describe()
print("Resumo estatístico das características:\n", stats_summary)


print(data.head())


class_balance = data['Abandono'].value_counts(normalize=True)
print("Balanceamento das classes:\n", class_balance)

stats_summary = data.describe()
print("Resumo estatístico das características:\n", stats_summary)


data = data.drop(data.columns[0], axis=1)


X = data.drop('Abandono', axis=1)
y = data['Abandono']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_smote)
X_test_scaled = scaler.transform(X_test) 


log_reg = LogisticRegression(C=0.1, solver='liblinear', max_iter=100, penalty='l1')
log_reg.fit(X_train_scaled, y_train_smote)


dump(log_reg, './Modelos/modelo_LogisticRegression_ComDataAugmentation.joblib')

predictions = log_reg.predict(X_test_scaled)
print(classification_report(y_test, predictions))
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-Score: {f1}')


plot_confusion_matrix(y_test, predictions,accuracy,precision,recall,f1, title='Confusion Matrix - Logistic Regression Com Data Augmentation')
