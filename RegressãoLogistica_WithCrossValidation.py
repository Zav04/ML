import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from joblib import dump
from ConfusionMatrix import plot_confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix



data = pd.read_excel('ML.xlsx')


class_balance = data['Abandono'].value_counts(normalize=True)
print("Balanceamento das classes:\n", class_balance)


stats_summary = data.describe()
print("Resumo estatístico das características:\n", stats_summary)

X = data.drop('Abandono', axis=1)
y = data['Abandono']


# correlation_matrix = data.corr()


# correlation_with_target = correlation_matrix['Abandono'].sort_values(ascending=False)
# print(correlation_with_target)



# low_correlation_threshold = 0.0001
# high_correlation_threshold = 0.9 


# low_correlation_features = correlation_with_target[abs(correlation_with_target) < low_correlation_threshold].index


# high_correlation_pairs = [(i, j) for i in correlation_matrix.columns for j in correlation_matrix.columns if i != j and abs(correlation_matrix.loc[i, j]) > high_correlation_threshold]


# features_to_remove = list(low_correlation_features)

# data = data.drop(columns=features_to_remove)



class_0 = data[data['Abandono'] == 0]
class_1 = data[data['Abandono'] == 1]


class_0_sample = class_0.sample(500, random_state=42)  
class_1_sample = class_1.sample(500, random_state=42)  


balanced_data = pd.concat([class_0_sample, class_1_sample])


balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)


X_balanced = balanced_data.drop('Abandono', axis=1)
y_balanced = balanced_data['Abandono']

X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.1, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) 



log_reg = LogisticRegression(C=0.1, solver='liblinear', max_iter=100, penalty='l1')
log_reg.fit(X_train_scaled, y_train) 


dump(log_reg, './Modelos/modelo_LogisticRegression.joblib')


y_pred_cv = cross_val_predict(log_reg, X_balanced, y_balanced, cv=10)

conf_matrix = confusion_matrix(y_balanced, y_pred_cv)


accuracy_cv = accuracy_score(y_balanced, y_pred_cv)
precision_cv = precision_score(y_balanced, y_pred_cv)
recall_cv = recall_score(y_balanced, y_pred_cv)
f1_cv = f1_score(y_balanced, y_pred_cv)



plot_confusion_matrix(y_balanced, y_pred_cv, accuracy_cv, precision_cv, recall_cv, f1_cv, title='Matriz de Confusão - Logistic Regression (CV)')


fp_filter = (y_pred_cv == 1) & (y_balanced == 0) 
fn_filter = (y_pred_cv == 0) & (y_balanced == 1) 


falsos_positivos = data.loc[y_balanced[fp_filter].index]
falsos_negativos = data.loc[y_balanced[fn_filter].index]

# Agora vamos escrever esses DataFrames para arquivos CSV separados
falsos_positivos.to_csv('LogisticRegressionfalsos_positivos.csv', sep=';', index=True)
falsos_negativos.to_csv('LogisticRegressionfalsos_negativos.csv', sep=';', index=True)


train_sizes, train_scores, test_scores = learning_curve(
    estimator=log_reg,
    X=X_balanced,
    y=y_balanced,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5,
    n_jobs=-1,
    scoring='accuracy'
)


train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)


plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, test_mean, 'o-', color="g", label="Cross-validation score")

plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="g")

plt.title("Cross-Validation - Logistic Regression")
plt.xlabel("Tamanho do Conjunto de Treino")
plt.ylabel("Accuracy")
plt.legend(loc="best")
plt.grid(True)
plt.show()
