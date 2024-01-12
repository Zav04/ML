

from joblib import load
import pandas as pd
import numpy as np


model = load('./Modelos/modelo_LogisticRegression.joblib')

random_data = np.random.rand(1, 27) 

# Fazer a previsão
prediction = model.predict(random_data)
print("Previsão:", prediction)
