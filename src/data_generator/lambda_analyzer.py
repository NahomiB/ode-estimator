import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


with open('lambdas.json', 'r') as file:
    lambda_values = json.load(file)

lambda_values = np.array(lambda_values)

mean_lambda = np.mean(lambda_values)
median_lambda = np.median(lambda_values)
std_lambda = np.std(lambda_values)
min_lambda = np.min(lambda_values)
max_lambda = np.max(lambda_values)

print(f"Media: {mean_lambda}")
print(f"Mediana: {median_lambda}")
print(f"Desviación estándar: {std_lambda}")
print(f"Mínimo: {min_lambda}")
print(f"Máximo: {max_lambda}")


plt.figure(figsize=(10, 6))
sns.histplot(lambda_values, kde=True, bins=30, color='skyblue')
plt.title('Distribución de los valores de lambda')
plt.xlabel('Valor de lambda')
plt.ylabel('Frecuencia')
plt.show()


plt.figure(figsize=(8, 6))
sns.boxplot(x=lambda_values, color='lightgreen')
plt.title('Boxplot de los valores de lambda')
plt.xlabel('Valor de lambda')
plt.show()

plt.figure(figsize=(8, 6))
stats.probplot(lambda_values, dist="norm", plot=plt)
plt.title('Q-Q Plot de los valores de lambda')
plt.show()


Q1 = np.percentile(lambda_values, 25)
Q3 = np.percentile(lambda_values, 75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = lambda_values[(lambda_values < lower_bound) | (lambda_values > upper_bound)]
print(f"Valores atípicos: {outliers}")
