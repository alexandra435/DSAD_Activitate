import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df = pd.read_csv("Wholesale customers data.csv", sep=',', header=0)

print(df.head())
print(df.shape)


coloane = [
    "Fresh", "Milk", "Grocery",
    "Frozen", "Detergents_Paper", "Delicassen"]

print(df.columns.tolist())


x = df[coloane]
print(x)

print('-----------------------------------')
print('------------CERINTA 2--------------')
print('-----------------------------------')

scaler=StandardScaler()
x_scaled=scaler.fit_transform(x)

print(x_scaled)

print('-----------------------------------')
print('-------------CERINTA3-------------')
print('-----------------------------------')

x_scaled_df = pd.DataFrame(x_scaled, columns=coloane)

matrice = x_scaled_df.corr()

plt.figure(figsize=(9, 6))
sns.heatmap(matrice, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

print('-----------------------------------')
print('-------------CERINTA 4-------------')
print('-----------------------------------')
pca=PCA()
pca.fit(x_scaled)

explained_variance = pca.explained_variance_ratio_

for i, variance in enumerate(explained_variance):
    print(f"PC{i+1}: {variance:.2%}")

print('-----------------------------------')
print('------------CERINTA 5--------------')
print('-----------------------------------')

suma_cumulata = np.cumsum(explained_variance)

for i in range(len(suma_cumulata)):
    if suma_cumulata[i] >= 0.80:
        print("Număr componente necesare:", i + 1)
        break




