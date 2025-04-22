import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


df = pd.read_csv("50_Startups.csv")
print(df.head())

"""
Dataset contains R&D Spend, Administration, Marketing Spend, State, and Profit.
"""

df_encoded = pd.get_dummies(df, drop_first=True)

sns.heatmap(df_encoded.corr().round(2), annot=True)
plt.title("Correlation Matrix")
plt.show()

"""
R&D Spend has strongest positive correlation with profit, followed by Marketing Spend.
"""

plt.subplot(1, 2, 1)
plt.scatter(df['R&D Spend'], df['Profit'])
plt.xlabel('R&D Spend')
plt.ylabel('Profit')

plt.subplot(1, 2, 2)
plt.scatter(df['Marketing Spend'], df['Profit'])
plt.xlabel('Marketing Spend')
plt.ylabel('Profit')
plt.tight_layout()
plt.show()

X = df[['R&D Spend', 'Marketing Spend']]
y = df['Profit']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

model = LinearRegression()
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
r2_train = r2_score(y_train, y_train_pred)
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
r2_test = r2_score(y_test, y_test_pred)

print("Profit Train RMSE:", rmse_train, "R2:", r2_train)
print("Profit Test RMSE:", rmse_test, "R2:", r2_test)

"""
Profit prediction performs best using R&D Spend and Marketing Spend due to high correlation.
"""