import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
# import input.py

df = pd.read_csv('Car_prices_known.csv')
df['Mileage'] = df['Mileage'].str.replace('km', '')
df['Mileage'] = df['Mileage'].astype(int)
df.drop(['ID', 'Levy', 'Model', 'Category', 'Doors', 'Wheel', 'Color'], axis = 1, inplace=True)
df.replace({'Fuel type':{'Petrol':0,'Diesel':1,'CNG':3, 'Hybrid':2, 'LPG' :4, 'Plug-in Hybrid':5, 'Hydrogen':6}},inplace=True)
df.drop('Manufacturer', axis =1, inplace=True)
df.replace({'Gear box type':{'Automatic':0, 'Tiptronic':1, 'Manual':2, 'Variator':3}}, inplace=True)
df.replace({'Leather interior':{'Yes':0, 'No':1}}, inplace=True)
df= df[~df['Engine volume'].str.contains('Turbo')]
df.replace({'Drive wheels': {'Front':0, '4x4':1, 'Rear':2}}, inplace=True)
df['Engine volume'] = df['Engine volume'].astype(float)
df['Car_age'] = 2025 - df['Prod. year']
df.drop(['Prod. year'], axis=1, inplace=True)
X = df.drop('Price', axis =1)
Y = df['Price']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=2)
regressor = RandomForestRegressor(n_estimators=100, random_state=0)
regressor.fit(X_train, Y_train)
predictions = regressor.predict(X_train)
r2 = r2_score(Y_train, predictions)
print(f'R-squared: {r2}')