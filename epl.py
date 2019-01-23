import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score

away = pd.read_csv("EPL_SET.csv")

#away = mat.loc[mat['Season'] == '2017-18']

match = pd.DataFrame()

GF1 = away.groupby('HomeTeam')['FTHG'].sum()

GF2 = away.groupby('AwayTeam')['FTAG'].sum()

GF = GF1 + GF2

GA1 = away.groupby('HomeTeam')['FTAG'].sum()

GA2 = away.groupby('AwayTeam')['FTHG'].sum()

GA = GA1 + GA2

GD = GF - GA

Wins1 = away.groupby('HomeTeam')['FTR'].value_counts()

Wins2 = away.groupby('AwayTeam')['FTR'].value_counts()

sat = Wins1.xs('H', level = 'FTR')

pat = Wins2.xs('A', level = 'FTR')

point = sat.add(pat, fill_value = 0)
point1 = point*3

rat = Wins1.xs('D', level = 'FTR')

lat = Wins2.xs('D', level = 'FTR')

pointe = rat.add(lat, fill_value = 0)

fat = Wins1.xs('A', level = 'FTR')

hat = Wins2.xs('H', level = 'FTR')

poin = fat.add(hat, fill_value = 0)

MP = point + pointe + poin
match['MP'] = MP

match['W'] = point
match['D'] = pointe
match['L'] = poin

points = point1 + pointe
match['Pts'] = points
match['GF'] = GF
match['GA'] = GA
match['GD'] = GD
match = match.sort_values(by=['Pts'], ascending=False)

print(match)

match['Pts'].plot(kind="bar")
plt.show()

cdf = match[['MP','W','D','L','Pts','GF', 'GA', 'GD']]
sat = pd.read_csv("epl.csv")
df =  sat[['MP','W','D','L','Pts','GF', 'GA', 'GD']]

train = cdf
test = df

regr = linear_model.LinearRegression()
x = np.asanyarray(train[['W','D','L','GF','GA','GD']])
y = np.asanyarray(train[['Pts']])
regr.fit (x, y)
# The coefficients
print ('Coefficients: ', regr.coef_)

y_hat= regr.predict(test[['W','D','L','GF','GA','GD']])
x = np.asanyarray(test[['W','D','L','GF','GA','GD']])
y = np.asanyarray(test[['Pts']])
print(y_hat)