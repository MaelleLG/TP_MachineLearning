import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

house_data_raw = pd.read_csv('house.csv')
house_data = house_data_raw[house_data_raw['loyer']<7000]

#donnees
plt.plot(house_data['surface'], house_data['loyer'], 'ro')

#ensemble des données
X = np.matrix([np.ones(house_data.shape[0]), house_data['surface']]).T
#vecteur cible (montant du loyer)
y = np.matrix(house_data['loyer']).T
#estimateur
theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
print(theta)

#regression lineaire
plt.plot([0,250],[theta.item(0),theta.item(0)+250*theta.item(1)])
plt.show()





