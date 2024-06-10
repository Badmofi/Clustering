import pandas as pd
import numpy as np

uber = pd.read_csv('uberNYC_August2014.csv')
data = uber.sample(100000)


print(data.head())
print(data.columns)
print(data["Lat"][:5])
print(data["Lon"][:5])

lat = np.array(data["Lat"])
print(lat)
lon = np.array(data["Lon"])
print(lon)
