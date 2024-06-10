import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
"""
======================
Part A: Prepare the Data
======================
"""
#read in data
uber = pd.read_csv('uberNYC_August2014.csv')
#create sample of 100,000 records
data = uber.sample(100000)
#create numpy arrays with the latitude and longitude
lat = np.array(data["Lat"])
lon = np.array(data["Lon"])
#split data into training and testing set
#X = lat
#Y = lon
X_train, X_test, y_train, y_test = train_test_split(lat, lon, test_size = 0.1)
#create array of 90,000 training features
trainArr = []
#create array of 10,000 testing features
for index in range(X_train.size):
    miniArray = [X_train[index], y_train[index]]
    trainArr.append(miniArray)
testArr = []
for index in range(X_test.size):
    miniArray = [X_test[index], y_test[index]]
    testArr.append(miniArray)
print(len(y_test),len(y_train), "Other", len(X_test),len(X_train))

""""
import pandas as pd
data = {
  "calories": [420, 380, 390],
  "duration": [50, 40, 45]
}
#load data into a DataFrame object:
df = pd.DataFrame(data)

print(df)
"""


"""
======================
Part B and C: Train and Visualize Training Data
======================
"""
def createKMeans(training):
    model = KMeans(
    n_clusters=8,
    n_init=10,
    random_state=42
    )
    p_labels = model.fit_predict(training)
    cluster_centers = model.cluster_centers_
    x=[]
    y=[]
    for index in range(len(training)):
      x.append(training[index][1])
      y.append(training[index][0])
    plt.scatter(x, y, marker='*', s=10, c=p_labels)

    cs_x=[]
    cs_y=[]
    for index in range(len(cluster_centers)):
       cs_x.append(cluster_centers[index][1])
       cs_y.append(cluster_centers[index][0])
    #k = black
    plt.scatter(cs_x, cs_y, marker='+', s=10, c='k')
    plt.title('NYC Uber Pickups - Training Set')
    plt.show()
createKMeans(trainArr)


"""
======================
Part 4: Predict Clusters of Testing Data
======================
"""

def predictTestData(testing):
   model = KMeans(
    n_clusters=8,
    n_init=10,
    random_state=42
    )
   p_labels = model.fit_predict(testing)
   cluster_centers = model.cluster_centers_
   x=[]
   y=[]
   for index in range(len(testing)):
      x.append(testing[index][1])
      y.append(testing[index][0])
   plt.scatter(x, y, marker='*', s=10, c=p_labels)

   cs_x=[]
   cs_y=[]
   for index in range(len(cluster_centers)):
       cs_x.append(cluster_centers[index][1])
       cs_y.append(cluster_centers[index][0])
   plt.scatter(cs_x, cs_y, marker='+', s=10, c='k')
   plt.title('NYC Uber Pickups - Predictions')
   plt.show()

predictTestData(testArr)
   
