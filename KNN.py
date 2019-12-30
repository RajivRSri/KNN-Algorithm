#Import Libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Set Working Directory
os.chdir("C:/R")

#Read Dataset
data = pd.read_csv("Classified Data", index_col = 0)
data.head()

#Standardize the dataset
from sklearn.preprocessing import StandardScaler
stdScaler = StandardScaler()
stdScaler.fit(data.drop("TARGET CLASS", axis = 1))
scaled = stdScaler.transform(data.drop("TARGET CLASS", axis = 1))
new_data = pd.DataFrame(scaled, columns = data.columns[:-1])

#Pair Plot
import seaborn as sns
sns.pairplot(new_data, hue = "TARGET CLASS")

#Split Dataset into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(new_data, data["TARGET CLASS"], train_size = 0.7, random_state = 0)

#Elbow method to find optimum K value
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
acc_rate = []
for index in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors = index)
    score = cross_val_score(knn, new_data, data["TARGET CLASS"], cv = 10)
    acc_rate.append(score.mean())

plt.figure(figsize=(10,5))
plt.plot(range(1, 40),
         acc_rate,
         color = 'red',
         linestyle = 'dashed',
         marker = 'o',
         markerfacecolor = 'red',
         markersize = 10)
plt.title("Accuracy Rate vs K - Value")
plt.xlabel("K - Value")
plt.ylabel("Accuracy Rate")
plt.show()

#At k = 23 the graph becomes stable
knn = KNeighborsClassifier(n_neighbors = 23)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)

#Accuracy
from sklearn.metrics import confusion_matrix, classification_report
acc = classification_report(Y_test, Y_pred)
print (acc)

