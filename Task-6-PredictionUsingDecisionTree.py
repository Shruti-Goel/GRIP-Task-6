import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics


df=pd.read_csv('Iris.csv')
df.info()

df.describe()

df.head()  #viewing the first 5 row of the dataset

df.isnull().sum()  #checking if there are any null values present in the dataset

plt.title("Iris Species %")
df['Species'].value_counts().plot(kind = 'pie', autopct = '%1.1f%%', shadow = True, explode = [0.02,0.02,0.02],colors=("m","g","y"))
plt.show()

features = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']
X = df.loc[:, features].values   #defining the feature matrix
y = df.Species


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33,random_state=0)

dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)

from sklearn import tree

feature_name =  ['sepal length(cm)','sepal width(cm)','petal length(cm)','petal width(cm)']
class_name= df.Species.unique()
plt.figure(figsize=(15,10))
tree.plot_tree(dtree, filled = True, feature_names = feature_name, class_names= class_name)

y_pred = dtree.predict(X_test)
y_pred

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, y_pred)

print('\nThe model predicts the type as: ', dtree.predict([[6.5, 3, 4.3, 0.5]]))