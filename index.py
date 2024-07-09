import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.svm import SVC



#Load dataset
df = pd.read_csv('C:\\Users\\hegde\\OneDrive\\Desktop\\Parkinsons Disease Prediction\\parkinsons.csv')
pd.set_option('display.max_columns', 25)
print(df.head())
print(df.shape)
print(df.info())

#drop name column
df.drop(columns='name', inplace=True)

#features and label
y = df['status']
X = df.drop(columns='status')

"""every other column is also unique in their values range. This is not good. We cannot feed 
a machine learning algorithm unstandardized features.So, let’s standardize it using the Standard Scaler."""

#Standardizing the features
scaler = StandardScaler()
X = scaler.fit_transform(X)
print(X)

#train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=34)
print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)

#model
svc = SVC()
svc.fit(X_train, y_train)
svc_prediction = svc.predict(X_test)

svc_score = accuracy_score(y_test, svc_prediction)
print(svc_score) # 0.8974358974358975

confmat = confusion_matrix(y_test,svc_prediction)
print(confmat)

sns.heatmap(confmat,annot=True)
plt.show()