import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split # type: ignore
from pandas import read_csv
import matplotlib.pyplot as plt

data = read_csv('C://Users//hp//Downloads//diabetes.csv')
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

imputer = SimpleImputer(missing_values=0, strategy='median')
X_train2 = imputer.fit_transform(X_train)
X_test2 = imputer.transform(X_test)

X_train3 = pd.DataFrame(X_train2)

plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.hist(X_train3.iloc[:, 4][y_train == 0], color='blue', alpha=0.5, label='Healthy')
plt.hist(X_train3.iloc[:, 4][y_train == 1], color='orange', alpha=0.5, label='Diabetes')
plt.title('Insulin vs Diagnosis')
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(X_train3.iloc[:, 3][y_train == 0], color='blue', alpha=0.5, label='Healthy')
plt.hist(X_train3.iloc[:, 3][y_train == 1], color='orange', alpha=0.5, label='Diabetes')
plt.title('SkinThickness vs Diagnosis')
plt.legend()

plt.show()


labels = {0:'Pregnancies',1:'Glucose',2:'BloodPressure',3:'SkinThickness',4:'Insulin',5:'BMI',6:'DiabetesPedigreeFunction',7:'Age'}
print(labels)
print("\nColumn #, # of Zero Values\n")
print((X_train3[:] == 0).sum())

