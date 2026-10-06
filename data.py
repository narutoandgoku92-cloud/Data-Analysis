import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
from sklearn.tree import DecisionTreeClassifier



#Reading the CSV file using pandas
df = pd.read_csv('archive\\Iris.csv')


#Separating the features and target variable
x = df[
    ['SepalLengthCm',
     
     'SepalWidthCm',
     
     'PetalLengthCm',
     
     'PetalWidthCm']
    
    ]
y = df['Species']


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 42)


model = DecisionTreeClassifier(random_state=42)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)

cm  = confusion_matrix(y_test, y_pred)

print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", cm)
print("\nAccuracy Score:", accuracy)



new_flower = [[5.1, 3.5, 1.4, 0.2]]

prediction = model.predict(new_flower)

print("\nPrediction for new flower:", prediction[0])