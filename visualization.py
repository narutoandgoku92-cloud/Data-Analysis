import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier


# Loading the dataset
df = pd.read_csv('archive\\Iris.csv')


# Looking at the first 5 rows of the dataset
print("\nFirst 5 rows of the dataset:\n")
print(df.head())


# Checking the shape of the dataset
print("\nShape of the dataset:\n")
print(df.shape)


# Checking the data types of the columns
print("\nData Types:\n")
print(df.dtypes)

print("\nSummary Statistics:\n")
print(df.describe())

# Checking for null values
print("\nNull Values:\n")
print(df.isnull().sum())


# Visualizing the number of samples for each species
sns.countplot(x='Species', data=df)

plt.title("Number of Iris Species Samples")
plt.xlabel("Species")
plt.ylabel("Count")
plt.show()


# Scatter Plot
sns.scatterplot(
    x='SepalLengthCm',
    y='PetalLengthCm',
    hue='Species',
    data=df
)

plt.title("Scatter Plot of Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.show()


# Pair Plot
sns.pairplot(df, hue='Species')
plt.show()


# Separating features and target variable
x = df.drop(columns=['Id', 'Species'])
y = df['Species']


print("\nFeatures:\n")
print(x.head())

print("\nTarget Variable:\n")
print(y.head())


# Splitting the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=30
)


# Checking the sizes
print("\nTraining set size:", x_train.shape)
print("Test set size:", x_test.shape)


# Creating the Decision Tree model
model = DecisionTreeClassifier(random_state=42)


# Training the model
model.fit(x_train, y_train)


# Making predictions
y_pred = model.predict(x_test)


# Calculating accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)


# Classification Report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:\n")
print(cm)


# Visualizing the confusion matrix
sns.heatmap(
    cm,
    annot=True,
    cmap='Blues',
    fmt='d'
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()