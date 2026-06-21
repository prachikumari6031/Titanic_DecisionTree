#import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


# Load dataset
df = pd.read_csv("titanic.csv")
df = df.fillna("missing")
df = df.astype(str)

# Convert categorical columns into numeric
le = LabelEncoder()
for col in df.columns:
     df[col] =  le.fit_transform(df[col])

# Feature and target
x = df.drop("Survived", axis=1)
y = df["Survived"]

# Split data 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train model
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(x_train, y_train)

# Predict + accuracy
y_pred = model.predict(x_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Raport: \n", classification_report(y_test, y_pred))

# Draw tree
plt.figure(figsize=(12,8))
plot_tree(model, feature_names=x.columns, class_names=["Not Survived", "Survived"], filled=True)
plt.show()

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))