import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, f1_score

data = pd.read_csv("E:\\CouDocuments\\535DataMining\\nndb_flat_out.csv")

# Perform one-hot encoding on the foodgroup column
one_hot_encoder = OneHotEncoder()
foodgroup_encoded = one_hot_encoder.fit_transform(data[['foodgroup']]).toarray()

# Add the results of one-hot encoding to the original data frame
# for i, col in enumerate(one_hot_encoder.get_feature_names_out()):
#     data[col] = foodgroup_encoded[:, i]

# Select features
X = data[['energy_kcal', 'protein_g', 'fat_g', 'carb_g', 'sugar_g', 'fiber_g']]

# Get the target variable y from the one-hot encoding result and convert it to class numbers.
y = foodgroup_encoded.argmax(axis=1)

for value in y:
    print(value)

# Split the data into training set and test set.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=39)

# Standardize the data.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build and train the SVM model.
svm_model = SVC()
svm_model.fit(X_train, y_train)

# Make predictions.
y_pred = svm_model.predict(X_test)

# Calculate the accuracy, recall and F1-score.
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print(f"SVM Model accuracy：{accuracy}")
print(f"SVM Model recall：{recall}")
print(f"SVM Model f1：{f1}")