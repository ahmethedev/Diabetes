# Ahmet Burak Dinç - Furkan Akman

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier

# Verisetini yükleme
file_path = "./diabetes.csv.xls" 
df = pd.read_csv(file_path)


print(df.head())

# Verisetinin özet istatistikleri
print(df.describe())

# Eksik değer kontrolü
print(df.isnull().sum())

# Eksik değerleri doldurma
df.fillna(df.median(), inplace=True)

# Ayırt edici özellikleri ölçeklendirme
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df.drop('Outcome', axis=1))
df_scaled = pd.DataFrame(scaled_features, columns=df.columns[:-1])
df_scaled['Outcome'] = df['Outcome']

# Hedef değişkenin dağılımı
sns.countplot(x='Outcome', data=df)
plt.show()

# Özellik ve hedef değişkenlerin ayrılması
X = df_scaled.drop('Outcome', axis=1)
y = df_scaled['Outcome']

# Veriyi eğitim ve test setlerine bölme
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeli oluşturma ve eğitme 
param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20, 30]}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# En iyi modeli alma
best_model = grid_search.best_estimator_

# Tahmin yapma
y_pred = best_model.predict(X_test)

# Modelin değerlendirilmesi
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Confusion matrix ve classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ROC AUC Skoru
roc_auc = roc_auc_score(y_test, y_pred)
print(f"ROC AUC Score: {roc_auc:.2f}")

# ROC Eğrisi
fpr, tpr, _ = roc_curve(y_test, y_pred)
plt.plot(fpr, tpr, label=f"ROC Curve (area = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='best')
plt.show()

# Confusion matrix görselleştirme
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Kullanıcı girdisi alma
def get_user_input():
    pregnancies = float(input("Number of Pregnancies: "))
    glucose = float(input("Glucose Level: "))
    blood_pressure = float(input("Blood Pressure: "))
    skin_thickness = float(input("Skin Thickness: "))
    insulin = float(input("Insulin Level: "))
    bmi = float(input("BMI: "))
    diabetes_pedigree_function = float(input("Diabetes Pedigree Function: "))
    age = float(input("Age: "))
    
    user_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]])
    user_data_scaled = scaler.transform(user_data)
    return user_data_scaled

# Kullanıcıdan veri alma ve tahmin yapma
user_data = get_user_input()
user_prediction = best_model.predict(user_data)

if user_prediction[0] == 1:
    print("The model predicts that the patient has diabetes.")
else:
    print("The model predicts that the patient does not have diabetes.")
