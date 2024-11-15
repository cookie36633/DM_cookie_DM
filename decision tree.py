import pandas as pd  
from sklearn.model_selection import train_test_split  
from sklearn.tree import DecisionTreeClassifier  
from sklearn.metrics import accuracy_score, classification_report  
  
# 读取CSV文件  
data = pd.read_csv('nndb_flat_out.csv')  
  
# 分离特征和标签  
X = data[['energy_kcal', 'protein_g', 'fat_g','carb_g','sugar_g','fiber_g','vita_mcg']]  # 使用新的列名  
y = data['foodgroup']  # 使用新的列名  
  
# 将数据集拆分为训练集和测试集  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  
  
# 创建决策树分类器  
clf = DecisionTreeClassifier()  
  
# 训练模型  
clf.fit(X_train, y_train)  
  
# 预测测试集  
y_pred = clf.predict(X_test)  
  
# 计算准确率  
accuracy = accuracy_score(y_test, y_pred)  
print(f'Accuracy: {accuracy:.2f}')  
  
# 打印分类报告  
print('Classification Report:')  
print(classification_report(y_test, y_pred))