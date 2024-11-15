import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 假设你的数据已经加载到DataFrame df中
df = pd.read_csv('nndb_flat_out.csv')  # 如果数据在CSV文件中

# 指定特征列和标签列
feature_columns = ['energy_kcal', 'protein_g', 'fat_g','carb_g','sugar_g','fiber_g']  # 或者使用索引，比如 [0, 2, 4]
label_column = 'foodgroup'  # 或者使用索引 1（注意索引是从0开始的）

# 提取特征和标签
X = df[feature_columns]
y = df[label_column]

# 拆分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# 创建随机森林分类器并训练模型
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 使用测试集进行预测
y_pred = clf.predict(X_test)

# 评估模型性能
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# 打印分类报告和混淆矩阵（可选）
print('Classification Report:')
print(classification_report(y_test, y_pred))

print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))