import numpy as np

from sklearn.datasets import load_iris

# 加载鸢尾花数据集
iris = load_iris()
print(iris.data)

# 查看数据集的介绍
X = iris.data
# 标签（150个元素的一维数组，包含0、1、2三个值分别代表三种鸢尾花）
y = iris.target

data = np.hstack((X, y.reshape(-1, 1)))
# 通过随机乱序函数将原始数据打乱
np.random.shuffle(data)
# 选择80%的数据作为训练集
train_size = int(y.size * 0.8)
train, test = data[:train_size], data[train_size:]
X_train, y_train = train[:, :-1], train[:, -1]
X_test, y_test = test[:, :-1], test[:, -1]

from sklearn.neighbors import KNeighborsClassifier

# 创建模型
model = KNeighborsClassifier()
# 训练模型
model.fit(X_train, y_train)
# 预测结果
y_pred = model.predict(X_test)
print(y_pred == y_test)
mai=model.score(X_test, y_test)
print(mai)
print(y_test)
print(y_pred)
from sklearn.metrics import classification_report, confusion_matrix

# 输出分类模型混淆矩阵
print('混淆矩阵: ')
print(confusion_matrix(y_test, y_pred))
# 输出分类模型评估报告
print('评估报告: ')
print(classification_report(y_test, y_pred))
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# 创建混淆矩阵显示对象
cm_display_obj = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), display_labels=iris.target_names)
# 绘制并显示混淆矩阵
cm_display_obj.plot(cmap=plt.cm.Reds)
plt.show()
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import RocCurveDisplay

# 手动构造一组真实值和对应的预测值
y_test_ex = np.array([0, 0, 0, 1, 1, 0, 1, 1, 1, 0])
y_pred_ex = np.array([1, 0, 0, 1, 1, 0, 1, 1, 0, 1])
# 通过roc_curve函数计算出FPR（假正例率）和TPR（真正例率）
fpr, tpr, _ = roc_curve(y_test_ex, y_pred_ex)
# 通过auc函数计算出AUC值并通过RocCurveDisplay类绘制图形
RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc(fpr, tpr)).plot()
plt.show()