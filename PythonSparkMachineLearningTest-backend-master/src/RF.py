import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_svmlight_file
import os

# 设定目录路径
TEST_DATA_DIR = "../dataset/useful_dataset/test"  # 根据实际情况更新路径

# 遍历目录中的每个文件
for file_name in os.listdir(TEST_DATA_DIR):
    if file_name.endswith('.csv.libsvm'):
        file_path = os.path.join(TEST_DATA_DIR, file_name)
        print(f"Processing {file_name}...")

        # 加载LIBSVM格式的数据文件
        X, y = load_svmlight_file(file_path)

        # 转换为稠密矩阵
        X_dense = X.toarray()

        # 替换无穷大和无穷小值为最大和最小的浮点数
        X_dense[np.isinf(X_dense)] = np.finfo(np.float64).max

        # 使用均值进行输入
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        X_imputed = imputer.fit_transform(X_dense)

        # 创建随机森林分类器
        model = RandomForestClassifier(n_estimators=11, max_depth=20, max_features='sqrt', max_leaf_nodes=32)

        # 初始化用于收集所有拆分的真实标签和预测标签的列表
        y_true_all, y_pred_all = [], []

        # 设置StratifiedKFold进行分层抽样
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # 遍历每个拆分
        for train_index, test_index in skf.split(X_imputed, y):
            X_train, X_test = X_imputed[train_index], X_imputed[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # 在训练集上训练模型
            model.fit(X_train, y_train)

            # 在测试集上预测
            y_pred = model.predict(X_test)

            # 将这次拆分的真实标签和预测标签添加到列表中
            y_true_all.extend(y_test)
            y_pred_all.extend(y_pred)

        # 计算所有拆分上的综合准确率
        accuracy = accuracy_score(y_true_all, y_pred_all)
        print(f'[INFO] Overall Accuracy for {file_name}: {accuracy:.2f}')

        # 计算并打印所有拆分上的综合分类报告
        report = classification_report(y_true_all, y_pred_all)
        print(f'[INFO] Overall Classification Report for {file_name}:\n{report}')
