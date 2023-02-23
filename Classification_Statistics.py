import numpy as np

# 统计构建混淆矩阵
def Classification_data_Statistics(Val_label, Test_label, label_num):
    # 构建二维list混淆矩阵
    Confusion_Matrix = np.zeros((len(label_num), len(label_num)))
    np.set_printoptions(precision=4, suppress=True)
    for i in range(len(Val_label)):
        # 判断真正例的情况
        if Test_label[i] == Val_label[i]:
            T_index = label_num.index(Test_label[i])
            Confusion_Matrix[T_index][T_index] = Confusion_Matrix[T_index][T_index] + 1
        # 判断误分情况
        else:
            # Test中被分类成了A类，实际上为B类的情况
            # 混淆矩阵中每一行之和表示该类别的真实样本数量，每一列之和表示被预测为该类别的样本数量
            A_index = label_num.index(Test_label[i])
            B_index = label_num.index(Val_label[i])
            Confusion_Matrix[B_index][A_index] = Confusion_Matrix[B_index][A_index] + 1
    print('测试集中点云数量为:', len(Test_label))
    print('验证集中点云数量为:', len(Val_label))
    print('混淆矩阵中元素数量总共为:', Confusion_Matrix.sum(axis=1).sum(axis=0))
    print('混淆矩阵:')
    print(Confusion_Matrix)
    return Confusion_Matrix
