from sklearn.metrics import confusion_matrix
import prettytable
import numpy as np

#计算每一类的IOU
def Intersection_over_Union(confusion_matrix):
    intersection = np.diag(confusion_matrix)#交集
    np.set_printoptions(precision=3, suppress=True)
    union = np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) - np.diag(confusion_matrix)#并集
    IoU = intersection / union #交并比，即IoU
    IoU = np.around(
        IoU,  # numpy数组或列表
        decimals=3  # 保留几位小数
    )
    return IoU

def kappa_cal(matrix):
    n = np.sum(matrix)
    sum_po = 0
    sum_pe = 0
    for i in range(len(matrix[0])):
        sum_po += matrix[i][i]
        row = np.sum(matrix[i, :])
        col = np.sum(matrix[:, i])
        sum_pe += row * col
    po = sum_po / n
    pe = sum_pe / (n * n)
    # print(po, pe)
    return (po - pe) / (1 - pe)

def calculate_prediction_recall(confMatrix, Total_num, classes=None,):
    """
    计算准确率和召回率:传入预测值及对应的真实标签计算
    :param label:标签
    :param pre:对应的预测值
    :param classes:类别名（None则为数字代替）
    :return:
    """
    #if classes:
    #    classes = list(range(classes))

    # 直接可以采用sklearn库计算传入标签计算混淆矩阵
    # from sklearn.metrics import confusion_matrix
    # confMatrix = confusion_matrix(label, pre)

    # print(classes)
    Average_Precision = 0
    Average_Recall = 0
    Overall_Accuracy = 0
    result_table = prettytable.PrettyTable()
    class_multi = 1
    result_table.field_names = ['   Class(类别名)   ', 'Prediction(精确率) ', 'Recall(召回率) ', 'F1_Score(F1分数)', 'IoU(交并比)']
    # 计算每一类的IoU
    IoU = Intersection_over_Union(confMatrix)
    for i in range(len(confMatrix)):
        label_total_sum_col = confMatrix.sum(axis=0)[i]
        label_total_sum_row = confMatrix.sum(axis=1)[i]
        if label_total_sum_col:     # 防止除0
            prediction = confMatrix[i][i] / label_total_sum_col
        else:
            prediction = 0
        if label_total_sum_row:
            recall = confMatrix[i][i] / label_total_sum_row
        else:
            recall = 0
        if (prediction + recall) != 0:
            F1_score = prediction * recall * 2 / (prediction + recall)
        else:
            F1_score = 0
        result_table.add_row([classes[i], np.round(prediction, 3), np.round(recall, 3),
                              np.round(F1_score, 3), IoU[i]])
        Overall_Accuracy = Overall_Accuracy + confMatrix[i][i]
        Average_Precision += prediction
        Average_Recall += recall
        class_multi *= prediction
    Overall_Accuracy = Overall_Accuracy / Total_num
    Average_Precision = Average_Precision / len(confMatrix)
    Average_Recall = Average_Recall / len(confMatrix)
    Average_F1_score = Average_Precision * Average_Recall * 2 / (Average_Precision + Average_Recall)
    geometric_mean = pow(class_multi, 1 / len(confMatrix))
    MIou = np.mean(IoU)#计算MIoU

    # Kappa系数是基于混淆矩阵的计算得到的模型评价参数(越接近 1 越好)
    # kappa = kappa_cal(confMatrix)

    # 返回准确率OA
    # 最后返回的是平均精确率(Average_Precision)、平均召回率(Average_Precision)、平均F1_score(Average_F1_score)、MIou、一个图表可视化
    return Average_Precision, Average_Recall, Average_F1_score, MIou, Overall_Accuracy, result_table