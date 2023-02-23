from DrawDataImport import draw_data_import
from Classification_Statistics import Classification_data_Statistics
from Draw_ConMat import plot_confusion_matrix
from ConMat_SaveCSV import SaveCSV
from Precision_Calculation import calculate_prediction_recall
from sklearn.metrics import cohen_kappa_score

if __name__ == '__main__':

    # 验证集路径
    Val_filename = r'E:\Deep_Learning\data\Mult_Lidardataset\RandLA-Net\网络训练结果\2-22\5-3验证\5-3验证.txt'
    # 测试集路径
    Test_filename = r'E:\Deep_Learning\data\Mult_Lidardataset\RandLA-Net\网络训练结果\2-22\5-3.txt'
    # 混淆矩阵csv写出保存路径
    Confusion_Matrix_csv_save = r'E:\Deep_Learning\data\Mult_Lidardataset\RandLA-Net\网络训练结果\2-22\5-3混淆矩阵.csv'
    # 混淆矩阵画图保存路径
    Confusion_Matrix_pic_save = r'E:\Deep_Learning\data\Mult_Lidardataset\RandLA-Net\网络训练结果\2-22\5-3混淆矩阵.png'

    # 验证集标签list
    Val_label = draw_data_import(Val_filename)
    # 测试标签list
    Test_label = draw_data_import(Test_filename)
    # 分类类别个数
    label_num = [1, 2, 3, 4, 5, 6]
    print('类别数量:',label_num)

    # 混淆矩阵统计构建
    Confusion_Matrix = Classification_data_Statistics(Val_label, Test_label, label_num)
    Total_num = len(Val_label)

    # 混淆矩阵中类别名称
    attack_types = ['Impermeable_Ground', 'Grass', 'Building', 'Tree', 'Car', 'Powerline']
    # attack_types = ['Road', 'Grass', 'Building', 'Tree', 'Car', 'Powerline']
    plot_confusion_matrix(Confusion_Matrix, classes=attack_types, pic_save = Confusion_Matrix_pic_save, normalize=True, title='Normalized confusion matrix')

    # 混淆矩阵写出保存
    SaveCSV(Confusion_Matrix_csv_save, Confusion_Matrix)

    # 计算精度
    Average_Precision, Average_Recall, Average_F1_score, MIou, Overall_Accuracy, result_table = calculate_prediction_recall(Confusion_Matrix, Total_num,classes=attack_types)

    kappa = cohen_kappa_score(Val_label, Test_label)
    print('总体精度(Overall_Accuracy):', Overall_Accuracy)
    print('平均精确率(Average_Precision):', Average_Precision)
    print('平均召回率(Average_Precision):', Average_Recall)
    print('平均F1_score(Average_F1_score):', Average_F1_score)
    print('平均交并比MIou为:', MIou)
    print('Kappa系数为:', kappa)
    print('每一类分类统计表:')
    print(result_table)