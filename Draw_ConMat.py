import csv
import itertools
import numpy as np
import matplotlib.pyplot as plt

# 绘制混淆矩阵
def plot_confusion_matrix(cm, classes, pic_save = 'ConfusionMatrix.png', normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    """
    # 混淆矩阵画图
    cm = np.array(cm)
    cm = cm.astype(np.int64)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=0)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=14)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=-15, fontsize=8)
    plt.yticks(tick_marks, classes, fontsize=8)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label',
           labelpad=-35,  # 调整y轴标签与y轴的距离
           y=0.45,  # 调整y轴标签的上下位置
           rotation=0,fontsize=12)
    plt.xlabel('Predicted label',
           labelpad=1,  # 调整X轴标签与x轴的距离
           x=0.5,  # 调整x轴标签的上下位置
           rotation=0,fontsize=12)
    # 当设置bbox_inches参数为tight时，边缘留白也就是边框的宽度你可以通过设置pad_inches的值来自定义，默认是0.1（单位：英寸）
    plt.savefig(pic_save, dpi=720, bbox_inches='tight', pad_inches=0.3)  # dpi     分辨率
    plt.show()


# 将数据集的第column列转换成float形式
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())  # strip()返回移除字符串头尾指定的字符生成的新字符串。

##加载数据，一行行的存入列表
def loadCSV(filename):
    dataSet = []
    with open(filename, 'r') as file:
        csvReader = csv.reader(file)
        for line in csvReader:
            dataSet.append(line)
    return dataSet

