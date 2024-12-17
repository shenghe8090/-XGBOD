
#python code for geochemical anomaly extraction
#Written By:
#Yongliang Chen

from utils.models.XGBOD_u2 import XGBOD_u2
from utils.models.XGBOD_u3 import XGBOD_u3
from utils.models.XGBOD_u4 import XGBOD_u4
from utils.models.XGBOD_uHBOS import XGBOD_uHBOS
from utils.models.XGBOD_uIForest import XGBOD_uIForest
from utils.models.XGBOD_uKNN import XGBOD_uKNN
from utils.models.XGBOD_uLOF import XGBOD_uLOF
from utils.models.XGBOD_uOCSVM import XGBOD_uOCSVM
from utils.modules.FileInputOutput import FileInputOutput
from utils.modules.PrePostProcess import PrePostprocess
from utils.modules.ROCAnalysis import ROC_Analysis
from pyod.models.xgbod import XGBOD
from xgboost import XGBClassifier
from time import time as current_time
import os
from sklearn.metrics import precision_recall_curve
#set the working directory
os.chdir(r'd:\\研究区矿产预测数据')


def run_model(clf_class, clf_params, clf_name, file_prefix, da, dep, ncases, d, colNum, linNum, xmin, xmax, ymin, ymax,
              deposit):
    print(f"{clf_name} modeling......Default")

    start_time = current_time()

    # 初始化并训练检测器
    clf = clf_class(**clf_params)
    clf.fit(da, dep)

    # 检查模型类型并获取异常分数或预测结果
    if hasattr(clf, 'decision_scores_'):
        decision_scores = clf.decision_scores_
    else:
        decision_scores = clf.predict(da)

    # 数据后处理
    distance = process.postprocess_data(ncases, d, decision_scores)

    # 输出网格文件
    DataInAndOut.output_grid_file(f"{file_prefix}_Anomaly.grd", colNum, linNum, xmin, xmax, ymin, ymax, distance)

    # ROC分析
    rocobj.compute_roc(1000, ncases, d, deposit, distance, f"{file_prefix}_AnomnalyRoc.txt",
                       f"{file_prefix}_Anomalygain.txt", f"{file_prefix}_Anomalylift.txt")
    area, se, z, aul = rocobj.compute_roc_area(ncases, d, deposit, distance)
    print(f"area{clf_name} = {area}, se{clf_name} = {se}, z{clf_name} = {z}, aul{clf_name} = {aul}")

    end_time = current_time()
    print("time =", end_time - start_time)

    # 计算precision-recall曲线
    precision, recall, _ = precision_recall_curve(dep, decision_scores)

    # 创建并写入数据到文本文件
    file_path = f'C:\\Users\\DELL\\Desktop\\pr曲线文件\\{file_prefix}.txt'
    with open(file_path, 'w') as file:
        for p, r in zip(precision, recall):
            file.write(f'{p}\t{r}\n')

if __name__ == '__main__':

    #Controlling parameters
    bins = 500
    deposits = 500#the maximum number of the input mineral deposits
    tolerance = 0

    #输入绘图区范围
    xmin = 126.1137161
    xmax = 128.3113404
    ymin = 41.36324691
    ymax = 42.81881714

    k = 100 #k is the number of rows
    l = 150 #int((xmax - xmin)/(ymax - ymin) * k + 0.5) #l is the number of collumns

    print ("k =",k)
    print ("l =",l)
    #construct class objects
    inputfile = "DepositGridFileName.txt"
    DataInAndOut = FileInputOutput(k,l,bins,deposits,xmin,xmax,ymin,ymax,inputfile)
    rocobj = ROC_Analysis()
    
    
    #DataInput
    deposit,data,auc1,youden,zmax,ncases,ndim,linNum,colNum = DataInAndOut.construct_griddata_from_grid_Files()


    #Preprocessing 
    process = PrePostprocess()
    d,da,dep,sample = process.preprocess_data(ncases,ndim,data,deposit)


    # 调用封装函数运行每个模型
    clf_params = {}
    run_model(XGBOD_uHBOS, clf_params, 'XGBOD_uHBOS', 'XGBOD_uHBOS', da, dep, ncases, d, colNum, linNum, xmin, xmax, ymin,
                    ymax, deposit)
    run_model(XGBOD_uLOF, clf_params, 'XGBOD_uLOF', 'XGBOD_uLOF', da, dep, ncases, d, colNum, linNum, xmin, xmax, ymin, ymax,
                    deposit)
    run_model(XGBOD_uOCSVM, clf_params, 'XGBOD_uOCSVM', 'XGBOD_uOCSVM', da, dep, ncases, d, colNum, linNum, xmin, xmax, ymin,
                    ymax, deposit)
    run_model(XGBOD_uKNN, clf_params, 'XGBOD_uKNN', 'XGBOD_uKNN', da, dep, ncases, d, colNum, linNum, xmin, xmax, ymin, ymax,
                    deposit)
    run_model(XGBOD_uIForest, clf_params, 'XGBOD_uIForest', 'XGBOD_uIForest', da, dep, ncases, d, colNum, linNum, xmin, xmax,
                    ymin, ymax, deposit)
    run_model(XGBOD_u2, clf_params, 'XGBOD_u2', 'XGBOD_u2', da, dep, ncases, d, colNum, linNum, xmin, xmax, ymin, ymax,
                    deposit)
    run_model(XGBOD_u3, clf_params, 'XGBOD_u3', 'XGBOD_u3', da, dep, ncases, d, colNum, linNum, xmin, xmax, ymin, ymax,
                    deposit)
    run_model(XGBOD_u4, clf_params, 'XGBOD_u4', 'XGBOD_u4', da, dep, ncases, d, colNum, linNum, xmin, xmax, ymin, ymax,
                    deposit)
    run_model(XGBOD, clf_params, 'XGBOD', 'XGBOD', da, dep, ncases, d, colNum, linNum, xmin, xmax, ymin, ymax, deposit)
    clf_params = {
        'n_estimators': 10,  # 决策树的数量
        'max_depth': 6,  # 每棵决策树的最大深度
        'learning_rate': 0.1,  # 学习率
        'objective': 'binary:logistic',  # 二元分类问题
    }
    run_model(XGBClassifier, clf_params, 'XGBoost', 'XGBoost', da, dep, ncases, d, colNum, linNum, xmin, xmax, ymin, ymax,
                    deposit)

    print ("OK!")
