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
from joblib import Parallel, delayed  # For parallel processing

# Set the working directory for the project files
os.chdir(r'd:\Geochemical_Anomaly_Data')


# Function to run a model for geochemical anomaly extraction
def run_model(clf_class, clf_params, clf_name, file_prefix, da, dep, ncases, d, colNum, linNum, xmin, xmax, ymin, ymax,
              deposit):
    print(f"{clf_name} modeling... Default")  # Print the model's name for tracking

    start_time = current_time()  # Start tracking the time for model fitting

    # Initialize and train the model with the given parameters
    clf = clf_class(**clf_params)
    clf.fit(da, dep)  # Fit the model to the data (features: da, labels: dep)

    # Get the decision scores or predictions from the model
    if hasattr(clf, 'decision_scores_'):
        decision_scores = clf.decision_scores_
    else:
        decision_scores = clf.predict(da)

    # Post-process the decision scores to prepare for output
    distance = process.postprocess_data(ncases, d, decision_scores)

    # Output the results as a grid file
    DataInAndOut.output_grid_file(f"{file_prefix}_Anomaly.grd", colNum, linNum, xmin, xmax, ymin, ymax, distance)

    # Perform ROC analysis to assess model performance
    rocobj.compute_roc(1000, ncases, d, deposit, distance, f"{file_prefix}_AnomalyRoc.txt",
                       f"{file_prefix}_AnomalyGain.txt", f"{file_prefix}_AnomalyLift.txt")
    area, se, z, aul = rocobj.compute_roc_area(ncases, d, deposit, distance)
    print(f"Area {clf_name} = {area}, SE {clf_name} = {se}, Z {clf_name} = {z}, AUL {clf_name} = {aul}")

    end_time = current_time()  # End time after model processing
    print("Time elapsed:", end_time - start_time)  # Print time taken for model execution

    # Compute the precision-recall curve for model evaluation
    precision, recall, _ = precision_recall_curve(dep, decision_scores)

    # Save the precision-recall curve to a file
    file_path = os.path.join('C:\\Users\\DELL\\Desktop\\PR_Curve_Files', f'{file_prefix}.txt')
    with open(file_path, 'w') as file:
        for p, r in zip(precision, recall):
            file.write(f'{p}\t{r}\n')  # Write precision and recall pairs to file


if __name__ == '__main__':
    # Define control parameters for model execution
    bins = 500  # Number of bins used for processing
    max_input_deposits = 500  # Maximum number of input mineral deposits
    tolerance = 0  # Tolerance for anomaly detection

    # Set the boundaries of the study area (geographical coordinates)
    xmin = 126.1137161
    xmax = 128.3113404
    ymin = 41.36324691
    ymax = 42.81881714

    k = 100  # Number of rows in the grid
    l = 150  # Number of columns in the grid

    print("Rows (k) =", k)  # Print grid dimensions
    print("Columns (l) =", l)

    # Initialize class instances for data handling and analysis
    inputfile = "DepositGridFileName.txt"  # Input data file
    DataInAndOut = FileInputOutput(k, l, bins, max_input_deposits, xmin, xmax, ymin, ymax, inputfile)  # File I/O handler
    rocobj = ROC_Analysis()  # ROC analysis handler

    # Load input data from files (deposits, data, and other metrics)
    deposit, data, auc1, youden, zmax, ncases, ndim, linNum, colNum = DataInAndOut.construct_griddata_from_grid_Files()

    # Preprocess data before feeding it to models
    process = PrePostprocess()
    d, da, dep, sample = process.preprocess_data(ncases, ndim, data, deposit)

    # List of models to run
    models = [
        (XGBOD_uHBOS, {}, 'XGBOD_uHBOS'),
        (XGBOD_uLOF, {}, 'XGBOD_uLOF'),
        (XGBOD_uOCSVM, {}, 'XGBOD_uOCSVM'),
        (XGBOD_uKNN, {}, 'XGBOD_uKNN'),
        (XGBOD_uIForest, {}, 'XGBOD_uIForest'),
        (XGBOD_u2, {}, 'XGBOD_u2'),
        (XGBOD_u3, {}, 'XGBOD_u3'),
        (XGBOD_u4, {}, 'XGBOD_u4'),
        (XGBOD, {}, 'XGBOD'),
        (XGBClassifier, {  # Default XGBoost parameters
            'n_estimators': 10,
            'max_depth': 6,
            'learning_rate': 0.1,
            'objective': 'binary:logistic',
        }, 'XGBoost')
    ]

    # Run the models in parallel using all available CPU cores
    Parallel(n_jobs=-1)(  # Parallel execution
        delayed(run_model)(clf, params, name, name, da, dep, ncases, d, colNum, linNum, xmin, xmax, ymin, ymax, deposit)
        for clf, params, name in models)  # For each model in the list, run the anomaly extraction

print("Processing complete!")  # Print when all models have finished processing
