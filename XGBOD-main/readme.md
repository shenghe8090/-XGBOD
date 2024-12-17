# **XGBOD**

This is the implementation of the XGBOD architecture described in this paper:

*XGBOD:  A high-performance extreme gradient boosting outlier detection framework for integrating the outputs of diverse anomaly detectors for detecting mineralization-related geochemical anomalies*

by Sheng He, Yongliang Chen

## Program language

- Python

## Dependencies required

- Python 3.9 or higher

## Usage

Run `train_XGBOD.py` to input the geochemical data and detect multivariate geochemical anomalies via XGBOD method.

In the "utils" folder, you can find the following:
    models: The XGBOD framework integrating different numbers of anomaly detectors. Different values of q represent different methods for generating transformed outlier scores (TOSs), ranging from using the XGBoost model directly for detecting mineralization-related anomalies (q=0) to utilizing all five anomaly detectors (OCSVM, IF, LOF, KNN, and HBOS) to generate TOSs (q=5).
    modules: Includes various modules that implement specific functionalities used in the code. However, it is not authorised to be made public, so please understand that it is blank.
    
## Contact

if you have any question, you can contact me via email [chenyongliang2009@hotmail.com](mail to:chenyongliang2009@hotmail.com)
