import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt
from sklearn import metrics
import xgboost as xgb # use xgb.cv 
from datetime import datetime, timedelta


def StandardizerTrainTest(X_train, X_test, standardizationList, binaryList, freqList, nonstdList):
    """Standardization and prevent data leakage

    Args:
        X_train (pd.dataframe): training data
        X_test (pd.dataframe): testing data
        standardizationList (list): continuous variables needed to be standardized 
        binaryList (list): binary variables  

    Returns:
        pd.DataFrame: train, test 
    """
    std = StandardScaler()
    std.fit(X_train[standardizationList])
    x_train_standardized1 = pd.DataFrame(std.transform(X_train[standardizationList]),columns=standardizationList)
    x_train_using = pd.concat([x_train_standardized1,X_train[binaryList], X_train[freqList], X_train[nonstdList]],axis=1)
    # to prevent data leakage 
    x_test_standardized1 = pd.DataFrame(std.transform(X_test[standardizationList]),columns=standardizationList)
    x_test_using = pd.concat([x_test_standardized1,X_test[binaryList], X_test[freqList], X_test[nonstdList]],axis=1)
    return x_train_using, x_test_using

def resampling(X_train, Y_train): 
    """generate the resampling data

    Args:
        X_train (dataframe): X_train
        Y_train (dataframe): X_train

    Returns:
        dataframe: X_smote, Y_smote, X_ADA, Y_ADA
    """
    smote = SMOTE(random_state=527)
    X_smote, Y_smote = smote.fit_resample(X_train,Y_train)
    adasyn = ADASYN(random_state=527)
    X_ADA , Y_ADA = adasyn.fit_resample(X_train, Y_train)
    
    return X_smote, Y_smote, X_ADA, Y_ADA

def _Flexible_test_prediction(new_threshold, test_Prob):
    """Predict the testing result with different threshold probabilities 

    Args:
        new_threshold (float): new threshold to be classified as 1 
        test_Prob (list like): the probability to be 1 in the testing data 

    Returns:
        list like: new binary prediction result 
    """
    res = np.where(test_Prob >= new_threshold, 1, 0)
    return res

def Metrics_TradeOff_Plot(Testing_Prob, Y_testData, plotTitle, plotSave): 
    """Make the metrics trade-off plot, and save

    Args:
        Testing_Prob (list like): the probability to be 1 in the testing data, one dimensional
        Y_testData (list like): Y testing data
        plotTitle (str): name of the plot
        plotSave (filepath): saving path and figure file name ended with .png 
    
    Returns:
        f1, accuracy, recall, precision 
    """
    new_new_threshold = np.arange(0,101,1)/100
    y_new_prediction = map(lambda x: _Flexible_test_prediction(x, Testing_Prob), 
                        new_new_threshold)
    newpred1 = np.array(list(y_new_prediction))
    f_one = []
    acc = []
    recall = []
    precision = []
    for i in range(newpred1.shape[0]):
        newpredloop = newpred1[i,:].reshape((-1,1))
        f_one.append(f1_score(Y_testData,newpredloop))
        acc.append(accuracy_score(Y_testData, newpredloop))
        recall.append(recall_score(Y_testData, newpredloop))
        precision.append(precision_score(Y_testData,newpredloop , zero_division = 0))
        
    plt.figure(figsize=(10,6))
    plt.plot(new_new_threshold, acc, label='Accuracy')
    plt.plot(new_new_threshold, recall, label='Recall')
    plt.plot(new_new_threshold, precision, label='Precision')
    plt.plot(new_new_threshold, f_one, label='F1')
    plt.hlines(y = 0.9, xmin=0, xmax=1, linestyles="dotted", colors="gray", label='90%')
    plt.hlines(y = 0.7, xmin=0, xmax=1, linestyles='dotted', colors="black", label='70%')
    plt.hlines(y = 0.5, xmin=0, xmax=1, linestyles='dotted', colors="brown", label='50%')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title(plotTitle)
    plt.legend()
    plt.grid(which = "major", linewidth = 1)
    plt.grid(which = "minor", linewidth = 0.3)
    plt.minorticks_on()
    plt.savefig(plotSave)
    
    return f_one, acc, recall, precision 

def XGBmodelfit(alg, dtrain, predictors, X_test, Y_test,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    """Fit the XGBoost automatically and return the needed matrics and importance figure 
    this function is a modified version of function in: 
    https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/ 
    !!! Important: the response variable name is FCSStaus !!!
    
    Args:
        alg (xgboost): xgboost model
        dtrain (dataframe): dataframe of training data (includes the response variable)
        predictors (list): predictor list 
        X_test (dataframe): X testing data 
        Y_test (dataframe): Y testing data
        useTrainCV (bool, optional): if use the CV to find the best n_estimator. Defaults to True.
        cv_folds (int, optional): CV folds. Defaults to 5.
        early_stopping_rounds (int, optional): for iteration. Defaults to 50. 
        
    Return: 
        alg: Trained XGBoost 
        dtest_predprob_pre: Predicted testing probability
        dtest_predictions_pre: Predicted testing labels 
        dtrain_predprob: Predicted training probability 
    """
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain["FCSStaus"].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain['FCSStaus'])
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
    dtest_predprob_pre = alg.predict_proba(X_test[predictors])[:,1]
    dtest_predictions_pre = alg.predict(X_test[predictors])
    #Print model report:
    print("\nModel Report")
    # print("Accuracy : %.4g" % metrics.accuracy_score(dtrain['FCSStaus'].values, dtrain_predictions))
    print( "AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['FCSStaus'], dtrain_predprob))
    print( "AUC Score (Test): %f" % metrics.roc_auc_score(Y_test, dtest_predprob_pre))
    # print( "F1 Macro Score (Test): %f" % metrics.f1_score(Y_test, dtest_predictions_pre, average="macro"))
    # print( "F1 Weighted Score (Test): %f" % metrics.f1_score(Y_test, dtest_predictions_pre, average="weighted"))
    print( "Recall (Test): %f" % metrics.recall_score(Y_test, dtest_predictions_pre))
          
    # importances = alg.feature_importances_
    # feature_names = np.array(predictors)  
    # # Plot feature importances
    # indices = np.argsort(importances)[::-1]
    # sorted_importances = importances[indices]
    # sorted_feature_names = feature_names[indices]

    # Plot feature importances
    # plt.bar(range(len(sorted_importances)), sorted_importances)
    # plt.xticks(range(len(sorted_importances)), sorted_feature_names, rotation='vertical')
    # plt.xlabel('Features')
    # plt.ylabel('Importance')
    # plt.title('Feature Importances')
    # plt.show()
    return alg, dtest_predprob_pre, dtest_predictions_pre, dtrain_predprob

def variable_distribution_crosscheck(varname, dataframe):
    """Make the plot of conditional distribution of 

    Args:
        varname (str): variable name, should be a continuous variable
        dataframe (dataFrame): DF which contains FCSStaus
    """
    frequencies, bin_edges = np.histogram(dataframe[varname], bins=150)
    # Calculate conditional class ratio for each bin
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    class_ratios = []
    for i in range(len(bin_edges) - 1):
        bin_indices = (dataframe[varname] >= bin_edges[i]) & (dataframe[varname] < bin_edges[i + 1])
        bin_y = dataframe['FCSStaus'][bin_indices]
        class_ratio = np.mean(bin_y)
        class_ratios.append(class_ratio)
    # Create the scatter plot
    plt.scatter(bin_centers, class_ratios)
    plt.xlabel(f'{varname}')
    plt.ylabel('Conditional Class Ratio')
    plt.title('Scatter Plot of Conditional Class Ratio')
    plt.show()
    
def XGBmodelfitTime(alg, dtrain, predictors, X_test, Y_test,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    """Fit the XGBoost automatically and return the needed matrics and importance figure 
    this function is a modified version of function in: 
    https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/ 
    !!! Important: the response variable name is FCSStaus !!!
    
    Args:
        alg (xgboost): xgboost model
        dtrain (dataframe): dataframe of training data (includes the response variable)
        predictors (list): predictor list 
        X_test (dataframe): X testing data 
        Y_test (dataframe): Y testing data
        useTrainCV (bool, optional): if use the CV to find the best n_estimator. Defaults to True.
        cv_folds (int, optional): CV folds. Defaults to 5.
        early_stopping_rounds (int, optional): for iteration. Defaults to 50. 
        
    Return: 
        alg, accuracy, auc, recall 
    """
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain["FCSStaus"].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain['FCSStaus'])
        
    #Predict training set:
    dtest_predprob_pre = alg.predict_proba(X_test[predictors])[:,1]
    dtest_predictions_pre = alg.predict(X_test[predictors])

    accuracy = metrics.accuracy_score(Y_test, dtest_predictions_pre)
    auc = metrics.roc_auc_score(Y_test, dtest_predprob_pre)
    recall = metrics.recall_score(Y_test, dtest_predictions_pre)

    return alg, accuracy, auc, recall 

def timeplot(xaxis,RF, xgb, LR, LR_resp, path, plotname): 
    rfvar = np.std(RF).round(2)
    xgbvar = np.std(xgb).round(2)
    lrvar = np.std(LR).round(2)
    lrvarresp = np.std(LR_resp).round(2)
    plt.figure(figsize=(8,6))
    plt.scatter(xaxis, RF, label = f"Random Forest, std ({rfvar})", color ='blue')
    plt.scatter(xaxis, xgb, label = f"XGBoost, std ({xgbvar})", color ='orange', marker='x')
    plt.scatter(xaxis, LR, label = f"Logistic Regression, std ({lrvar})", color ='green', marker='v')
    plt.scatter(xaxis, LR_resp, label = f"Logistic Regression ADASYN, std ({lrvarresp})", color ='red', marker='v')
    rfmean = np.mean(RF).round(2)
    xgbmean = np.mean(xgb).round(2)
    lrmean = np.mean(LR).round(2)
    lrmeanresp = np.mean(LR_resp).round(2)
    plt.hlines(y = rfmean,xmin=xaxis[0], xmax=xaxis[-1] ,label=f"Random Forest, mean ({rfmean})",linestyle='--')
    plt.hlines(y = xgbmean, xmin=xaxis[0], xmax=xaxis[-1],label=f"XGBoost, mean ({xgbmean})", linestyle='dotted',
            color='orange')
    plt.hlines(y = lrmean, xmin=xaxis[0], xmax=xaxis[-1],label=f"Logistic Regression, mean ({lrmean})", linestyle='--',
            color='green')
    plt.hlines(y = lrmeanresp, xmin=xaxis[0], xmax=xaxis[-1],label=f"Logistic Regression ADASYN, mean ({lrmeanresp})", linestyle='--',
        color='red')
    index_start = xaxis.index('2020_2')
    index_end = xaxis.index('2020_7')
    # Calculate the middle index
    middle_index = (index_start + index_end) // 2
    # Add a vertical line at the middle index
    plt.axvline(x=middle_index, color='red', linestyle='--', alpha = 0.5)
    plt.ylim(0, 1)
    # Rotate x-axis labels (optional)
    plt.xticks(rotation=45)
    # Add labels and title
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title(plotname)
    # Add a legend
    plt.legend(loc = 'lower right')
    plt.grid(which = "major", linewidth = 1)
    plt.grid(which = "minor", linewidth = 0.3)
    plt.minorticks_on()
    plt.savefig(path)
    
def tplot(xaxis,RF, xgb, LR, LR_ADA, alpha, bootLR, bootRF, bootXGB, bootLRADA, path): 
    dates = xaxis
    formatted_dates0 = []
    formatted_dates1 = []
    formatted_dates = []
    formatted_dates2 = []
    for date_str in dates:
        year, month = date_str.split('_')
        date0 = datetime(int(year), int(month), 1)
        date1 = datetime(int(year), int(month), 1)+ timedelta(days=5)
        date = datetime(int(year), int(month), 1) + timedelta(days=10)
        date2 = datetime(int(year), int(month), 1) + timedelta(days=15)
        formatted_dates0.append(date0)
        formatted_dates1.append(date1)
        formatted_dates.append(date)
        formatted_dates2.append(date2)
    plotdata = {}
    RF1 = np.array(RF)
    xgb1 = np.array(xgb)
    LR1 = np.array(LR)
    LRADA1 = np.array(LR_ADA)
    for data, name in zip([bootLR, bootRF, bootXGB, bootLRADA], ['LR','RF','XGB', 'LRADA']): 
        upperList = []
        lowerList = []
        for i in data.keys():
                stats = data[i]
                p = ((1.0-alpha)/2.0) * 100
                lower = np.percentile(stats, p)
                p = (alpha+((1.0-alpha)/2.0)) * 100
                upper = np.percentile(stats, p)
                upperList.append(upper)
                lowerList.append(lower)
        plotdata[f"{name}_upper"] = np.array(upperList)
        plotdata[f"{name}_lower"] = np.array(lowerList)
    plt.figure(figsize=(8,6))

    plt.errorbar(formatted_dates0,RF1, yerr=np.array([np.abs(RF1 - plotdata['RF_upper']), RF1 - plotdata['RF_lower'] ]), 
                 color ='blue',linestyle='none', marker='o', label = "Random Forest"  )
    plt.errorbar(formatted_dates1,xgb1, yerr=np.array([np.abs(xgb1 - plotdata['XGB_upper']), xgb1 - plotdata['XGB_lower']]), 
                 color ='orange',linestyle='none', marker='x', label = "XGBoost" )
    plt.errorbar(formatted_dates,LR1, yerr=np.array([np.abs(LR1 - plotdata['LR_upper']), LR1 - plotdata['LR_lower']]), 
                 color ='green', linestyle='none', marker='v', label = "Logistic Regression" )
    plt.errorbar(formatted_dates2,LRADA1, yerr=np.array([np.abs(LRADA1 - plotdata['LRADA_upper']),
                                                         LRADA1 - plotdata['LRADA_lower']]), 
                color ='red', linestyle='none', marker='o', label = "Logistic Regression ADASYN" )
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend(loc = 'lower right')
    plt.grid(which = "major", linewidth = 1)
    plt.grid(which = "minor", linewidth = 0.3)
    plt.minorticks_on()
    plt.savefig(path)