#!/usr/bin/env python
"""
After running the model searches, identify the best performing configuration for
each target and feature type. Evaluate these configurations on the held-out sites. 

Copyright (c) 2021 The University of Texas Southwestern Medical Center. See LICENSE.md for details.
"""
__author__ = 'Kevin P Nguyen'
__email__ = 'kevin3.nguyen@utsouthwestern.edu'

import pandas as pd
import pickle
import sys
import os
import glob
import re
import numpy as np
import nilearn as nil
import nilearn.regions as nilregions
from sklearn import metrics, model_selection
import matplotlib.pyplot as plt


for strTargetTime in ['base', '1y', '2y', '4y']:
    lsResults = []
    for strFeature in ['alff', 'falff', 'ReHo']:

        for strAtlas in ['basc197', 'schaefer', 'bghat']:
            strInputsCsv = f'LOSO/UPDRS_total_{strTargetTime}/{strFeature}_{strAtlas}_motioncorrected/inputs.csv'
            if not os.path.exists(strInputsCsv):
                continue
            dfInputs = pd.read_csv(strInputsCsv, index_col=0)
            strTargetsCsv = f'LOSO/UPDRS_total_{strTargetTime}/{strFeature}_{strAtlas}_motioncorrected/targets.csv'
            dfTarget = pd.read_csv(strTargetsCsv, index_col=0)
            arrTarget = dfTarget.values.flatten()

            for strModel in ['ElasticNet', 'LinearSVR', 'GradientBoostingRegressor', 'RandomForestRegressor']:
                try:
                    with open(f'LOSO/UPDRS_total_{strTargetTime}/'
                                f'{strFeature}_{strAtlas}_motioncorrected/{strModel}_results.pkl',
                                'rb') as f:
                        dictModel = pickle.load(f)
                except FileNotFoundError:
                    continue
                
                dfSites = pd.read_csv('../../data/clinical/Center-Subject_List.csv')
                dfSites.index = dfSites['PATNO']

                # Get the numerical IDs of included subjects
                subjects = [int(re.search(r'sub-(\d*)', x)[1]) for x in dfInputs.index]
                dfSites = dfSites.loc[subjects]
                # Determine test folds depending on study site. 
                #   Fold 0: site 290 (16 subs at 1y)
                #   Fold 1: site 32 (13 subs)
                #   Fold 2: site 88 (11 subs)
                # all remaining sites are small, so won't use as test folds
                dictSitesToGroups = {290: 0, 32: 1, 88: 2}
                def site_to_group(x):
                    if x in dictSitesToGroups.keys():
                        return dictSitesToGroups[x]
                    else:
                        return -1
                dfSites['test_fold'] = dfSites['CNO'].apply(site_to_group)

                outer = model_selection.PredefinedSplit(dfSites['test_fold'])
                nSplits = 3
                arrPredictions = np.zeros(dfInputs.shape[0])
                arrTrainR2 = np.zeros((nSplits,))
                arrTrainRMSE = np.zeros((nSplits,))
                arrValR2 = np.zeros((nSplits,))
                arrValRMSE = np.zeros((nSplits,))

                for i, (arrTrainIdx, arrTestIdx) in enumerate(outer.split(dfInputs, arrTarget)):
                    dfInputsTest = dfInputs.iloc[arrTestIdx]
                    dfInputsTrain = dfInputs.iloc[arrTrainIdx]
                    model = dictModel['estimator'][i].best_estimator_
                    arrPredictions[arrTestIdx] = model.predict(dfInputsTest.astype(np.float64))
                    arrTrainPredictions = model.predict(dfInputsTrain.astype(np.float64))
                    arrTrainR2[i] = metrics.r2_score(arrTarget[arrTrainIdx], arrTrainPredictions)
                    arrTrainRMSE[i] = np.sqrt(metrics.mean_squared_error(arrTarget[arrTrainIdx], arrTrainPredictions))

                    dfSearchResults = pd.DataFrame(dictModel['estimator'][i].cv_results_)
                    idxBestModel = dictModel['estimator'][i].best_index_
                    arrValR2[i] = dfSearchResults['mean_test_rsquare'].iloc[idxBestModel]
                    arrValRMSE[i] = -dfSearchResults['mean_test_rmse'].iloc[idxBestModel]

                fThresh = 35
                # fThresh = np.median(arrTarget)
                arrClassTrue = arrTarget > fThresh
                arrClassPred = arrPredictions > fThresh
                trueneg, falsepos, falseneg, truepos = metrics.confusion_matrix(arrClassTrue,
                                                                                arrClassPred).ravel()
                # Remove subjects that were not included in any of the held-out sites (not predicted)
                arrTargetTest = arrTarget[arrPredictions != 0]
                arrPredictionsTest = arrPredictions[arrPredictions != 0]

                lsResults += [{'Feature': strFeature,
                                'Atlas': strAtlas,
                                'Model': strModel,
                                'Test R2' : metrics.r2_score(arrTargetTest, arrPredictionsTest),
                                'Test RMSE': np.sqrt(metrics.mean_squared_error(arrTargetTest, arrPredictionsTest)),
                                'Test AUC': metrics.roc_auc_score(arrClassTrue, arrClassPred),
                                'Test precision': metrics.precision_score(arrClassTrue, arrClassPred),
                                'Test recall': metrics.recall_score(arrClassTrue, arrClassPred),
                                'Test accuracy': metrics.accuracy_score(arrClassTrue, arrClassPred),
                                'Test f1': metrics.f1_score(arrClassTrue, arrClassPred),
                                'Test NPV': trueneg / (trueneg + falseneg),
                                'Test specificity': trueneg / (trueneg + falsepos),
                                'Val Mean R2': np.mean(arrValR2),
                                'Val Std R2': np.std(arrValR2),
                                'Val Mean RMSE': np.mean(arrValRMSE),
                                'Val Std RMSE': np.std(arrValRMSE),
                                'Train Mean R2': np.mean(arrTrainR2),
                                'Train Std R2': np.std(arrTrainR2),
                                'Train Mean RMSE': np.mean(arrTrainRMSE),
                                'Train Std RMSE': np.std(arrTrainRMSE)}]
                dictTime = {'base': 'baseline',
                            '1y': 'year 1',
                            '2y': 'year 2',
                            '4y': 'year 4'}

    dfResults = pd.DataFrame(lsResults)
    dfResults.to_excel(f'LOSO/all_results_{strTargetTime}.xlsx')