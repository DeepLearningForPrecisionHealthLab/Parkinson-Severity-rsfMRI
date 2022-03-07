#!/usr/bin/env python
"""
After running the model searches, identify the best performing configuration for
each target and feature type. Evaluate these configurations on the outer CV test folds. 

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
from ppmiutils import dataset
import numpy as np
import nilearn as nil
import nilearn.datasets
import nilearn.regions as nilregions
from sklearn import metrics, model_selection
import matplotlib.pyplot as plt

metadata = dataset.ImageData()
clinical = dataset.ClinicalData()
dictBasc = nilearn.datasets.fetch_atlas_basc_multiscale_2015()
strSchaeferPath = '../../atlases/Schaefer/Schaefer2018_100Parcels_7Networks_w_SubCortAtlas_MNI152_2mm.nii'
dictAtlases = {'basc197': dictBasc['scale197'],
               'basc444': dictBasc['scale444'],
               'schaefer': strSchaeferPath
               }

for strTargetTime in ['base', '1y', '2y', '4y']:
    lsResults = []
    for strFeature in ['alff', 'falff', 'ReHo']:

        for strAtlas, strAtlasPath in dictAtlases.items():
            strInputsCsv = f'UPDRS_total_{strTargetTime}_abs/{strFeature}_{strAtlas}_motioncorrected/inputs.csv'
            if not os.path.exists(strInputsCsv):
                continue
            dfInputs = pd.read_csv(strInputsCsv, index_col=0)
            strTargetsCsv = f'UPDRS_total_{strTargetTime}_abs/{strFeature}_{strAtlas}_motioncorrected/targets.csv'
            dfTarget = pd.read_csv(strTargetsCsv, index_col=0, header=None)
            arrTarget = dfTarget.values.flatten()

            for strModel in ['ElasticNet', 'LinearSVR', 'GradientBoostingRegressor', 'RandomForestRegressor']:
                try:
                    with open(f'UPDRS_total_{strTargetTime}_abs/'
                                f'{strFeature}_{strAtlas}_motioncorrected/{strModel}_results.pkl',
                                'rb') as f:
                        dictModel = pickle.load(f)
                except FileNotFoundError:
                    continue
                outer = model_selection.LeaveOneOut()
                arrPredictions = np.zeros(dfInputs.shape[0])
                arrTrainR2 = np.zeros_like(arrPredictions)
                arrTrainRMSE = np.zeros_like(arrPredictions)
                arrValR2 = np.zeros_like(arrPredictions)
                arrValRMSE = np.zeros_like(arrPredictions)

                for i, (arrTrainIdx, arrTestIdx) in enumerate(outer.split(dfInputs, arrTarget)):
                    dfInputsTest = dfInputs.iloc[arrTestIdx]
                    dfInputsTrain = dfInputs.iloc[arrTrainIdx]
                    model = dictModel['estimator'][i].best_estimator_
                    arrPredictions[i] = model.predict(dfInputsTest.astype(np.float64))
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

                lsResults += [{'Feature': strFeature,
                                'Atlas': strAtlas,
                                'Model': strModel,
                                'Test R2' : metrics.r2_score(arrTarget, arrPredictions),
                                'Test RMSE': np.sqrt(metrics.mean_squared_error(arrTarget, arrPredictions)),
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
    dfResults.to_excel(f'results/all_results_{strTargetTime}.xlsx')