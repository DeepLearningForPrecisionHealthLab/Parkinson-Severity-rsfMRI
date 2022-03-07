#!/usr/bin/env python
"""
Train a predictor of UPDRS from ReHo and fALFF features. Use leave-one-site out
nested CV. For each outer fold, leave out one of the three largest sites in the
dataset. For the inner CV, use a standard stratified K-fold split.

Copyright (c) 2021 The University of Texas Southwestern Medical Center. See LICENSE.md for details.
"""
__author__ = 'Kevin P Nguyen'
__email__ = 'kevin3.nguyen@utsouthwestern.edu'


import glob, re, argparse, sys, os
from ppmiutils import shallowmodels, dataset
import termcolor
import pandas as pd
import numpy as np
from sklearn import linear_model, ensemble, svm, model_selection
from scipy import stats

def run_search(strMeasure, strAtlasName, strTargetTime):

    # Results directory for original LOOCV run
    strOrigDir = f'UPDRS_total_{strTargetTime}_abs/{strMeasure}_{strAtlasName}_motioncorrected'
    # Output directory
    strOutDir = f'LOSO/UPDRS_total_{strTargetTime}/{strMeasure}_{strAtlasName}_motioncorrected'

    print(termcolor.colored(f'Using {strMeasure} and {strAtlasName} atlas', 'green',
                            'on_grey'))

    dfInputs = pd.read_csv(os.path.join(strOrigDir, 'inputs.csv'), index_col=0)
    dfTarget = pd.read_csv(os.path.join(strOrigDir, 'targets.csv'), index_col=0, header=None)
    dfSites = pd.read_csv('../../data/clinical/Center-Subject_List.csv')
    dfSites.index = dfSites['PATNO']

    # Get the numerical IDs of included subjects
    subjects = [int(re.search(r'sub-(\d*)', x)[1]) for x in dfInputs.index]
    dfSites = dfSites.loc[subjects]
    # Determine test folds depending on study site. 
    #   Fold 0: site 290 (16 subs)
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
    nBins = 2 if strTargetTime == '4y' else 3
    inner = shallowmodels.StratifiedKFoldContinuous(n_splits=10, n_bins=nBins, shuffle=True, random_state=989)
    os.makedirs(strOutDir, exist_ok=True)
    dfInputs.to_csv(os.path.join(strOutDir, 'inputs.csv'))
    dfTarget.to_csv(os.path.join(strOutDir, 'targets.csv'))
    arrTarget = dfTarget.values.flatten()
    regression = shallowmodels.RegressorPanel(dfInputs.astype(np.float64), arrTarget,
                                              strOutDir,
                                              outerCV=outer, innerCV=inner, randomSeed=645, strSortBy='rmse')
    regression.dictModels = {'ElasticNet': (linear_model.ElasticNet(max_iter=5000),
                                            {'alpha': np.logspace(0, 1.5, 1000),
                                             'l1_ratio': stats.uniform(0, 1.0)}
                                            ),
                             'LinearSVR': (svm.LinearSVR(tol=0.001, max_iter=50000),
                                           {'C': np.logspace(-3, 0, 1000, base=10),
                                            'epsilon': np.logspace(-2, 0, 1000),
                                            'loss': ['epsilon_insensitive', 'squared_epsilon_insensitive']}
                                           ),
                             'GradientBoostingRegressor': (ensemble.GradientBoostingRegressor(loss='ls',
                                                                                              criterion='friedman_mse'),
                                                           {'learning_rate': np.logspace(-2, -1, 1000),
                                                            'n_estimators': np.linspace(10, 1000, 1000,
                                                                                        dtype=int),
                                                            'min_samples_split': stats.uniform(0.1, 0.7),
                                                            'min_samples_leaf': stats.randint(1, 6),
                                                            'max_depth': stats.randint(1, 4)}
                                                           ),
                             'RandomForestRegressor': (ensemble.RandomForestRegressor(criterion='mse', random_state=432),
                                                        {'n_estimators': np.logspace(1, 3, 100).astype(int),
                                                        'min_samples_split': stats.uniform(0.01, 0.5),
                                                        'min_samples_leaf': stats.randint(1, 6),
                                                        'max_depth': stats.randint(1, 8)}
                                                       )
                             }

    regression.run_all_models(nIters=100, nJobs=24)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature', '-f', type=str)
    parser.add_argument('--atlas', '-a', type=str)
    parser.add_argument('--time', '-t', type=str)
    args = parser.parse_args()
    print(args.feature, args.atlas, args.time)
    run_search(args.feature, args.atlas, args.time)