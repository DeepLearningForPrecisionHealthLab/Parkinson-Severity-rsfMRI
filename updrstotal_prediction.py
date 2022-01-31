#!/usr/bin/env python
"""
Train a predictor of UPDRS from either ReHo or fALFF features. 

Given a particular fMRI measurement type (ReHo or fALFF), brain atlas name
(basc197, basc444, or schaefer), and target timepoint (baseline, 1y, 2y, or 4y
UPDRS), do the following:

1. Find preprocessed ReHo and fALFF spatial maps
2. Apply the given brain atlas to obtain mean regional values 
3. Add clinical and demographic covariates
4. Set-up nested cross-validation with a leave-one-out outer loop and 10-fold
   inner loop.
5. Perform hyperparameter optimization for elastic net, SVM, random forest, and
   gradient boosting regressors using a random search. Select the best
   hyperparameter configurations based on the inner 10-fold cross-validation and
   evaluate on the outer cross-validation loop. 
6. Save out results to a given directory.

Example:
python updrstotal_prediction.py --feature ReHo --atlas schaefer --time 2y --outdir <path to output location>

See python updrstotal_prediction.py --help for descriptions and valid options for each argument.

"""
__author__ = 'Kevin P Nguyen'
__email__ = 'kevin3.nguyen@utsouthwestern.edu'


import glob, re, argparse, sys, os
import nilearn as nil
import nilearn.regions as nilregions
from ppmiutils import shallowmodels, dataset
import termcolor
import pandas as pd
import numpy as np
from sklearn import linear_model, ensemble, svm, model_selection
from scipy import stats

metadata = dataset.ImageData()
clinical = dataset.ClinicalData()
dictBasc = nil.datasets.fetch_atlas_basc_multiscale_2015()
strSchaeferPath = '../../atlases/Schaefer/Schaefer2018_100Parcels_7Networks_w_SubCortAtlas_MNI152_2mm.nii'
# Paths to brain atlases
dictAtlases = {'basc197': dictBasc['scale197'],
               'basc444': dictBasc['scale444'],
               'schaefer': strSchaeferPath
               }

def run_search(strMeasure, strAtlasName, strTargetTime, strOutDir):

    strOutDir = os.path.join(strOutDir,    
        f'UPDRS_total_{strTargetTime}_abs/{strMeasure}_{strAtlasName}')
    
    # check if already run
    if os.path.exists(os.path.join(strOutDir, 'summary.xlsx')):
        return

    print(termcolor.colored(f'Using {strMeasure} and {strAtlasName} atlas', 'green',
                            'on_grey'))

    # Find all image files for the selected fMRI measure
    lsFeaturePaths = glob.glob(
        f'../../data/PPMI_regional_dllab/motioncorrected/sub-*/ses-*/results/{strMeasure}.nii.gz')
    lsFeaturePaths.sort()
    # Parse subject/session name from file path
    lsSessions = [re.search(r'(sub-\d*/ses-\d*)/', s).group(1) for s in lsFeaturePaths]
    # Compute region mean values based on the selected brain atlas    
    strAtlasPath = dictAtlases[strAtlasName]
    imgAtlas = nil.image.load_img(strAtlasPath)
    imgAtlasResamp = nil.image.resample_to_img(imgAtlas, lsFeaturePaths[0], interpolation='nearest')
    arrRegions, _ = nilregions.img_to_signals_labels(lsFeaturePaths, imgAtlasResamp)
    # Construct dataframe
    dfFeatures = pd.DataFrame(arrRegions, index=lsSessions)
    dfSessionInfo = metadata.dfFunc.loc[lsSessions]
    # Drop duplicate subjects. We only want to use the earliest scan for each subject.
    dfSessionInfo.drop_duplicates('sub', inplace=True)
    # Keep only PD subjects
    dfSessionInfo = dfSessionInfo.loc[dfSessionInfo['group'].isin(['PD', 'GenCohort PD'])]

    # Get UPDRS subscores
    dfUPDRS1 = clinical.get_updrs_total_longitudinal('1', dfSelectedSubjects=dfSessionInfo)
    dfUPDRS1Q = clinical.get_updrs_total_longitudinal('1Q', dfSelectedSubjects=dfSessionInfo)
    dfUPDRS2 = clinical.get_updrs_total_longitudinal('2Q', dfSelectedSubjects=dfSessionInfo)
    dfUPDRS3 = clinical.get_updrs_total_longitudinal('3', dfSelectedSubjects=dfSessionInfo)
    dfUPDRS4 = clinical.get_updrs_total_longitudinal('4', dfSelectedSubjects=dfSessionInfo).fillna(0)
    dfUPDRS = pd.DataFrame()
    # Add up subscores to get UPDRS total score
    for strTime in ['base', '1y', '2y', '4y']:
        dfUPDRS['updrstot_' + strTime] = dfUPDRS1['updrs1tot_' + strTime] + dfUPDRS1Q['updrs1Qtot_' + strTime]\
                                         + dfUPDRS2['updrs2Qtot_' + strTime] + dfUPDRS3['updrs3tot_' + strTime] \
                                         + dfUPDRS4['updrs4tot_' + strTime]
    dfUPDRS.index = dfSessionInfo.index

    dfTarget = dfUPDRS['updrstot_' + strTargetTime]
    dfTarget.dropna(inplace=True)
    dfSessionInfoValid = dfSessionInfo.loc[dfTarget.index]
    print(termcolor.colored(f'{dfTarget.shape[0]} subjects', 'blue', 'on_grey'))

    # Add clinical and demographic confounders
    dfDemo = clinical.get_demographics(dfSessionInfoValid)
    dfDemo.index = dfSessionInfoValid.index
    dfPDFeatures = clinical.get_pd_features(dfSessionInfoValid)
    dfPDFeatures.index = dfSessionInfoValid.index
    dfMoca = clinical.get_moca_longitudinal(dfSessionInfoValid)
    dfMoca.index = dfSessionInfoValid.index
    dfGds = clinical.get_gds_total_longitudinal(dfSessionInfoValid)
    dfGds.index = dfSessionInfoValid.index

    # Concatenate all features
    dfInputs = pd.concat([dfFeatures.loc[dfTarget.index], dfDemo, dfPDFeatures, dfMoca['moca_base'], dfGds['gdstotal_base']],
                         axis=1)
    # For non-baseline prediction target times, add baseline UPDRS as another feature
    if strTargetTime != 'base':
        dfInputs = pd.concat([dfInputs, dfUPDRS['updrstot_base'].loc[dfTarget.index]], axis=1)

    # Setup cross-validation: leave-one-out outer loop and K-fold inner loop
    outer = model_selection.LeaveOneOut()
    inner = shallowmodels.StratifiedKFoldContinuous(n_splits=10, n_bins=3, shuffle=True, random_state=989)
    
    # Save data to .csv
    os.makedirs(strOutDir, exist_ok=True)
    dfInputs.to_csv(os.path.join(strOutDir, 'inputs.csv'))
    dfTarget.to_csv(os.path.join(strOutDir, 'targets.csv'))
    arrTarget = dfTarget.values.flatten()
    
    # Define models and hyperparameter ranges for random search
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

    regression.run_all_models(nIters=100, nJobs=16)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature', '-f', type=str, choices=['ReHo', 'falff'], help='fMRI measurement type')
    parser.add_argument('--atlas', '-a', type=str, choices=['basc197', 'basc444', 'schaefer'], help='Brain atlas name')
    parser.add_argument('--time', '-t', type=str, choices=['base', '1y', '2y', '4y'], help='Prediction target time')
    parser.add_argument('--outdir', '-o', type=str, help='Output base directory')
    args = parser.parse_args()
    print(args.feature, args.atlas, args.time)
    run_search(args.feature, args.atlas, args.time, args.outdir)
