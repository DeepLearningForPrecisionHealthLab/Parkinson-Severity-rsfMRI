#!/usr/bin/env python
"""
Docstring
______
3/5/19
"""

import os
import sys
import argparse
from scipy import stats
import numpy as np
import pandas as pd
from sklearn import model_selection, pipeline, preprocessing, impute, feature_selection, metrics, svm, linear_model, \
    ensemble
import pickle
import copy

__author__ = 'Kevin P Nguyen'
__email__ = 'kevin3.nguyen@utsouthwestern.edu'


class StratifiedKFoldContinuous(model_selection.StratifiedKFold):
    def __init__(self, n_splits=3, n_bins=5, shuffle=False, random_state=None):
        self.n_bins = n_bins
        super(StratifiedKFoldContinuous, self).__init__(n_splits, shuffle, random_state)

    def split(self, X, y, groups=None):
        yBinned = pd.qcut(y, self.n_bins, labels=False, duplicates='drop')
        # yBinned.index = y.index
        return super(StratifiedKFoldContinuous, self).split(X, yBinned, groups)

class StratifiedShuffleSplitContinuous(model_selection.StratifiedShuffleSplit):
    def __init__(self, n_splits=3, test_size=0.2, n_bins=5, random_state=None):
        self.n_bins = n_bins
        super(StratifiedShuffleSplitContinuous, self).__init__(n_splits, test_size, random_state=random_state)

    def split(self, X, y, groups=None):
        yBinned = pd.qcut(y, self.n_bins, labels=False)
        # yBinned.index = y.index
        return super(StratifiedShuffleSplitContinuous, self).split(X, yBinned, groups)

def rmse(true, predict):
    return np.sqrt(np.mean(np.square(true - predict)))


def rsquare(true, predict):
    ssTot = np.sum(np.square(true - np.mean(true)))
    ssRes = np.sum(np.square(true - predict))
    return 1 - (ssRes / (ssTot + np.finfo(float).eps))


class RegressorPanel:
    outerCV: model_selection.BaseCrossValidator
    innerCV: model_selection.BaseCrossValidator

    def __init__(self, data, target,
                 strOutputDir=None,
                 outerCV=3,
                 innerCV=5,
                 strSortBy='rsquare',
                 randomSeed=432,
                 cvGroups=None):
        """
        Set of shallow learning models for regression.

        :param data: dataframe of input data, or a path to a file containing the the input data (CSV or pkl)
        :type data: pandas.DataFrame
        :param target:
        :type target: pandas.DataFrame
        :param strOutputDir:
        :type strOutputDir:
        :param outerCV:
        :type outerCV:
        :param innerCV:
        :type innerCV:
        :param strSortBy: 'rsquare' or 'rmse'; which metric to use to select the best model in each random search
        :type strSortBy: str
        :param randomSeed:
        :type randomSeed:
        """

        self.data = data
        self.target = target
        self.strOutputDir = strOutputDir

        if isinstance(outerCV, int):
            self.outerCV = model_selection.KFold(n_splits=outerCV, shuffle=True, random_state=randomSeed)
        else:
            self.outerCV = outerCV
        if isinstance(innerCV, int):
            self.innerCV = model_selection.KFold(n_splits=innerCV, shuffle=True, random_state=randomSeed)
        else:
            self.innerCV = innerCV

        self.dictModels = None
        self.set_default_models()
        self.preprocessing = None
        self.set_default_preprocessing()
        self.featureSelection = None
        self.strSortBy = strSortBy
        self.randomSeed = randomSeed
        self.cvGroups = cvGroups

    def set_models(self, *args):
        """
        Select scikit-learn models to train. Pass in any number of tuples, where each tuple contains a scikit-learn
        regression model class and the dictionary of hyperparameter ranges to search.
        For example, set_models((sklearn.linear_model.Lasso, {'alpha': scipy.stats.uniform(0.1, 10.0),
                                                            'max_iter': scipy.stats.randint(500, 5000)},
                                sklearn.ensemble.RandomForestRegressor, {'n_estimators': stats.randint(200, 12000),
                                                                      'min_samples_split': stats.uniform(0.01, 0.5),
                                                                      'min_samples_leaf': stats.randint(1, 6),
                                                                      'max_depth': stats.randint(1, 10)}
                                )
        If models are not specified, the default models and hyperparameters will be used.

        :param args: tuples containing (regression model, hyperparameter dictionary) pairs
        :type args: tuple
        :return:
        :rtype:
        """
        self.dictModels = {tupModel[0].__name__: tupModel for tupModel in args}

    def set_default_models(self):
        self.dictModels = {'Lasso': (linear_model.Lasso(fit_intercept=True, max_iter=5000),
                                     {'alpha': np.logspace(-2, 2, 1000)}
                                     ),
                           'Ridge': (linear_model.Ridge(fit_intercept=True, max_iter=5000),
                                     {'alpha': np.logspace(-2, 2, 1000)}
                                     ),
                           'ElasticNet': (linear_model.ElasticNet(max_iter=5000),
                                          {'alpha': np.logspace(-2, 2, 1000),
                                           'l1_ratio': stats.uniform(0, 1.0)}
                                          ),
                           'LinearSVR': (svm.LinearSVR(tol=0.001, max_iter=50000),
                                         {'C': np.logspace(-2, 3, 1000, base=10),
                                          'epsilon': np.logspace(-2, 0, 1000),
                                          'loss': ['epsilon_insensitive', 'squared_epsilon_insensitive']}
                                         ),
                           'RBFSVR': (svm.SVR(kernel='rbf', tol=0.001),
                                      {'gamma': np.logspace(-4, 0, 1000),
                                       'C': np.logspace(-3, 3, 1000, base=10),
                                       'epsilon': np.logspace(-2, 0, 1000)}
                                      ),
                           'PolySVR': (svm.SVR(kernel='poly'),
                                       {'gamma': np.logspace(-4, 0, 1000),
                                        'degree': [2, 3, 4]}
                                       ),
                           'RandomForestRegressor': (ensemble.RandomForestRegressor(criterion='mse', random_state=432),
                                                     {'n_estimators': np.logspace(1, 4, 100).astype(int),
                                                      'min_samples_split': stats.uniform(0.01, 0.5),
                                                      'min_samples_leaf': stats.randint(1, 6),
                                                      'max_depth': stats.randint(1, 10)}
                                                     ),
                           'AdaBoostRegressor': (ensemble.AdaBoostRegressor(base_estimator=None, loss='linear',
                                                                            random_state=432),
                                                 {'n_estimators': np.logspace(1, 4, 100).astype(int),
                                                  'learning_rate': stats.uniform(0.001, 5)}
                                                 ),
                           'GradientBoostingRegressor': (ensemble.GradientBoostingRegressor(loss='ls',
                                                                                            criterion='friedman_mse'),
                                                         {'learning_rate': np.logspace(-2, -1, 1000),
                                                          'n_estimators': np.logspace(1, 4, 100).astype(int),
                                                          'min_samples_split': stats.uniform(0.01, 0.5),
                                                          'min_samples_leaf': stats.randint(1, 6),
                                                          'max_depth': stats.randint(1, 6)}
                                                         )
                           }

    def set_preprocessing(self, pipeline):
        """
        Set the input data preprocessing  pipeline that will be applied to all data before it is passed into a
        regression model. Typically, this pipeline would contain some kind of scaling/normalization and NaN imputer.
        If no pipeline is specified, the default will be used:
            pipeline.Pipeline([('scaler', preprocessing.StandardScaler()),
                                  ('imputer', impute.SimpleImputer()),
                                  ('selector', feature_selection.SelectKBest(feature_selection.f_regression))])

        :param pipeline: a scikit-learn pipeline object.
        :type pipeline: sklearn.pipeline.Pipeline
        :return:
        :rtype:
        """
        self.preprocessing = pipeline

    def set_default_preprocessing(self):
        self.preprocessing = pipeline.Pipeline([('StandardScaler', preprocessing.StandardScaler()),
                                                ('SimpleImputer', impute.SimpleImputer())])

    def set_feature_selection(self, tupFeatSelection):
        """
        Add a feature selection method that will be used after preprocessing and before model fitting. Takes a tuple
        containing (sklearn feature selection class, dictionary of hyperparams for this selector)

        :param tupFeatSelection: tuple containing (feature selection class, dictionary of hyperparams)
        :type tupFeatSelection: tuple
        :return:
        :rtype:
        """
        self.featureSelection = tupFeatSelection

    def run_single_model(self, strModelName, nIters=100, nJobs=1):
        """
        Run a random search on one model

        :param strModelName: key referring to one of the models in self.dictModels
        :type strModelName: str
        :param nIters: number of iterations in the random search
        :type nIters: int
        :param nJobs: number of parallel jobs
        :type nJobs: int
        :return: dictResults: dictionary containing mean train, val, and test RMSE and Rsquare; dictCVScores:
        dictionary containing the exhaustive results returned by sklearn.model_selection.cross_validate
        :rtype: dict, dict
        """
        if strModelName not in self.dictModels.keys():
            raise ValueError('Incorrect model name: {}'.format(strModelName))

        pipe = copy.deepcopy(self.preprocessing)

        if self.featureSelection is not None:
            strSelectorName = self.featureSelection[0].__class__.__name__
            pipe.steps.append((strSelectorName, self.featureSelection[0]))

        pipe.steps.append((strModelName, self.dictModels[strModelName][0]))

        dictParamsOrig = self.dictModels[strModelName][1]
        # Append the model name to the hyperparam keys so that RandomSearch can recognize them
        dictParams = {strModelName + '__' + strKey: val for strKey, val in dictParamsOrig.items()}

        if self.featureSelection is not None:
            dictSelectorParamsOrig = self.featureSelection[1]
            dictSelectorParams = {strSelectorName + '__' + strKey: val for strKey, val in dictSelectorParamsOrig.items()}
            dictParams = dict(dictParams, **dictSelectorParams)

        dictScorers = {'rmse': metrics.make_scorer(rmse,
                                                   greater_is_better=False),
                       'rsquare': metrics.make_scorer(rsquare,
                                                      greater_is_better=True)}

        random = model_selection.RandomizedSearchCV(pipe, dictParams,
                                                    scoring=dictScorers,
                                                    cv=self.innerCV,
                                                    n_iter=nIters,
                                                    return_train_score=True,
                                                    n_jobs=nJobs,
                                                    iid=False,
                                                    random_state=self.randomSeed,
                                                    refit=self.strSortBy)

        dictCVScores = model_selection.cross_validate(random, X=self.data, y=self.target, cv=self.outerCV,
                                                      groups=self.cvGroups,
                                                      scoring=dictScorers,
                                                      return_train_score=True, return_estimator=True)
        if hasattr(self.outerCV, 'n_splits'):
            nOuterSplits = self.outerCV.n_splits
        elif hasattr(self.outerCV, 'get_n_splits'):
            nOuterSplits = self.outerCV.get_n_splits(self.data)
        elif hasattr(self.outerCV, '__len__'):
            nOuterSplits = self.outerCV.__len__()
        lsTrainRMSE = -dictCVScores['train_rmse']
        lsTrainRSquare = dictCVScores['train_rsquare']
        lsBestModelIdx = [dictCVScores['estimator'][nFold].best_index_ for nFold in range(nOuterSplits)]
        lsValRMSE = [-dictCVScores['estimator'][nFold].cv_results_['mean_test_rmse'][lsBestModelIdx[nFold]] for nFold in
                     range(nOuterSplits)]
        lsValRSquare = [dictCVScores['estimator'][nFold].cv_results_['mean_test_rsquare'][lsBestModelIdx[nFold]] for
                        nFold in range(nOuterSplits)]
        lsTestRMSE = -dictCVScores['test_rmse']
        lsTestRSquare = dictCVScores['test_rsquare']
        dictResults = {'mean_train_rmse': np.mean(lsTrainRMSE),
                       'std_train_rmse': np.std(lsTrainRMSE),
                       'mean_val_rmse': np.mean(lsValRMSE),
                       'std_val_rmse': np.std(lsValRMSE),
                       'mean_test_rmse': np.mean(lsTestRMSE),
                       'std_test_rmse': np.std(lsTestRMSE),
                       'mean_train_rsquare': np.mean(lsTrainRSquare),
                       'std_train_rsquare': np.std(lsTrainRSquare),
                       'mean_val_rsquare': np.mean(lsValRSquare),
                       'std_val_rsquare': np.std(lsValRSquare),
                       'mean_test_rsquare': np.mean(lsTestRSquare),
                       'std_test_rsquare': np.std(lsTestRSquare)
                       }

        return dictResults, dictCVScores

    def run_all_models(self, nIters=100, verbose=True, nJobs=1):
        """
        Run random hyperparam searches on all models

        :param TX: tuple containing which treatment groups to train on, or use 'All' to use all groups
        :type TX: tuple, int
        :param nIters: number of random search iterations
        :type nIters: int
        :param verbose: flag for verbose output
        :type verbose: bool
        :param nJobs: number of parallel jobs
        :type nJobs: int
        :return: result dataframe
        :rtype: pandas.DataFrame
        """

        if verbose:
            print('Writing results to {}'.format(self.strOutputDir))
            if hasattr(self.outerCV, 'n_splits'):
                nOuterSplits = self.outerCV.n_splits
            elif hasattr(self.outerCV, 'get_n_splits'):
                nOuterSplits = self.outerCV.get_n_splits(self.data)
            elif hasattr(self.outerCV, '__len__'):
                nOuterSplits = self.outerCV.__len__()
            else:
                nOuterSplits = '?'

            if hasattr(self.innerCV, 'n_splits'):
                nInnerSplits = self.innerCV.n_splits
            elif hasattr(self.innerCV, '__len__'):
                nInnerSplits = self.innerCV.__len__()
            else:
                nInnerSplits = '?'
            print('Using {} outer folds and {} inner folds'.format(nOuterSplits, nInnerSplits))
            print('{} subjects, {} features'.format(self.data.shape[0], self.data.shape[1]))

        dictResultsAll = {}
        if self.strOutputDir is not None:
            os.makedirs(self.strOutputDir, exist_ok=True)

        for strModelName, tupModel in self.dictModels.items():
            if verbose:
                print('Training model {}'.format(strModelName), flush=True)
            dictResults, dictCVScores = self.run_single_model(strModelName, nIters, nJobs)
            dictResultsAll[strModelName] = dictResults
            if self.strOutputDir:
              with open(self.strOutputDir + '/{}_results.pkl'.format(strModelName), 'wb') as file:
                  pickle.dump(dictCVScores, file)
            if verbose:
                print('{}: \nmean val rmse: {} \nmean test rmse: {}' \
                      '\nmean val rsquare: {} \nmean test rsquare: {}'.format(strModelName,
                                                                              dictResults['mean_val_rmse'],
                                                                              dictResults['mean_test_rmse'],
                                                                              dictResults['mean_val_rsquare'],
                                                                              dictResults['mean_test_rsquare']),
                      flush=True)

        dfSummary = pd.DataFrame(dictResultsAll).T
        if self.strOutputDir:
            writer = pd.ExcelWriter(self.strOutputDir + '/summary.xlsx')
            dfSummary.to_excel(writer)
            writer.close()

        return dfSummary


def plot_hyperparams(strResultsFile, strMetric='rmse', nCols=5):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pickle
    with open(strResultsFile, 'rb') as f:
        dictResults = pickle.load(f)

    lsFigs = []
    for idxF, fold in enumerate(dictResults['estimator']):
        dfFold = pd.DataFrame(fold.cv_results_)
        lsHyperparams = [col for col in dfFold.columns if col.startswith('param_')]
        nHyperparams = len(lsHyperparams)
        if nCols > nHyperparams:
            nCols = nHyperparams
        nRows = np.ceil(nHyperparams / nCols)

        fig, ax = plt.subplots(int(nRows), nCols, figsize=(12, 4*nRows), dpi=600)
        for idx, strHyperparam in enumerate(lsHyperparams):
            if isinstance(ax, np.ndarray):
                axis = ax.flatten()[idx]
            else:
                axis = ax

            dfFold.plot(strHyperparam, 'mean_test_' + strMetric, ax=axis, style='.', legend=False)
            axis.set_xlabel(strHyperparam.split('__')[-1])
            axis.set_ylabel('mean_test_' + strMetric)
        fig.suptitle('Fold {}'.format(idxF))
        lsFigs += [fig]
    return lsFigs

if __name__ == '__main__':
    # Specify input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('data', metavar='D', type=str, help='Path to dataframe file (csv or pkl)')
    parser.add_argument('--target', '-t', type=str, required=True, help='Path to data labels (regression target) file')
    parser.add_argument('--output', '-o', type=str, default='none', help='Path to output directory. Default is the '
                                                                         'same directory as the data file')
    parser.add_argument('--outerfolds', '-f', type=int, default=3, help='Number of outer CV folds')
    parser.add_argument('--innerfolds', '-g', type=int, default=5, help='Number of inner CV folds')
    parser.add_argument('--featselection', '-s', type=str, default='n', help='y/n, include a F regression feature '
                                                                             'selector in the preprocessing pipeline')
    parser.add_argument('--nIters', '-n', type=int, default=100, help='Number of iterations in each random search')
    parser.add_argument('--nJobs', '-q', type=int, default=1, help='Number of parallel operations')
    args = parser.parse_args()

    if args.output == 'none':
        args.output = os.path.dirname(args.data)

    regressors = RegressorPanel(args.data,
                                args.target,
                                strOutputDir=args.output,
                                outerCV=args.outerfolds,
                                innerCV=args.innerfolds
                                )
    regressors.set_feature_selection((feature_selection.SelectPercentile(feature_selection.f_regression),
                                      {'percentile': np.linspace(0.05, 0.5, 10)}))
    regressors.run_all_models(nIters=args.nIters, nJobs=args.nJobs)