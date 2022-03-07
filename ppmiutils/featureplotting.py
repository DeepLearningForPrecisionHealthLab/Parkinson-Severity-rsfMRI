#!/usr/bin/env python
"""
Tools for getting feature importance from trained shallow models and then visualizing in brain space.

Copyright (c) 2021 The University of Texas Southwestern Medical Center. See LICENSE.md for details.
"""
__author__ = 'Kevin P Nguyen'
__email__ = 'kevin3.nguyen@utsouthwestern.edu'

import pickle, sys, os
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from . import dataset
from nilearn import image, plotting


class FeaturePlotter:
    def __init__(self):
        self.sessinfo = dataset.ImageData()
        self.clinical = dataset.ClinicalData()
        self.lsClinLabels = ['Age', 'Male', 'Hispanic', 'Am. Indian/Alask. Nat.', 'Asian', 'Afr-Amer.', 'Pac. Isl.',
                             'Caucasian', 'Other race', 'Years of edu.', 'Right-handed', 'Left-handed', 'Ambidextrous',
                             'Time since diag.', 'Symptom duration', 'Tremor', 'Rigidity', 'Bradykinesia',
                             'Postural Instability', 'Left-dominant symptoms', 'Right-dominant symptoms',
                             'Equal-sided symptoms']
        self.dictClinLabels = {'age': 'Age',
                               'sex_male': 'Male',
                               'hispanic': 'Hispanic',
                               'race_amindalnat': 'Am. Indian/Alask. Nat.',
                               'race_asian': 'Asian',
                               'race_black': 'Afr-Amer.',
                               'race_pacifisl': 'Pac. Isl.',
                               'race_white': 'Caucasian',
                               'race_other': 'Other race',
                               'eduyears': 'Years of edu.',
                               'handed_right': 'Right-handed',
                               'handed_left': 'Left-handed',
                               'handed_both': 'Ambidextrous',
                               'days_dx': 'Time since diag.',
                               'days_sx': 'Symptom duration',
                               'tremor': 'Tremor',
                               'rigid': 'Rigidity',
                               'brady': 'Bradykinesia',
                               'posins': 'Postural Instability',
                               'domside_left': 'Left-dominant symptoms',
                               'domside_right': 'Right-dominant symptoms',
                               'domside_both': 'Equal-sided symptoms',
                               'moca_base': 'Baseline MoCA',
                               'gdstotal_base': 'Baseline GDS',
                               'updrstot_base': 'Baseline MDS-UPDRS'}


    def _values_to_img(self, arrValues, imgAtlas):
        arrAtlas = imgAtlas.get_data()
        arrValuesImg = np.zeros_like(arrAtlas, dtype=np.float)
        if arrValues.size != (arrAtlas.max()):
            raise ValueError('There are {} values and {} ROIs'.format(arrValues.size, arrAtlas.max()))
        for i in range(arrValues.size):
            arrValuesImg[arrAtlas == (i + 1)] = arrValues[i]
        return image.new_img_like(data=arrValuesImg, ref_niimg=imgAtlas)


    def _get_target(self, strTarget, dfSessions, strTime, bDelta=False):
        if strTarget == '3on_score':
            dfUPDRS = self.clinical.get_updrs_total_longitudinal('3', bOnMed=True, dfSelectedSubjects=dfSessions)
        elif strTarget == '3off_score':
            dfUPDRS = self.clinical.get_updrs_total_longitudinal('3', bOnMed=False, dfSelectedSubjects=dfSessions)
        elif strTarget == 'total_score':
            dfUPDRS1 = self.clinical.get_updrs_total_longitudinal('1', dfSelectedSubjects=dfSessions)
            dfUPDRS1Q = self.clinical.get_updrs_total_longitudinal('1Q', dfSelectedSubjects=dfSessions)
            dfUPDRS2 = self.clinical.get_updrs_total_longitudinal('2Q', dfSelectedSubjects=dfSessions)
            dfUPDRS3 = self.clinical.get_updrs_total_longitudinal('3', dfSelectedSubjects=dfSessions)
            dfUPDRS = pd.DataFrame()
            for t in ['base', '1y', '2y', '4y']:
                dfUPDRS['updrstot_' + t] = dfUPDRS1['updrs1tot_' + t] + dfUPDRS1Q['updrs1Qtot_' + t] \
                                                 + dfUPDRS2['updrs2Qtot_' + t] + dfUPDRS3['updrs3tot_' + t]
        if bDelta:
            dfT1 = dfUPDRS.filter(like='tot_' + strTime).iloc[:, 0]
            dfT0 = dfUPDRS.filter(like='tot_base').iloc[:, 0]
            dfTarget = dfT1.astype(np.float64) - dfT0.astype(np.float64)
        else:
            dfTarget = dfUPDRS.filter(like='tot_' + strTime)
        return dfTarget

    def plot_features(self, strModelPkl, strAtlasPath, strRoiLabels,
                      strTarget, strTime, bDelta=False,
                      strImportanceAttr='feature_importances_',
                      nFeaturesPlot=-1):
        strModelDir = os.path.dirname(strModelPkl)
        strInputPath = os.path.join(strModelDir, 'inputs.csv')
        dfFeatures = pd.read_csv(strInputPath, index_col=0)
        imgAtlas = image.load_img(strAtlasPath)

        with open(strRoiLabels, 'r') as f:
            lsRoiLabels = f.read().split('\n')
        nRois = len(lsRoiLabels)
        lsClinLabels = [self.dictClinLabels[s] for s in dfFeatures.columns[nRois:]]
        dfFeatures.columns = np.concatenate((lsRoiLabels, lsClinLabels))

        with open(strModelPkl, 'rb') as f:
            dictModel = pickle.load(f)

        nSplits = len(dictModel['estimator'])
        nFeatures = getattr(dictModel['estimator'][0].best_estimator_.steps[-1][1], strImportanceAttr).shape[-1]
        arrFeatures = np.zeros((nSplits, nFeatures))
        for i in range(len(dictModel['estimator'])):
            model = dictModel['estimator'][i].best_estimator_.steps[-1][1]
            arrFeatures[i,:] = getattr(model, strImportanceAttr)

        dfSessions = self.sessinfo.dfFunc.loc[dfFeatures.index]
        dfTarget = self._get_target(strTarget, dfSessions, strTime, bDelta)

        dfImportance = pd.DataFrame(index=dfFeatures.columns, columns=['Median Feature Importance', 'Correlation'])
        dfImportance['Feature Importance'] = np.abs(np.median(arrFeatures, 0))
        dfImportance['Feature Importance'] /= dfImportance['Feature Importance'].abs().max()
        if strImportanceAttr == 'feature_importances_':
            for n in range(dfImportance.shape[0]):
                arrFeat = dfFeatures.iloc[:, n].values
                arrNan = np.isnan(arrFeat) | np.isnan(dfTarget.astype(np.float64).values.flatten())
                r, p = stats.pearsonr(dfTarget.values.flatten()[~arrNan], arrFeat.astype(np.float64)[~arrNan])

                dfImportance['Correlation'].iloc[n] = 'negative' if r < 0 else 'positive'
            strLegendTitle = 'Correlation with target'
        else:
            dfImportance['Correlation'] = ['positive' if a > 0 else 'negative' for a in np.median(arrFeatures, 0)]
            strLegendTitle = 'Sign of coefficient'
        dfImportance['Feature'] = dfImportance.index
        fig, axis = plt.subplots(1, 1, dpi=600, figsize=(5, int(0.5 * nFeaturesPlot)))
        bars = sns.barplot(y='Feature', x='Feature Importance', hue='Correlation',
                           data=dfImportance.sort_values('Feature Importance', ascending=False).iloc[:nFeaturesPlot],
                           ax=axis, dodge=False,
                           palette=sns.xkcd_palette(['pale red', 'windows blue']), saturation=0.75,
                           hue_order=['positive', 'negative'],
                           orient='h')

        plt.legend(loc='lower right', 
                   # title=strLegendTitle,
                   title='Association with target')
        # for item in bars.get_xticklabels():
        #     item.set_rotation(90)

        arrAtlasVals = dfImportance['Feature Importance'].loc[lsRoiLabels].values
        arrAtlasVals[np.argsort(arrAtlasVals)[:-nFeaturesPlot]] = 0
        arrNeg = np.array(dfImportance['Correlation'].loc[lsRoiLabels].values.flatten() == 'negative')
        arrAtlasVals[arrNeg] *= -1

        imgImportance = self._values_to_img(arrAtlasVals, imgAtlas)
        fig, ax = plt.subplots(3, 1, figsize=(8, 5))
        brainX = plotting.plot_stat_map(imgImportance, cmap='coolwarm', display_mode='x', cut_coords=4, alpha=0.95, axes=ax[0])
        brainY = plotting.plot_stat_map(imgImportance, cmap='coolwarm', display_mode='y', cut_coords=4, alpha=0.95, axes=ax[1])
        brainZ = plotting.plot_stat_map(imgImportance, cmap='coolwarm', display_mode='z', cut_coords=4, alpha=0.95, axes=ax[2])

        return bars, [brainX, brainY, brainZ], arrFeatures, fig

