#!/usr/bin/env python
"""
Functions for handling the PPMI CSV files

Copyright (c) 2021 The University of Texas Southwestern Medical Center. See LICENSE.md for details.
"""
import glob
import os
import scipy.io
import numpy as np
import pandas as pd
import itertools

__author__ = 'Kevin P Nguyen'
__email__ = 'kevin3.nguyen@utsouthwestern.edu'

# Conversion from visit codes to months in the study
dictVisitCodes = {'BL': 'Baseline',
                  'V01': 'Month 3',
                  'V02': 'Month 6',
                  'V03': 'Month 9',
                  'V04': 'Month 12',
                  'V05': 'Month 18',
                  'V06': 'Month 24',
                  'V07': 'Month 30',
                  'V08': 'Month 36',
                  'V09': 'Month 42',
                  'V10': 'Month 48',
                  'V11': 'Month 54',
                  'V12': 'Month 60',
                  'V13': 'Month 72',
                  'V14': 'Month 84',
                  'V15': 'Month 96',
                  'ST': 'Symptomatic Therapy',
                  'U01': 'Unscheduled Visit 01',
                  'SC': 'Screening',
                  'PW': 'Premature Withdrawawl',
                  'RS1': 'Re-Screen'}
dictVisitToMonth = {'BL': 0,
                    'V01': 3,
                    'V02': 6,
                    'V03': 9,
                    'V04': 12,
                    'V05': 18,
                    'V06': 24,
                    'V07': 30,
                    'V08': 36,
                    'V09': 42,
                    'V10': 48,
                    'V11': 54,
                    'V12': 60,
                    'V13': 72,
                    'V14': 84,
                    'V15': 96,
                    'ST': -1,
                    'U01': -1,
                    'SC': -2,
                    'PW': -3}


def convert_visit_month(visitCodes):
    """
    Convert visit codes to full names.

    :param visitCodes: some iterable containing subject visit codes
    :type visitCodes:
    :return: list of visit names 
    """
    try:
        lsMonths = [dictVisitCodes[strCode] for strCode in visitCodes]
    except KeyError:
        raise KeyError('Invalid visit code found')

    return lsMonths


def extract_conn_fc(strCONNDir, strConditionName=None, roiLabels=None):
    """
    Extract the FC matrices from the CONN .mat files that were generated by running the DLLab CONN Pipeline.
    Vectorize the upper triangle values from each subject, and return a dataframe.

    :param strConditionName:
    :type strConditionName:
    :param strCONNDir:
    :type strCONNDir:
    :return:
    :rtype:
    """
    if strConditionName is None:
        strConditionName = ''
    lsFCFiles = glob.glob(strCONNDir + os.sep + '*' + os.sep + 'resultsROI*Condition{}*'.format(strConditionName))
    lsFCFiles.sort()
    dictFC = {}
    for strFCFile in lsFCFiles:
        strID = strFCFile.split('/')[-2]
        dictMat = scipy.io.loadmat(strFCFile)
        nROIs = dictMat['names'][0].size
        arr2DFC = dictMat['Z']
        arr2DFC = arr2DFC[:nROIs, :nROIs]
        arr1DFC = arr2DFC[np.triu_indices_from(arr2DFC, 1)]
        dictFC[strID] = arr1DFC

    # Generate FC matrix edge names combinatorally from the ROI labels
    if roiLabels is None:
        # Get the ROI labels generated by CONN
        roiLabels = list(dictMat['names'][0])
    lsEdgeLabels = list(itertools.combinations(roiLabels, 2))

    dfFC = pd.DataFrame(dictFC).T
    dfFC.columns = lsEdgeLabels
    return dfFC
