#!/usr/bin/env python
"""
Functions for quickly loading in clinical data tables

Copyright (c) 2021 The University of Texas Southwestern Medical Center. See LICENSE.md for details.
"""
import os
import pandas as pd
import numpy as np
import datetime as dt
from dateutil import relativedelta
import pathlib
from . import datatools

__author__ = 'Kevin P Nguyen'
__email__ = 'kevin3.nguyen@utsouthwestern.edu'

strPackageDir = os.path.dirname(os.path.realpath(__file__))

class ImageData():
    def __init__(self,
                 strPathFuncData='../data/ppmi_scan_list_fmri_all.csv',
                 strPathDatData='../data/ppmi_scan_metadata_dat_all.csv',
                 strDirFuncSource='/project/bioinformatics/DLLab/STUDIES/PPMI/ImagingData/PPMI_fMRI/Source',
                 strDirDatSource='/project/bioinformatics/DLLab/STUDIES/PPMI/ImagingData/PPMI_DAT/Source'
                 ):
        self.strPathFuncData = os.path.join(strPackageDir, strPathFuncData)
        self.strDirFuncSource = strDirFuncSource
        self.strPathDatData = os.path.join(strPackageDir, strPathDatData)
        self.strDirDatSource = strDirDatSource

        self.dfFunc = self._load_func()
        self.dfDat = self._load_dat()

    def _load_func(self):
        dfData = pd.read_csv(self.strPathFuncData)
        # dfData = dfData.loc[['ep2d' in strDesc or 'MPRAGE' in strDesc for strDesc in dfData['Description']]]
        dfData = dfData.set_index(['Subject ID', 'Visit'])
        dfDataFunc = dfData.loc[dfData['Modality'] == 'fMRI']  # contains just theDatI scans

        # Check the BIDS source directory for the functional and anatomical files
        dfPaths = pd.DataFrame(columns=['sub', 'date', 'visit', 'sex', 'age', 'group', 'func', 'anat'])
        for tupIndex, dfRow in dfDataFunc.iterrows():
            strDate = dfRow['Study Date']
            date = dt.datetime.strptime(strDate, '%m/%d/%Y').date()
            nID = tupIndex[0]
            strVisit = tupIndex[1]

            # Nifti files should be in Source/sub-NNNN/ses-X/func or Source/sub-NNNN/ses-X/anat
            strFuncPath = self.strDirFuncSource + os.sep + 'sub-{}/ses-{}/func/'.format(nID, date.strftime('%Y%m%d'))
            pathFunc = pathlib.Path(strFuncPath)
            strAnatPath = self.strDirFuncSource + os.sep + 'sub-{}/ses-{}/anat/'.format(nID, date.strftime('%Y%m%d'))
            pathAnat = pathlib.Path(strAnatPath)
            if pathFunc.exists() & pathAnat.exists():
                pathFuncNii = pathFunc.glob('*.nii*')
                pathAnatNii = pathAnat.glob('*.nii*')
                dfPaths.loc['sub-{}/ses-{}'.format(nID, date.strftime('%Y%m%d'))] = [nID,
                                                                                     strDate, strVisit,
                                                                                     dfRow['Sex'], dfRow['Age'],
                                                                                     dfRow['Research Group'],
                                                                                     sorted(pathFuncNii)[0],
                                                                                     sorted(pathAnatNii)[0]]
        return dfPaths

    def _load_dat(self):
        dfData = pd.read_csv(self.strPathDatData)
        dfData = dfData.loc[['DaTSCAN' in strDesc for strDesc in dfData['Description']]]
        dfData = dfData.set_index(['Subject ID', 'Visit'])

        # Check the BIDS source directory for the DaT files
        dfPaths = pd.DataFrame(columns=['sub', 'date', 'visit', 'sex', 'age', 'group', 'dat'])
        for tupIndex, dfRow in dfData.iterrows():
            strDate = dfRow['Study Date']
            date = dt.datetime.strptime(strDate, '%m/%d/%Y').date()
            nID = tupIndex[0]
            strVisit = tupIndex[1]

            # Nifti files should be in Source/sub-NNNN/ses-X/func or Source/sub-NNNN/ses-X/anat
            strDatPath = self.strDirDatSource + os.sep + 'sub-{}/ses-{}/dat/'.format(nID, date.strftime('%Y%m%d'))
            pathDat = pathlib.Path(strDatPath)
            if pathDat.exists():
                pathDatNii = pathDat.glob('*.nii*')
                dfPaths.loc['sub-{}/ses-{}'.format(nID, date.strftime('%Y%m%d'))] = [nID,
                                                                                     strDate, strVisit,
                                                                                     dfRow['Sex'], dfRow['Age'],
                                                                                     dfRow['Research Group'],
                                                                                     sorted(pathDatNii)]
        return dfPaths


class ClinicalData:
    def __init__(self,
                 strPathUPDRS1='../data/clinical/MDS_UPDRS_Part_I.csv',
                 strPathUPDRS1Q='../data/clinical/MDS_UPDRS_Part_I__Patient_Questionnaire.csv',
                 strPathUPDRS2Q='../data/clinical/MDS_UPDRS_Part_II__Patient_Questionnaire.csv',
                 strPathUPDRS3='../data/clinical/MDS_UPDRS_Part_III.csv',
                 strPathUPDRS4='../data/clinical/MDS_UPDRS_Part_IV.csv',
                 strPathMoCA='../data/clinical/Montreal_Cognitive_Assessment__MoCA_.csv',
                 strPathPDFeat='../data/clinical/PD_Features.csv',
                 strPathDemo='../data/clinical/Screening___Demographics.csv',
                 strPathGroup='../data/clinical/Patient_Status.csv',
                 strPathGDS='../data/clinical/Geriatric_Depression_Scale__Short_.csv',
                 strPathSocEc='../data/clinical/Socio-Economics.csv'
                ):
        self.strPathUPDRS1 = os.path.join(strPackageDir, strPathUPDRS1)
        self.strPathUPDRS1Q = os.path.join(strPackageDir, strPathUPDRS1Q)
        self.strPathUPDRS2Q = os.path.join(strPackageDir, strPathUPDRS2Q)
        self.strPathUPDRS3 = os.path.join(strPackageDir, strPathUPDRS3)
        self.strPathUPDRS4 = os.path.join(strPackageDir, strPathUPDRS4)
        self.strPathMoCA = os.path.join(strPackageDir, strPathMoCA)
        self.strPathPDFeat = os.path.join(strPackageDir, strPathPDFeat)
        self.strPathDemo = os.path.join(strPackageDir, strPathDemo)
        self.strPathGroup = os.path.join(strPackageDir, strPathGroup)
        self.strPathGDS = os.path.join(strPackageDir, strPathGDS)
        self.strPathSocEc = os.path.join(strPackageDir, strPathSocEc)

        self.dfUPDRS1 = self._remove_u(pd.read_csv(self.strPathUPDRS1, index_col=0))
        self.dfUPDRS1Q = self._remove_u(pd.read_csv(self.strPathUPDRS1Q, index_col=0))
        self.dfUPDRS2Q = self._remove_u(pd.read_csv(self.strPathUPDRS2Q, index_col=0))
        self.dfUPDRS3 = self._remove_u(pd.read_csv(self.strPathUPDRS3, index_col=0))
        self.dfUPDRS4 = self._remove_u(pd.read_csv(self.strPathUPDRS4, index_col=0))
        self.dfMoCA = self._remove_u(pd.read_csv(self.strPathMoCA, index_col=0))
        self.dfPDFeat = self._remove_u(pd.read_csv(self.strPathPDFeat, index_col=0))
        self.dfDemo = self._remove_u(pd.read_csv(self.strPathDemo, index_col=0))
        self.dfGroup = pd.read_csv(self.strPathGroup, index_col=0)
        self.dfGDS = pd.read_csv(self.strPathGDS, index_col=0)
        self.dfSocEc = pd.read_csv(self.strPathSocEc, index_col=0)

    def _remove_u(self, df):
        dfNew = df.replace(to_replace=['u', 'U'], value=np.nan)
        return dfNew

    def get_updrs_total_longitudinal(self, strPart, bOnMed=True, dfSelectedSubjects=ImageData().dfFunc):
        """
        Get a dataframe containing UPDRS total scores at baseline, 1 year, 2 year, and 4 year visits.

        :param strPart: which UPDRS part to load, '1', '1Q', '2Q', '3', or '4'
        :type strPart: str
        :param dfSelectedSubjects: optional, a dataframe containing 'sub' and 'visit' columns that specifies what
        subjects to use and which visit code to consider the baseline. By default, use ppmiutils.dataset.ImageData(
        ).dfFunc to select only subjects that have fMRI images and use the earliest scan date as baseline. If None,
        use all subjects in the UPDRS data and use the "Baseline" visit as the baseline.
        :type dfSelectedSubjects: pandas.DataFrame
        :return: dataframe containing baseline, 2 year, and 4 year UPDRS part III total scores and visit dates
        :rtype: pandas.DataFrame
        """
        dictPrefixes = {'1': 'NP1', '1Q': 'NP1', '2Q': 'NP2', '3': 'NP3', '4': 'NP4'} # column name prefixes
        strPrefix = dictPrefixes[strPart]

        dfUPDRS = getattr(self, 'dfUPDRS{}'.format(strPart)).copy()
        # Filter for on or off-med results, depending on user preference
        if (strPart == '3') & bOnMed:
            dfUPDRS.drop(dfUPDRS.index[(dfUPDRS['PAG_NAME'] == 'NUPDRS3') & \
                         (dfUPDRS['EVENT_ID'].isin(['V04', 'V06', 'V08', 'V10', 'V12']))], inplace=True)
        elif strPart == '3':
            dfUPDRS = dfUPDRS.loc[(dfUPDRS['PAG_NAME'] == 'NUPDRS3') & \
                                  (dfUPDRS['EVENT_ID'].isin(['V04', 'V06', 'V08', 'V10', 'V12']))]

        dfUPDRS['sub'] = dfUPDRS['PATNO']
        if strPart == '3':
            dfUPDRS['date'] = pd.to_datetime(dfUPDRS['INFODT'], format='%b-%y')
        else:
            dfUPDRS['date'] = pd.to_datetime(dfUPDRS['INFODT'])
        dfUPDRS['visit'] = datatools.convert_visit_month(dfUPDRS['EVENT_ID'])
        dfUPDRS['updrs{}tot'.format(strPart)] = dfUPDRS.iloc[:, [strPrefix in strCol for strCol in dfUPDRS.columns]].sum(
            axis=1)

        return self._get_scores_longitudinal(dfUPDRS, 'updrs{}tot'.format(strPart), dfSelectedSubjects)

    def get_updrs_data_longitudinal(self, strPart, bOnMed=True, dfSelectedSubjects=ImageData().dfFunc):
        """
        Get a dataframe containing UPDRS part III scores (individually) at baseline, 1 year, 2 year, and 4 year visits.

        :param strPart: which UPDRS part to load, '1', '1Q', '2Q', '3', or '4'
        :type strPart: str
        :param dfSelectedSubjects: optional, a dataframe containing 'sub' and 'visit' columns that specifies what
        subjects to use and which visit code to consider the baseline. By default, use ppmiutils.dataset.ImageData(
        ).dfFunc to select only subjects that have fMRI images and use the earliest scan date as baseline. If None,
        use all subjects in the UPDRS data and use the "Baseline" visit as the baseline.
        :type dfSelectedSubjects: pandas.DataFrame
        :return: dataframe containing baseline, 2 year, and 4 year UPDRS part III total scores and visit dates
        :rtype: pandas.DataFrame
        """
        dictPrefixes = {'1': 'NP1', '1Q': 'NP1', '2Q': 'NP2', '3': 'NP3', '4': 'NP4'}
        strPrefix = dictPrefixes[strPart]

        dfUPDRS = getattr(self, 'dfUPDRS{}'.format(strPart)).copy()
        # Filter for on or off-med results, depending on user preference
        if (strPart == '3') & bOnMed:
            dfUPDRS.drop(dfUPDRS.index[(dfUPDRS['PAG_NAME'] == 'NUPDRS3') & \
                         (dfUPDRS['EVENT_ID'].isin(['V04', 'V06', 'V08', 'V10', 'V12']))])
        elif strPart == '3':
            dfUPDRS.drop(dfUPDRS.index[(dfUPDRS['PAG_NAME'] == 'NUPDRS3A') & \
                                       (dfUPDRS['EVENT_ID'].isin(['V04', 'V06', 'V08', 'V10', 'V12']))])
        dfUPDRS['sub'] = dfUPDRS['PATNO']
        dfUPDRS['date'] = pd.to_datetime(dfUPDRS['INFODT'], format='%b-%y')
        dfUPDRS['visit'] = datatools.convert_visit_month(dfUPDRS['EVENT_ID'])
        lsCols = dfUPDRS.columns[[strPrefix in strCol for strCol in dfUPDRS.columns]].tolist()

        return self._get_scores_longitudinal(dfUPDRS, lsCols, dfSelectedSubjects)

    def get_updrs_total_score(self, dfSelectedSubjects=ImageData().dfFunc):
        dfUPDRS1 = self.get_updrs_total_longitudinal('1', dfSelectedSubjects=dfSelectedSubjects)
        dfUPDRS1Q = self.get_updrs_total_longitudinal('1Q', dfSelectedSubjects=dfSelectedSubjects)
        dfUPDRS2 = self.get_updrs_total_longitudinal('2Q', dfSelectedSubjects=dfSelectedSubjects)
        dfUPDRS3 = self.get_updrs_total_longitudinal('3', dfSelectedSubjects=dfSelectedSubjects)
        dfUPDRS4 = self.get_updrs_total_longitudinal('4', dfSelectedSubjects=dfSelectedSubjects).fillna(0)
        dfUPDRS = pd.DataFrame()
        for strTime in ['base', '1y', '2y', '4y']:
            dfUPDRS['updrstot_' + strTime] = dfUPDRS1['updrs1tot_' + strTime] + dfUPDRS1Q['updrs1Qtot_' + strTime] \
                                             + dfUPDRS2['updrs2Qtot_' + strTime] + dfUPDRS3['updrs3tot_' + strTime] \
                                             + dfUPDRS4['updrs4tot_' + strTime]
        dfUPDRS.index = dfSelectedSubjects.index
        return dfUPDRS

    def get_moca_longitudinal(self, dfSelectedSubjects=ImageData().dfFunc):
        """
        Get a dataframe containing MoCA total scores at baseline, 1 year, 2 year, and 4 year visits.

        :param dfSelectedSubjects: optional, a dataframe containing 'sub' and 'visit' columns that specifies what
        subjects to use and which visit code to consider the baseline. By default, use ppmiutils.dataset.ImageData(
        ).dfFunc to select only subjects that have fMRI images and use the earliest scan date as baseline. If None,
        use all subjects in the UPDRS data and use the "Baseline" visit as the baseline.
        :type dfSelectedSubjects: pandas.DataFrame
        :return: dataframe containing baseline, 2 year, and 4 year UPDRS part III total scores and visit dates
        :rtype: pandas.DataFrame
        """
        dfMoCA = self.dfMoCA.copy()
        dfMoCA['sub'] = dfMoCA['PATNO']
        dfMoCA['date'] = pd.to_datetime(dfMoCA['INFODT'], format='%m/%Y')
        dfMoCA['visit'] = datatools.convert_visit_month(dfMoCA['EVENT_ID'])
        dfMoCA['moca'] = dfMoCA['MCATOT']

        return self._get_scores_longitudinal(dfMoCA, 'moca', dfSelectedSubjects)

    def get_pd_features(self, dfSelectedSubjects=ImageData().dfFunc):
        """
        Get PD diagnosis information, including days since diagnosis, days since symptom start, and various symptoms
        at presentation.
        tremor
        rigidity
        bradykinesia
        postural instability
        dominant side

        :param dfSelectedSubjects: optional, a dataframe containing 'sub' and 'date' columns that specifies what
        subjects to use and which date to measure "days since diagnosis" from. By default,
        use ppmiutils.dataset.ImageData().dfFunc to select only subjects that have fMRI images and use the
        earliest scan date as baseline. If None, use all subjects in the UPDRS data and use the "Baseline" visit as
        the baseline.
        :type dfSelectedSubjects: pandas.DataFrame
        :return:
        :rtype: pandas.DataFrame
        """
        dfPDFeat = self.dfPDFeat.copy()
        dfPDFeat['sub'] = dfPDFeat['PATNO']
        dfPDFeat['date'] = pd.to_datetime(dfPDFeat['INFODT'], format='%b-%y')
        dfPDFeat['visit'] = datatools.convert_visit_month(dfPDFeat['EVENT_ID'])
        lsSymptomDates = []
        for nIdx, row in dfPDFeat.iterrows():
            strMo = row['SXMO']
            try:
                nMo = int(strMo)
            except ValueError:
                # if month is listed, just use January
                nMo = 1
            strYr = row['SXYEAR']
            lsSymptomDates += [dt.datetime.strptime('{:02d}-{}'.format(int(nMo), strYr), '%m-%Y')]
        dfPDFeat['symptom_date'] = lsSymptomDates
        dfPDFeat['diag_date'] = pd.to_datetime(dfPDFeat['PDDXDT'], format='%b-%y')

        if dfSelectedSubjects is not None:
            dfSubDates = dfSelectedSubjects[['sub', 'date']]
            # # Use the visit date of the earliest scan
            # dfSubDates = dfSubDates.loc[~ dfSubDates['sub'].duplicated(keep='first')]
            dfSubDates['date'] = pd.to_datetime(dfSubDates['date'])
        else:
            dfSubDates = dfPDFeat[['sub', 'date']]
        dfSubDates.set_index('sub', inplace=True)

        dfPDFeat.set_index('sub', inplace=True)
        dfFeatures = pd.DataFrame(index=dfSubDates.index)
        dfPDFeat = dfPDFeat.loc[dfFeatures.index]
        dfFeatures['days_dx'] = (dfSubDates['date'] - dfPDFeat['diag_date'])
        dfFeatures['days_dx'] = dfFeatures['days_dx'].apply(lambda x: x.days)
        dfFeatures['days_sx'] = (dfSubDates['date'] - dfPDFeat['symptom_date'])
        dfFeatures['days_sx'] = dfFeatures['days_sx'].apply(lambda x: x.days)
        dfFeatures['tremor'] = dfPDFeat['DXTREMOR']
        dfFeatures['rigid'] = dfPDFeat['DXRIGID']
        dfFeatures['brady'] = dfPDFeat['DXBRADY']
        dfFeatures['posins'] = dfPDFeat['DXPOSINS']
        # dictSides = {1: 'left', 2: 'right', 3: 'symmetric'}
        # dfFeatures['domside'] = dfPDFeat['DOMSIDE'].apply(lambda x: dictSides[x] if not pd.isnull(x) else np.nan)
        dfFeatures['domside_left'] = dfPDFeat['DOMSIDE'].apply(lambda x: (x==1) if not pd.isnull(x) else np.nan)
        dfFeatures['domside_right'] = dfPDFeat['DOMSIDE'].apply(lambda x: (x == 2) if not pd.isnull(x) else np.nan)
        dfFeatures['domside_both'] = dfPDFeat['DOMSIDE'].apply(lambda x: (x == 3) if not pd.isnull(x) else np.nan)
        return dfFeatures

    def get_gds_data_longitudinal(self, dfSelectedSubjects=ImageData().dfFunc):
        """
        Get a dataframe containing GDS individual responses at baseline, 1 year, 2 year, and 4 year visits.

        :param dfSelectedSubjects: optional, a dataframe containing 'sub' and 'visit' columns that specifies what
        subjects to use and which visit code to consider the baseline. By default, use ppmiutils.dataset.ImageData(
        ).dfFunc to select only subjects that have fMRI images and use the earliest scan date as baseline. If None,
        use all subjects in the UPDRS data and use the "Baseline" visit as the baseline.
        :type dfSelectedSubjects: pandas.DataFrame
        :return: dataframe containing baseline, 2 year, and 4 year GDS scores and visit dates
        :rtype: pandas.DataFrame
        """

        dfGDS = self.dfGDS.copy()
        dfGDS['sub'] = dfGDS['PATNO']
        dfGDS['date'] = pd.to_datetime(dfGDS['INFODT'], format='%m/%Y')
        dfGDS['visit'] = datatools.convert_visit_month(dfGDS['EVENT_ID'])
        lsCols = dfGDS.columns[['GDS' in strCol for strCol in dfGDS.columns]].tolist()

        return self._get_scores_longitudinal(dfGDS, lsCols, dfSelectedSubjects)

    def get_gds_total_longitudinal(self, dfSelectedSubjects=ImageData().dfFunc):
        """
        Get a dataframe containing GDS total score at baseline, 1 year, 2 year, and 4 year visits.

        :param dfSelectedSubjects: optional, a dataframe containing 'sub' and 'visit' columns that specifies what
        subjects to use and which visit code to consider the baseline. By default, use ppmiutils.dataset.ImageData(
        ).dfFunc to select only subjects that have fMRI images and use the earliest scan date as baseline. If None,
        use all subjects in the UPDRS data and use the "Baseline" visit as the baseline.
        :type dfSelectedSubjects: pandas.DataFrame
        :return: dataframe containing baseline, 2 year, and 4 year GDS scores and visit dates
        :rtype: pandas.DataFrame
        """

        dfGDS = self.dfGDS.copy()
        dfGDS['sub'] = dfGDS['PATNO']
        dfGDS['date'] = pd.to_datetime(dfGDS['INFODT'], format='%m/%Y')
        dfGDS['visit'] = datatools.convert_visit_month(dfGDS['EVENT_ID'])
        lsCols = dfGDS.columns[['GDS' in strCol for strCol in dfGDS.columns]].tolist()
        dfGDS['gdstotal'] = dfGDS[lsCols].sum(axis=1)

        return self._get_scores_longitudinal(dfGDS, 'gdstotal', dfSelectedSubjects)



    def _get_scores_longitudinal(self, dfData, col, dfSelectedSubjects):
        """
        Get baseline, 2 year, and 4 year scores as a dataframe

        :param dfData: dataframe of clinical data
        :type dfData: pandas.DataFrame
        :param col: column or list of columns containing the desired score(s)
        :type col: str, list
        :param dfSelectedSubjects: dataframe containing the desired subjects and corresponding "baseline" visit codes
        :type dfSelectedSubjects: pandas.DataFrame
        :return: dataframe containing baseline, 1 year, 2 year, and 4 year scores and visit dates
        :rtype: pandas.DataFrame
        """
        if dfSelectedSubjects is not None:
            dfSubVisits = dfSelectedSubjects[['sub', 'visit']]
            # # Use the visit date of the earliest scan
            # dfSubVisits = dfSubVisits.loc[~ dfSubVisits['sub'].duplicated(keep='first')]
        else:
            dfSubVisits = dfData[['sub', 'visit']].loc[dfData['visit'] == 'Baseline']
        dfSubVisits.set_index('sub', inplace=True)
        if not isinstance(col, list):
            lsColIn = [col]
        else:
            lsColIn = col

        lsColOut = []
        for strCol in lsColIn:
            lsColOut += [strCol + '_base',
                         strCol + '_1y',
                         strCol + '_2y',
                         strCol + '_4y']
        lsColOut += ['date_base', 'date_1y', 'date_2y', 'date_4y']

        dfScores = pd.DataFrame(index=dfSubVisits.index,
                                columns=lsColOut)
        for i, (nSub, dfVisit) in enumerate(dfSubVisits.iterrows()):
            strVisit = dfVisit['visit']
            # strVisit = dfSubVisits['visit'].loc[dfSubVisits['sub'] == nSub][0]
            row = dfData.loc[(dfData['PATNO'] == int(nSub)) & (dfData['visit'] == strVisit)]
            dictScores = {}
            if row.size == 0:
                continue
            # If there are multiple entries for this visit, take the last one
            for strCol in lsColIn:
                dictScores[strCol + '_base'] = row[strCol].iloc[-1]
            dictScores['date_base'] = row['date'].iloc[-1]

            # Find the score in 1 year (+/- a month)
            row1 = dfData.loc[(dfData['PATNO'] == int(nSub))
                               & ((dfData['date'] >= (row['date'] + dt.timedelta(days=365 - 60)).values[0])
                                  & (dfData['date'] <= (row['date'] + dt.timedelta(days=365 + 60)).values[0]))]

            if row1.size > 0:
                for strCol in lsColIn:
                    dictScores[strCol + '_1y'] = row1[strCol].iloc[-1]
                dictScores['date_1y'] = row1['date'].iloc[-1]

            # Find the score in 2 years (+/- a month)
            row2 = dfData.loc[(dfData['PATNO'] == int(nSub))
                               & ((dfData['date'] >= (row['date'] + dt.timedelta(days=2 * 365 - 60)).values[0])
                                  & (dfData['date'] <= (row['date'] + dt.timedelta(days=2 * 365 + 60)).values[0]))]

            if row2.size > 0:
                for strCol in lsColIn:
                    dictScores[strCol + '_2y'] = row2[strCol].iloc[-1]
                dictScores['date_2y'] = row2['date'].iloc[-1]

            # Find the score in 4 years (+/- a month)
            row4 = dfData.loc[(dfData['PATNO'] == int(nSub))
                               & ((dfData['date'] >= (row['date'] + dt.timedelta(days=4 * 365 - 60)).values[0])
                                  & (dfData['date'] <= (row['date'] + dt.timedelta(days=4 * 365 + 60)).values[0]))]

            if row4.size > 0:
                for strCol in lsColIn:
                    dictScores[strCol + '_4y'] = row4[strCol].iloc[-1]
                dictScores['date_4y'] = row4['date'].iloc[-1]

            dfScores.iloc[i] = dictScores
        return dfScores

    def get_subject_group(self, dfSelectedSubjects=ImageData().dfFunc):
        """
        Get subject group at recruitment, enrollment, and imaging, and genetic risk factors if any

        :param dfSelectedSubjects: optional, a dataframe containing 'sub' and 'date' columns that specifies what
        subjects to use and which date to measure "days since diagnosis" from. By default,
        use ppmiutils.dataset.ImageData().dfFunc to select only subjects that have fMRI images and use the
        earliest scan date as baseline. If None, use all subjects in the UPDRS data and use the "Baseline" visit as
        the baseline.
        :type dfSelectedSubjects: pandas.DataFrame
        :return:
        :rtype: pandas.DataFrame
        """
        dfGroup = self.dfGroup.copy()
        dfGroup['sub'] = dfGroup.index
        if dfSelectedSubjects is not None:
            dfSubs = dfSelectedSubjects['sub'].drop_duplicates()
        else:
            dfSubs = dfGroup['sub']

        dfGroupOut = pd.DataFrame(index=dfSubs)
        dfGroupOut['group_recruitment'] = dfGroup['RECRUITMENT_CAT'].loc[dfSubs]
        dfGroupOut['group_imaging'] = dfGroup['IMAGING_CAT'].loc[dfSubs]
        dfGroupOut['group_enrollment'] = dfGroup['ENROLL_CAT'].loc[dfSubs]
        dfGroupOut['group_desc'] = dfGroup['DESCRP_CAT'].loc[dfSubs]
        return dfGroupOut

    def get_demographics(self, dfSelectedSubjects=ImageData().dfFunc):
        """
        Get subject demographics: race, age, sex, handedness, and years of education

        :param dfSelectedSubjects: optional, a dataframe containing 'sub' and 'date' columns that specifies what
        subjects to use and which date to measure "days since diagnosis" from. By default,
        use ppmiutils.dataset.ImageData().dfFunc to select only subjects that have fMRI images and use the
        earliest scan date as baseline. If None, use all subjects in the UPDRS data and use the "Baseline" visit as
        the baseline.
        :type dfSelectedSubjects: pandas.DataFrame
        :return:
        :rtype: pandas.DataFrame
        """

        dfDemo = self.dfDemo.copy()
        dfSocEc = self.dfSocEc.copy()
        dfDemo.set_index('PATNO', inplace=True)
        dfSocEc.set_index('PATNO', inplace=True)

        if dfSelectedSubjects is not None:
            dfSubDates = dfSelectedSubjects[['sub', 'date']]
            # Use the visit date of the earliest scan
            dfSubDates = dfSubDates.loc[~ dfSubDates['sub'].duplicated(keep='first')]
            dfSubDates['date'] = pd.to_datetime(dfSubDates['date'])
        else:
            dfSubDates = pd.DataFrame()
            dfSubDates['sub'] = dfDemo['PATNO']
            dfSubDates['date'] = dfDemo['PRJENRDT']
        dfSubDates.set_index('sub', inplace=True)
        dfDemo = dfDemo.loc[dfSubDates.index].drop_duplicates()
        dfSocEc = dfSocEc.loc[dfSubDates.index].drop_duplicates()

        dfDemoOut = pd.DataFrame(index=dfSubDates.index)
        dfDates = pd.to_datetime(dfSubDates['date'])
        dfBirthyears = pd.to_datetime(dfDemo['BIRTHDT'].astype(int), format='%Y')
        dfDemoOut['age'] = [relativedelta.relativedelta(dfDates.loc[sub], dfBirthyears.loc[sub]).years for sub in
                            dfDates.index]
        dfDemoOut['sex_male'] = dfDemo['GENDER'].apply(lambda x: x == 2)
        dfDemoOut['hispanic'] = dfDemo['HISPLAT']
        dfDemoOut['race_amindalnat'] = dfDemo['RAINDALS']
        dfDemoOut['race_asian'] = dfDemo['RAASIAN']
        dfDemoOut['race_black'] = dfDemo['RABLACK']
        dfDemoOut['race_pacifisl'] = dfDemo['RAHAWOPI']
        dfDemoOut['race_white'] = dfDemo['RAWHITE']
        dfDemoOut['race_other'] = dfDemo['RANOS']
        dfDemoOut['eduyears'] = dfSocEc['EDUCYRS'].astype(int)
        dfDemoOut['handed_right'] = dfSocEc['HANDED'].apply(lambda x: x == 1)
        dfDemoOut['handed_left'] = dfSocEc['HANDED'].apply(lambda x: x == 2)
        dfDemoOut['handed_both'] = dfSocEc['HANDED'].apply(lambda x: x == 3)

        return dfDemoOut