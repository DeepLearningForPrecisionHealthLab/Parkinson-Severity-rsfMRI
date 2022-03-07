#!/usr/bin/env python
"""

Copyright (c) 2021 The University of Texas Southwestern Medical Center. See LICENSE.md for details.
"""
__author__ = 'Kevin P Nguyen'
__email__ = 'kevin3.nguyen@utsouthwestern.edu'

import glob
import os

def generate_fmri_list(strImgDir, strOutPrefix='./image_paths'):
    """
    Produce text files containing paths to the anatomical and functional images of all the subjects that have them.

    :param strImgDir: path to BIDS directory where images are located
    :type strImgDir: str
    :param strOutPrefix: prefix for saving the image lists as txt files
    :type strOutPrefix: str
    :return: None
    :rtype:
    """
    lsSubDirs = glob.glob(strImgDir + os.sep + 'sub-*')
    lsAnatFiles = []
    lsFuncFiles = []

    for strSubDir in lsSubDirs:
        lsSesDirs = glob.glob(strSubDir + os.sep + 'ses-*')
        for strSesDir in lsSesDirs:
            if os.path.exists(strSesDir + os.sep + 'anat') and os.path.exists(strSesDir + os.sep + 'func'):
                lsAnatFiles += [glob.glob(strSesDir + os.sep + 'anat/*.nii*')]
                lsFuncFiles += [glob.glob(strSesDir + os.sep + 'func/*.nii*')]

    with open(strOutPrefix + '_anat.txt', 'w') as file:
        file.writelines(lsAnatFiles)
    with open(strOutPrefix + '_func.txt', 'w') as file:
        file.writelines(lsFuncFiles)
