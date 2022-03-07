#!/usr/bin/env python
"""
Computation of rs-fMRI derivatives, such as dynamic ALFF, fALFF, and ReHo

Copyright (c) 2021 The University of Texas Southwestern Medical Center. See LICENSE.md for details.
"""
__author__ = 'Kevin P Nguyen'
__email__ = 'kevin3.nguyen@utsouthwestern.edu'


import glob, os
from nilearn import image as nilimage
from nipype.pipeline.engine import Workflow, Node, JoinNode
from nipype import IdentityInterface, Rename, DataSink
from nipype.interfaces.fsl import ExtractROI, Merge
from CPAC.alff import alff
from CPAC.reho import reho


def dynamic_measures(strFunc, strMask, strDir,
                     nWindowSizeTR=50,
                     nStrideTR=2,
                     fFilterHP=0.01, fFilterLP=0.1, nClusterSize=27, nJobs=1):
    '''
    Compute dynamic (sliding-window) ALFF, fALFF, and ReHo for an fMRI image.

    :param strFunc: path to preprocessed (normalized, denoised, etc.) fMRI NIfTI file
    :type strFunc: str
    :param strMask: path to binary brain mask NIfTI file
    :type strMask: str
    :param strDir: path to output directory
    :type strDir: str
    :param nWindowSizeTR: window length in TRs. Typically this is set to 1 / (frequency of slowest signals),
        e.g. 1 / 0.01 Hz = 100 s = 50 TRs (if TR is 2 s)
    :type nWindowSizeTR: int
    :param nStrideTR: stride in TRs
    :type nStrideTR: int
    :param fFilterHP: cutoff in Hz for highpass filter. Default 0.01 Hz
    :type fFilterHP: float
    :param fFilterLP: cutoff in Hz for lowpass filter. Default 0.1 Hz
    :type fFilterLP: float
    :param nClusterSize: cluster size (neighborhood) for ReHo computation. Can be 7, 19, or 27. Default 27
    :type nClusterSize: int
    :param nJobs: number of parallel processes
    :type nJobs: int
    '''

    strDir = os.path.realpath(strDir)

    if nClusterSize not in [7, 19, 27]:
        raise ValueError('{} is not a valid cluster size. Must be 7, 19, or 27'.format(nClusterSize))

    workflow = Workflow('dynamic', base_dir=os.path.join(strDir, 'working_dir'))

    # Get the length of the timeseries and the TR
    imgFunc = nilimage.load_img(strFunc)
    nTotalTRs = imgFunc.shape[-1]
    tr = imgFunc.header.get_zooms()[-1]
    # Define the starting timepoint of each window
    lsWindowStarts = list(range(0, nTotalTRs-nWindowSizeTR+1, nStrideTR))

    # This node iterates the ALFF and ReHo subworkflows over all the windows
    windowNode = Node(IdentityInterface(fields=['window']), name='windownode')
    windowNode.iterables = ('window', lsWindowStarts)

    # Use fslroi to extract the window from the full timeseries into a new image
    windowExtract = Node(ExtractROI(t_size=nWindowSizeTR), name='extractwindow')
    windowExtract.inputs.in_file = strFunc
    workflow.connect(windowNode, 'window', windowExtract, 't_min')

    # Compute ALFF and fALFF with CPAC's workflow
    alffWf = alff.create_alff()
    alffWf.inputs.hp_input.hp = fFilterHP
    alffWf.inputs.lp_input.lp = fFilterLP
    alffWf.inputs.inputspec.rest_mask = strMask
    workflow.connect(windowExtract, 'roi_file', alffWf, 'inputspec.rest_res')

    # Compute ReHo with CPAC's workflow
    rehoWf = reho.create_reho()
    rehoWf.inputs.inputspec.cluster_size = nClusterSize
    rehoWf.inputs.inputspec.rest_mask = strMask
    workflow.connect(windowExtract, 'roi_file', rehoWf, 'inputspec.rest_res_filt')

    # Use fslmerge to join the results for each window into a single timeseries image
    joinAlff = JoinNode(Merge(dimension='t', tr=tr), name='merge_alff', joinsource='windownode',
                        joinfield='in_files')
    workflow.connect(alffWf, 'outputspec.alff_img', joinAlff, 'in_files')

    joinFalff = JoinNode(Merge(dimension='t', tr=tr), name='merge_falff', joinsource='windownode',
                        joinfield='in_files')
    workflow.connect(alffWf, 'outputspec.falff_img', joinFalff, 'in_files')

    joinReho = JoinNode(Merge(dimension='t', tr=tr), name='merge_reho', joinsource='windownode',
                         joinfield='in_files')
    workflow.connect(rehoWf, 'outputspec.raw_reho_map', joinReho, 'in_files')

    # Copy outputs into a user-friendly location
    outputNode = Node(DataSink(base_directory=strDir, remove_dest_dir=True), name='datasink')
    workflow.connect(joinAlff, 'merged_file', outputNode, 'results.@alff')
    workflow.connect(joinFalff, 'merged_file', outputNode, 'results.@falff')
    workflow.connect(joinReho, 'merged_file', outputNode, 'results.@reho')

    if nJobs > 1:
        # Run with multiprocess parallelization
        workflow.run(plugin='MultiProc', plugin_args={'n_procs': nJobs})
    else:
        workflow.run()


def static_measures(strFunc, strMask, strDir,
                    fFilterHP=0.01, fFilterLP=0.1, nClusterSize=27, nJobs=1):
    '''
    Compute static ALFF, fALFF, and ReHo for an fMRI image.

    :param strFunc: path to preprocessed (normalized, denoised, etc.) fMRI NIfTI file
    :type strFunc: str
    :param strMask: path to binary brain mask NIfTI file
    :type strMask: str
    :param strDir: path to output directory
    :type strDir: str
    :param fFilterHP: cutoff in Hz for highpass filter. Default 0.01 Hz
    :type fFilterHP: float
    :param fFilterLP: cutoff in Hz for lowpass filter. Default 0.1 Hz
    :type fFilterLP: float
    :param nClusterSize: cluster size (neighborhood) for ReHo computation. Can be 7, 19, or 27. Default 27
    :type nClusterSize: int
    :param nJobs: number of parallel processes
    :type nJobs: int
    '''

    strDir = os.path.realpath(strDir)

    if nClusterSize not in [7, 19, 27]:
        raise ValueError('{} is not a valid cluster size. Must be 7, 19, or 27'.format(nClusterSize))

    workflow = Workflow('static', base_dir=os.path.join(strDir, 'working_dir'))

    # Compute ALFF and fALFF with CPAC's workflow
    alffWf = alff.create_alff()
    alffWf.inputs.hp_input.hp = fFilterHP
    alffWf.inputs.lp_input.lp = fFilterLP
    alffWf.inputs.inputspec.rest_mask = strMask
    alffWf.inputs.inputspec.rest_res = strFunc

    # Compute ReHo with CPAC's workflow
    rehoWf = reho.create_reho()
    rehoWf.inputs.inputspec.cluster_size = nClusterSize
    rehoWf.inputs.inputspec.rest_mask = strMask
    rehoWf.inputs.inputspec.rest_res_filt = strFunc

    # Copy outputs into a user-friendly location
    outputNode = Node(DataSink(base_directory=strDir, remove_dest_dir=True), name='datasink')
    workflow.connect(alffWf, 'outputspec.alff_img', outputNode, 'results.@alff')
    workflow.connect(alffWf, 'outputspec.falff_img', outputNode, 'results.@falff')
    workflow.connect(rehoWf, 'outputspec.raw_reho_map', outputNode, 'results.@reho')

    if nJobs > 1:
        # Run with multiprocess parallelization
        workflow.run(plugin='MultiProc', plugin_args={'n_procs': nJobs})
    else:
        workflow.run()