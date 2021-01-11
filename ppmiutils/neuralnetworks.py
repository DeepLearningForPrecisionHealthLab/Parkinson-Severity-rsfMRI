#!/usr/bin/env python
"""
Module for generating random NN .ini files and parsing them to build models
______
10/11/18
"""
__author__ = 'Kevin P Nguyen'
__email__ = 'kevin3.nguyen@utsouthwestern.edu'

import os
import sys
import glob
import json
import pandas as pd
import configparser
import numpy as np
import matplotlib.pyplot as plt
from keras import callbacks


def rmse(yTrue, yPred):
    """
    Custom loss function that computes RMSE
    :param yTrue:
    :param yPred:
    :return: RMSE
    """
    import keras.backend as K
    return K.sqrt(K.mean(K.pow(yTrue - yPred, 2)))


def rsquare(yTrue, yPred):
    """
    Returns the coefficient of determination R^2
    :param yTrue:
    :param yPred:
    :return:
    """
    import keras.backend as K
    ssRes = K.sum(K.square(yTrue - yPred))
    ssTotal = K.sum(K.square(yTrue - K.mean(yTrue)))
    return (1 - ssRes / (ssTotal + K.epsilon()))


def rsquare_loss(yTrue, yPred):
    """
        Returns the 1 minus the coefficient of determination R^2
        :param yTrue:
        :param yPred:
        :return:
        """
    import keras.backend as K
    ssRes = K.sum(K.square(yTrue - yPred))
    ssTotal = K.sum(K.square(yTrue - K.mean(yTrue)))
    return ssRes / (ssTotal + K.epsilon())


class RSquareCallback(callbacks.Callback):
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.rsquare = {'train_rsquare': [], 'val_rsquare': []}

    def on_epoch_end(self, epoch, logs=None):
        y_train_pred = self.model.predict(self.x_train).flatten()
        y_test_pred = self.model.predict(self.x_test).flatten()
        rsquare_train = self._rsquare(self.y_train, y_train_pred)
        rsquare_test = self._rsquare(self.y_test, y_test_pred)
        self.rsquare['train_rsquare'] += [rsquare_train]
        self.rsquare['val_rsquare'] += [rsquare_test]

    def _rsquare(self, yTrue, yPred):
        ssRes = np.sum(np.square(yTrue - yPred))
        ssTotal = np.sum(np.square(yTrue - np.mean(yTrue)))
        return (1 - ssRes / (ssTotal + 1e-10))

def make_ini_dnn_regressor(nModels, strPath, dictParamRange):
    """
        Generates a number of .ini files specifying random feed-forward DNN architectures
        :param nModels: number of files to generate
        :param strPath: directory where files will be saved
        :param dictParamRange: dictionary where keys are hyperparameter names and values are lists from which to
        randomly select. Any hyperparameters not specified will use the default value lists.
            Available hyperparameters include:
                denselayers: number of dense layers (default [1, 2, 3, 4, 5])
                firstlayerneurons: number of neurons in the first dense layer (default [128, 256, 512, 768, 1024])
                kernel_regularizer: weight regularizer for dense layer weights (default ['l1', 'l2', 'l1_l2'])
                activity_regularizer: activity regularizer for dense layer weights (default [None])
                taper: percentage decrease in layer size (number of neurons) at each dense layer (default [0, 0.5])
                dropout: dropout ratio to add after a layer (default [0, 0.5])
                activation: type of activation after each layer (default ['ReLU'])
                batchnorm: add batch normalization if true (defualt [False])
                optimizer: Keras optimizer (default ['Nadam'])
                lr: learning rate for optimizer (default np.arange(0.0001, 0.005, 0.0001).tolist())
                decay: learning rate decay (default [0])
        :return: None
        """
    dictParamRangeDefault = {'denselayers': [1, 2, 3, 4, 5],
                             'firstlayerneurons': [128, 256, 512, 768, 1024],
                             'kernel_l1': [0, 0.01, 0.05],
                             'kernel_l2': [0, 0.01, 0.05],
                             'activity_l1': [0, 0.01, 0.05],
                             'activity_l2': [0, 0.01, 0.05],
                             'taper': [0, 0.5],
                             'batchnorm': [False],
                             'dropout': [0, 0.5],
                             'activation': ['ReLU'],
                             'optimizer': ['Nadam'],
                             'lr': np.arange(0.0001, 0.005, 0.0001).tolist(),
                             'schedule_decay': [0]}
    if dictParamRange is None:
        dictParamRange = dictParamRangeDefault
    else:
        for strKey in dictParamRangeDefault.keys():
            if strKey not in dictParamRange.keys():
                dictParamRange[strKey] = dictParamRangeDefault[strKey]

    for m in np.arange(nModels):
        config = configparser.ConfigParser()
        # Select model optimizer hyperparams
        config['optimizer'] = {'optimizer': np.random.choice(dictParamRange['optimizer']),
                               'lr': np.random.choice(dictParamRange['lr']),
                               'schedule_decay': np.random.choice(dictParamRange['schedule_decay'])}

        # Create input
        config['Input_main'] = {'name': 'Input_main'}

        # Use the same activation for all layers
        strActivation = np.random.choice(dictParamRange['activation'])

        strLayerPrev = 'Input_main'

        # Select dense layer hyperparams
        nDenseLayers = np.random.choice(dictParamRange['denselayers'])
        for n in np.arange(nDenseLayers):

            strLayer = 'Dense_' + str(n)
            if n == 0:
                nNeurons = np.random.choice(dictParamRange['firstlayerneurons'])
            else:
                # Chance to taper the number of neurons
                nNeurons = int((1 - np.random.choice(dictParamRange['taper'])) * nNeurons)
                # But don't let the layer get smaller than 4 neurons
                nNeurons = np.max((nNeurons, 4))
            config[strLayer + '/' + strLayerPrev] = {'units': nNeurons,
                                                     'kernel_l1': np.random.choice(dictParamRange['kernel_l1']),
                                                     'kernel_l2': np.random.choice(dictParamRange['kernel_l2']),
                                                     'activity_l1': np.random.choice(dictParamRange['activity_l1']),
                                                     'activity_l2': np.random.choice(dictParamRange['activity_l2']),
                                                     'name': strLayer}
            strLayerPrev = strLayer

            # Chance to add batch norm
            if np.random.choice(dictParamRange['batchnorm']):
                strLayer = 'BatchNormalization_' + str(n)
                config[strLayer + '/' + strLayerPrev] = {'name': strLayer}
                strLayerPrev = strLayer

            # Add activation
            strLayer = strActivation + '_Dense' + str(n)
            config[strLayer + '/' + strLayerPrev] = {'name': strLayer}
            strLayerPrev = strLayer

            # Chance to add dropout
            fDropout = np.random.choice(dictParamRange['dropout'])
            if fDropout > 0:
                strLayer = 'Dropout_' + str(n)
                config[strLayer + '/' + strLayerPrev] = {'rate': fDropout,
                                                         'name': strLayer}
                strLayerPrev = strLayer

        # Create the output layer
        strLayer = 'Dense_output'
        config[strLayer + '/' + strLayerPrev] = {'units': 1,
                                                 'name': strLayer}

        os.makedirs(strPath + '/model{:03d}'.format(m), exist_ok=False)

        with open(strPath + '/model{:03d}/model{:03d}.ini'.format(m, m), 'w') as file:
            config.write(file)

def make_ini_dnn_classifier(nModels, strPath, dictParamRange):
    """
        Generates a number of .ini files specifying random feed-forward DNN architectures
        :param nModels: number of files to generate
        :param strPath: directory where files will be saved
        :param dictParamRange: dictionary where keys are hyperparameter names and values are lists from which to
        randomly select. Any hyperparameters not specified will use the default value lists.
            Available hyperparameters include:
                denselayers: number of dense layers (default [1, 2, 3, 4, 5])
                firstlayerneurons: number of neurons in the first dense layer (default [128, 256, 512, 768, 1024])
                kernel_regularizer: weight regularizer for dense layer weights (default ['l1', 'l2', 'l1_l2'])
                activity_regularizer: activity regularizer for dense layer weights (default [None])
                taper: percentage decrease in layer size (number of neurons) at each dense layer (default [0, 0.5])
                dropout: dropout ratio to add after a layer (default [0, 0.5])
                activation: type of activation after each layer (default ['ReLU'])
                batchnorm: add batch normalization if true (defualt [False])
                optimizer: Keras optimizer (default ['Nadam'])
                lr: learning rate for optimizer (default np.arange(0.0001, 0.005, 0.0001).tolist())
                decay: learning rate decay (default [0])
        :return: None
        """
    dictParamRangeDefault = {'denselayers': [1, 2, 3, 4, 5],
                             'firstlayerneurons': [128, 256, 512, 768, 1024],
                             'kernel_l1': [0, 0.01, 0.05],
                             'kernel_l2': [0, 0.01, 0.05],
                             'activity_l1': [0, 0.01, 0.05],
                             'activity_l2': [0, 0.01, 0.05],
                             'taper': [0, 0.5],
                             'batchnorm': [False],
                             'dropout': [0, 0.5],
                             'activation': ['ReLU'],
                             'optimizer': ['Nadam'],
                             'lr': np.arange(0.0001, 0.005, 0.0001).tolist(),
                             'schedule_decay': [0]}
    if dictParamRange is None:
        dictParamRange = dictParamRangeDefault
    else:
        for strKey in dictParamRangeDefault.keys():
            if strKey not in dictParamRange.keys():
                dictParamRange[strKey] = dictParamRangeDefault[strKey]

    for m in np.arange(nModels):
        config = configparser.ConfigParser()
        # Select model optimizer hyperparams
        config['optimizer'] = {'optimizer': np.random.choice(dictParamRange['optimizer']),
                               'lr': np.random.choice(dictParamRange['lr']),
                               'schedule_decay': np.random.choice(dictParamRange['schedule_decay'])}

        # Create input
        config['Input_main'] = {'name': 'Input_main'}

        # Use the same activation for all layers
        strActivation = np.random.choice(dictParamRange['activation'])

        strLayerPrev = 'Input_main'

        # Select dense layer hyperparams
        nDenseLayers = np.random.choice(dictParamRange['denselayers'])
        for n in np.arange(nDenseLayers):

            strLayer = 'Dense_' + str(n)
            if n == 0:
                nNeurons = np.random.choice(dictParamRange['firstlayerneurons'])
            else:
                # Chance to taper the number of neurons
                nNeurons = int((1 - np.random.choice(dictParamRange['taper'])) * nNeurons)
                # But don't let the layer get smaller than 4 neurons
                nNeurons = np.max((nNeurons, 4))
            config[strLayer + '/' + strLayerPrev] = {'units': nNeurons,
                                                     'kernel_l1': np.random.choice(dictParamRange['kernel_l1']),
                                                     'kernel_l2': np.random.choice(dictParamRange['kernel_l2']),
                                                     'activity_l1': np.random.choice(dictParamRange['activity_l1']),
                                                     'activity_l2': np.random.choice(dictParamRange['activity_l2']),
                                                     'name': strLayer}
            strLayerPrev = strLayer

            # Chance to add batch norm
            if np.random.choice(dictParamRange['batchnorm']):
                strLayer = 'BatchNormalization_' + str(n)
                config[strLayer + '/' + strLayerPrev] = {'name': strLayer}
                strLayerPrev = strLayer

            # Add activation
            strLayer = strActivation + '_Dense' + str(n)
            config[strLayer + '/' + strLayerPrev] = {'name': strLayer}
            strLayerPrev = strLayer

            # Chance to add dropout
            fDropout = np.random.choice(dictParamRange['dropout'])
            if fDropout > 0:
                strLayer = 'Dropout_' + str(n)
                config[strLayer + '/' + strLayerPrev] = {'rate': fDropout,
                                                         'name': strLayer}
                strLayerPrev = strLayer

        # Create the output layer
        strLayer = 'Dense_output'
        config[strLayer + '/' + strLayerPrev] = {'units': 1,
                                                 'name': strLayer,
                                                 'activation': 'sigmoid'}

        os.makedirs(strPath + '/model{:03d}'.format(m), exist_ok=False)

        with open(strPath + '/model{:03d}/model{:03d}.ini'.format(m, m), 'w') as file:
            config.write(file)

def _config_to_dict(config):
    # Convert numerical values in a ConfigParser object into the appropriate floats or ints
    dictArgs = {}
    for strKey in config.keys():
        strArg = config[strKey]
        if strArg.isnumeric():
            dictArgs[strKey] = int(strArg)
        elif strArg == 'none':
            dictArgs[strKey] = None
        else:
            try:
                dictArgs[strKey] = float(strArg)
            except ValueError:
                dictArgs[strKey] = config[strKey]
    return dictArgs


def generate_model(strModelIni, dictInputShapes, loss='mse', metrics=None, loss_weights=None):
    """
    Parse a neural network .ini file to generate a Keras model.
    :param strModelIni: path to the .ini file
    :param dictInputShapes: a dictionary containing the shapes of each model input. Keys should match the input
    labeling used in the .ini file. E.g. if the .ini file contains 'Input_image' and 'Input_treatment' layers,
    then this dictionary should contain values {'image': tupImageShape, 'treatment': tupTreatmentShape}
    :param loss: a loss function or a string name of one of the built-in Keras loss functions
    :param metrics: a list of metrics
    :return: compiled Keras model
    """
    from keras.models import Model
    from keras import layers
    from keras import optimizers

    config = configparser.ConfigParser()
    config.read(strModelIni)

    lsLayers = list(config.keys())
    lsLayers.remove('optimizer')
    if 'custom' in lsLayers:
        lsLayers.remove('custom')
    lsLayers.remove('DEFAULT')
    dictTensors = {}  # Store the output tensors of each layer
    lsInputLayers = []  # Store the name of the input layers
    for strLayerDesc in lsLayers:
        strLayer = strLayerDesc.split('/')[0]
        strLayerInputs = strLayerDesc.split('/')[1:]
        strLayerType = strLayer.split('_')[0]
        if strLayerType == 'Input':
            strInputLabel = strLayer.split('_')[1]
            dictTensors[strLayer] = layers.Input(shape=dictInputShapes[strInputLabel],
                                                 name=config[strLayerDesc]['name'])
            lsInputLayers += [strLayer]
        else:
            configArgs = config[strLayerDesc]
            dictArgs = _config_to_dict(configArgs)
            if len(strLayerInputs) == 1:
                inputTensor = dictTensors[strLayerInputs[0]]
            else:
                inputTensor = [dictTensors[strInput] for strInput in strLayerInputs]
            if 'kernel_l1' not in dictArgs.keys():
                # legacy .ini file that doesn't define regularizer coefficients
                dictTensors[strLayer] = getattr(layers, strLayerType)(**dictArgs)(inputTensor)
            else:
                from keras import regularizers
                fKernL1 = dictArgs.pop('kernel_l1')
                fKernL2 = dictArgs.pop('kernel_l2')
                fActL1 = dictArgs.pop('activity_l1')
                fActL2 = dictArgs.pop('activity_l2')
                dictTensors[strLayer] = getattr(layers, strLayerType)(kernel_regularizer=regularizers.l1_l2(fKernL1,
                                                                                                            fKernL2),
                                                                      activity_regularizer=regularizers.l1_l2(fActL1,
                                                                                                              fActL2),
                                                                      **dictArgs)(inputTensor)


    model = Model([dictTensors[layer] for layer in lsInputLayers], dictTensors[strLayer])
    dictOptArgs = _config_to_dict(config['optimizer'])
    fLR = dictOptArgs.pop('lr')
    strOpt = dictOptArgs.pop('optimizer')
    if fLR == 0:
        opt = getattr(optimizers, strOpt)(**dictOptArgs)
    else:
        opt = getattr(optimizers, strOpt)(lr=fLR, **dictOptArgs)
    model.compile(opt, loss=loss, metrics=metrics, loss_weights=loss_weights)
    return model


class RandomSearchResults:
    def __init__(self, strModelsDir, strLossFile):
        """
        Class for reading in the training results of a random model architecture search
        :param strModelsDir: path to directory with all the models
        """
        self.dfLoss = pd.DataFrame(columns=['val_loss', 'loss']) # dataframe containing the final loss for each
        # model
        self.lsModelPaths = [] # list of model directories
        self.lsModels = []  # list of ModelResults objects

        listModels = glob.glob(strModelsDir + os.sep + 'model*')
        listModels.sort()

        for strModel in listModels:
            self.lsModelPaths += [strModel]
            results = ModelResults(strModel + os.sep + strLossFile)
            self.lsModels += [results]

            self.dfLoss.loc[strModel.split(os.sep)[-1]] = results.dfLoss.loc['mean']

    def plot_nlayers(self):
        """
        Plot loss vs. number of layers
        """
        nModels = len(self.lsModels)
        listNLayers = []
        for n in range(nModels):
            nLayers = np.sum(['dense' in section for section in self.lsModels[n].get_config.sections()])
            nLayers += np.sum(['conv' in section for section in self.lsModels[n].get_config.sections()])
            listNLayers += [nLayers]

        fig, ax = plt.subplots(1, 1)
        ax.scatter(listNLayers, self.dfLoss['loss'], label='loss')
        ax.scatter(listNLayers, self.dfLoss['val_loss'], label='val_loss')
        ax.legend()
        ax.set_xlabel('Layers')
        ax.set_ylabel('Loss')
        fig.show()


class ModelResults:
    def __init__(self, strLossFile):
        """
        Class for reading and storing the training history of one model
        :param strLossFile: path to the pickle file containing the training history of the model,
        e.g. /path/to/model/loss_pla_craddock100_highpass.pkl
        """
        self.strModelPath = '/'.join(strLossFile.split(os.sep)[:-1])
        self.strModelName = strLossFile.split(os.sep)[-2]
        self.dfLoss = pd.DataFrame(columns=['loss', 'val_loss'])
        try:
            self.lsHistory = pd.read_pickle(strLossFile)
            for nFold, dfHistory in enumerate(self.lsHistory):
                self.dfLoss.loc[nFold] = [dfHistory['loss'].iloc[-1],
                                          dfHistory['val_loss'].iloc[-1]]
            dfMean = self.dfLoss.mean()
            dfStd = self.dfLoss.std()
            self.dfLoss.loc['mean'] = dfMean
            self.dfLoss.loc['std'] = dfStd
        except FileNotFoundError:
            print('Results not found for {}'.format(self.strModelName))
            self.dfLoss.loc['mean'] = np.nan
            self.dfLoss.loc['std'] = np.nan

    def get_config(self):
        """
        Get the hyperparameter configuration from this model's .ini file
        :return: a ConfigParser object
        """
        config = configparser.ConfigParser()
        config.read(self.strModelPath + '/' + self.strModelName + '.ini')
        return config

    def plot_history(self, strTitle=None, strSavePath=None):
        """
        Plot training and validation curve for this model
        :param strTitle: optional, title for the plot
        :param strSavePath: optional, path to save the figure
        :return: figure handle
        """

        fig, ax = plt.subplots(1, 1)
        lsHistory = self.lsHistory
        lsBlues = np.linspace(0.3, 0.9, len(lsHistory))
        lsOranges = np.linspace(0.9, 0.4, len(lsHistory))

        for nFold, dfHistory in enumerate(lsHistory):
            dfRMSE = dfHistory
            dfRMSE['rmse'] = np.sqrt(dfHistory['loss'])
            dfRMSE['val_rmse'] = np.sqrt(dfHistory['val_loss'])
            dfRMSE.plot(y='rmse', ax=ax, c=(0, 1-lsBlues[nFold], lsBlues[nFold]), label='Loss Fold {}'.format(nFold))
            dfRMSE.plot(y='val_rmse', ax=ax, c=(1, lsOranges[nFold], 0), label='Val Loss Fold {}'.format(nFold))

        ax.legend()
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        if strTitle is not None:
            ax.set_title(strTitle)
        else:
            strTitle = self.strModelName
            ax.set_title(strTitle)
        if strSavePath is not None:
            fig.savefig(strSavePath)
        plt.show()

        return fig


    def summary(self):
        """
        Print the model's .ini file
        :return:
        """
        with open(self.strModelPath + '/' + self.strModelName + '.ini', 'r') as file:
            for strLine in file:
                sys.stdout.write(strLine)

class KerasJson:
    def __init__(self, strModelName='model', strKerasVersion='2.2.4', strBackend='tensorflow'):
        self.dictModel = {'class_name': 'Model',
                          'config': {'name': strModelName,
                                     'layers': [],
                                     'input_layers': [],
                                     'output_layers': []},
                          'keras_version': strKerasVersion,
                          'backend': strBackend}
        self.nLayers = 0

    def add_layer(self, class_name, layer_name, inbound_nodes, **kwargs):
        dictLayer = {'name': layer_name,
                     'class_name': class_name,
                     'config': kwargs,
                     'inbound_nodes': inbound_nodes
                    }
        dictLayer['config']['name'] = layer_name
        self.dictModel['config']['layers'] += [dictLayer]
        self.nLayers += 1

    def set_input_layers(self, lsLayers):
        self.dictModel['config']['input_layers'] = [[strLayer, 0, 0] for strLayer in lsLayers]

    def set_output_layers(self, lsLayers):
        self.dictModel['config']['output_layers'] = [[strLayer, 0, 0] for strLayer in lsLayers]

    def get_dict(self):
        return self.dictModel

    def get_string(self):
        return json.dumps(self.dictModel, default=_convert_np_int)

    def save(self, strSavePath):
        with open(strSavePath, 'w') as file:
            json.dump(self.dictModel, file, default=_convert_np_int, indent='\t')

def _convert_np_int(obj):
    if isinstance(obj, np.int64):
        return int(obj)
    else:
        raise TypeError
