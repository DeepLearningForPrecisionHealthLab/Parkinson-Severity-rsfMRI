# Predicting Longitudinal Severity in Parkinson’s Disease using Current Clinical and Neuroimaging Measures
This project contains code for our manuscript "Predicting Longitudinal Severity in Parkinson’s Disease using Current Clinical and Neuroimaging Measures" current under journal review. 

## Files
### Model selection and hyperparameter optimization
- `updrstotal_prediction.py` is the main script for training and evaluating a set of models to predict MDS-UPDRS total score at a specified time, using a specified set of rs-fMRI inputs. Model hyperparameters are optimized using a random search, and model performance is evaluated using nested cross-validation. 
- `updrstotal_prediction_loso.py` performs a leave-one-site-out version of this analysis
### Measurment of test performance 
- After running these 2 scripts, `collect_results.py` and `collect_results_loso.py` identify the best performing models and hyperparameter configurations for
each target and feature type based on mean inner cross-validation performance. Evaluate these models on the held-out data of the outer cross-validation loop. 

## Main dependencies
* nilearn >= 0.6.2
* scipy >= 1.2.0
* scikit-learn >= 0.20.2
* pandas >= 0.25.3
* numpy >= 1.15.4