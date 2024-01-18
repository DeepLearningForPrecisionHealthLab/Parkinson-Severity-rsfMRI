# Predicting Parkinson's disease trajectory using clinical and neuroimaging baseline measures
Nguyen KP, Raval V, Treacher A, Mellema C, Yu FF, Pinho MC, et al. (2021): Predicting Parkinson's disease trajectory using clinical and neuroimaging baseline measures. Parkinsonism & related disorders 85: 44–51.
[10.1016/j.parkreldis.2021.02.0260](https://doi.org/10.1016/j.parkreldis.2021.02.0260)

## Files
### Subject cohorts
- `subject_lists.xlsx` contains the subject and session IDs from the [PPMI database](https://www.ppmi-info.org/) for the images included in our analysis.
### Model selection and hyperparameter optimization
- `updrstotal_prediction.py` is the main script for training and evaluating a set of models to predict MDS-UPDRS total score at a specified time, using a specified set of rs-fMRI inputs. Model hyperparameters are optimized using a random search, and model performance is evaluated using nested cross-validation. 
- `updrstotal_prediction_loso.py` performs a leave-one-site-out version of this analysis
### Measurment of test performance 
- After running these 2 scripts, `collect_results.py` and `collect_results_loso.py` identify the best performing models and hyperparameter configurations for
each target and feature type based on mean inner cross-validation performance. Evaluate these models on the held-out data of the outer cross-validation loop. 

## Additional resources
To further facilitate reproducibility, we have made some [additional files available here](https://cloud.biohpc.swmed.edu/index.php/s/LsiC2KetzJFNy8w). These include:
- Our derived ReHo and fALFF maps
- Docker image with our custom fMRI preprocessing pipeline
- Our modified [Schaefer 2018](https://pubmed.ncbi.nlm.nih.gov/28981612/) parcellation with subcortical ROIs and the anatomical ROI labels used in our figures. The BASC197 and BASC444 parcellations are available through nilearn. 

## Main dependencies
* nilearn >= 0.6.2
* scipy >= 1.2.0
* scikit-learn >= 0.20.2
* pandas >= 0.25.3
* numpy >= 1.15.4

## License
Copyright (c) 2021 The University of Texas Southwestern Medical Center.
All rights reserved.
 
Redistribution and use in source and binary forms, with or without
modification, are permitted for academic and research use only (subject to the limitations in the disclaimer below) provided that the following conditions are met:
 
* Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

* Neither the name of the copyright holders nor the names of its
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.
 
NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.