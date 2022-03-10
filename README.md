# Final Project - Machine Learning - MC^2
Authors:
- Marugán Pinos, Rocío
- Pérez Maiztegui, Alex

# Objective:

Obtain a prediction for the uploaded data using four separate models (one for each predicted variable) trained with the deep learning package h2o.

# How to use:

1.First of all, the script *“Predict_Marugan_PerezMaiztegui.R”* and the models *(“modelo_M”, “modelo_MM”, “modelo_Combo”, “modelo_DCombo”)* should be downloaded to a folder which must be set as working directory.  Bear in mind that the query dataset must be also found in this directory.

2. The query file must be named as: *“X_test.csv”*. 
3. Run the script. Notice, that in the first run all packages will be loaded, and if required, installed, so this might take some time.
4. The output, the predicted results will be automatically saved as: *“Y_test.csv”* in the same path. Subsequent running of the script will overwrite this file.

# Additional aspects:

The  script used to develop and train the models used in this prediction is available in the same folder as: *“Model_synthesis_Marugan_Perez.r”*
