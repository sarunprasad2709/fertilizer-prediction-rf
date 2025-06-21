# fertilizer-prediction-rf
Random Forest model for fertilizer prediction

A machine learning solution for predicting optimal fertilizer types based on soil and crop conditions using Random Forest classification.

 ## Dataset Overview
- Training Data: 750,000 samples with 9 features
- Test Data: 250,000 samples  
- Target Classes: 7 fertilizer types (14-35-14, 10-26-26, 17-17-17, 28-28, 20-20, DAP, Urea)
- Features: Temperature, Humidity, Moisture, Soil Type, Crop Type, Nitrogen, Potassium, Phosphorous

## Model Performance
- Algorithm: Random Forest (100 estimators, max_depth=20)
- Validation Accuracy: 17.34%
- MAP@3 Score: 0.3036
- Top Features: Phosphorous (17.1%), Nitrogen (17.0%), Moisture (15.7%)

## Files
- fertilizer_prediction.py - Complete code
- submission.csv - Final predictions
