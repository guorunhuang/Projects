'''
Files
test.csv

sample_id — Unique identifier for each prediction row (one row per image–target pair).
image_path — Relative path to the image (e.g., test/ID1001187975.jpg).
target_name — Name of the biomass component to predict for this row. One of: Dry_Green_g, Dry_Dead_g, Dry_Clover_g, GDM_g, Dry_Total_g.
The test set contains over 800 images.

train/

Directory containing training images (JPEG), referenced by image_path.
test/

Directory reserved for test images (hidden at scoring time); paths in test.csv point here.
train.csv

sample_id — Unique identifier for each training sample (image).
image_path — Relative path to the training image (e.g., images/ID1098771283.jpg).
Sampling_Date — Date of sample collection.
State — Australian state where sample was collected.
Species — Pasture species present, ordered by biomass (underscore-separated).
Pre_GSHH_NDVI — Normalized Difference Vegetation Index (GreenSeeker) reading.
Height_Ave_cm — Average pasture height measured by falling plate (cm).
target_name — Biomass component name for this row (Dry_Green_g, Dry_Dead_g, Dry_Clover_g, GDM_g, or Dry_Total_g).
target — Ground-truth biomass value (grams) corresponding to target_name for this image.
sample_submission.csv

sample_id — Copy from test.csv; one row per requested (image, target_name) pair.
target — Your predicted biomass value (grams) for that sample_id.
What you must predict
For each sample_id in test.csv, output a single numeric target value in sample_submission.csv. 
Each row corresponds to one (image_path, target_name) pair; you must provide the predicted biomass (grams) 
for that component. The actual test images are made available to your notebook at scoring time.
'''

'''
Using device: cpu
Loading training data...
Training data shape: (1785, 9)
Target variables: ['Dry_Clover_g' 'Dry_Dead_g' 'Dry_Green_g' 'Dry_Total_g' 'GDM_g']

==================================================
Extracting features from unique images (will be reused for all targets)
==================================================
Number of unique images: 357
Extracting features...
Processing: 0/357
Processing: 100/357
Processing: 200/357
Processing: 300/357
Processing: 400/357
Processing: 500/357
Processing: 600/357
Processing: 700/357
Processing: 800/357
Processing: 900/357
Processing: 1000/357
Processing: 1100/357
Processing: 1200/357
Processing: 1300/357
Processing: 1400/357
Processing: 1500/357
Processing: 1600/357
Processing: 1700/357
Total features extracted: 26

==================================================
Training target: Dry_Green_g
==================================================
Training MLP with 26 input features...
Early stopping at epoch 18
Training R²: 0.5408
Validation R²: 0.2815

==================================================
Training target: Dry_Dead_g
==================================================
Training MLP with 26 input features...
Early stopping at epoch 68
Training R²: 0.5807
Validation R²: 0.2652

==================================================
Training target: Dry_Clover_g
==================================================
Training MLP with 26 input features...
Early stopping at epoch 31
Training R²: 0.7043
Validation R²: 0.6625

==================================================
Training target: GDM_g
==================================================
Training MLP with 26 input features...
Early stopping at epoch 18
Training R²: 0.5519
Validation R²: 0.3956

==================================================
Training target: Dry_Total_g
==================================================
Training MLP with 26 input features...
Early stopping at epoch 23
Training R²: 0.4625
Validation R²: 0.1569

Model training completed!

Train takes: 112.91 seconds
Loading test data...
Number of test images: 1
Extracting features...
Processing: 0/1
Adding 3 missing features: ['Month', 'Species', 'State']...
Prediction progress: 0/5

Prediction completed! Submission file saved as submission.csv
Submission file shape: (5, 2)

Prediction statistics:
              count       mean  std        min        25%        50%  \
sample_id                                                              
Dry_Clover_g    1.0   0.000000  NaN   0.000000   0.000000   0.000000   
Dry_Dead_g      1.0  10.643952  NaN  10.643952  10.643952  10.643952   
Dry_Green_g     1.0  36.046967  NaN  36.046967  36.046967  36.046967   
Dry_Total_g     1.0  62.635036  NaN  62.635036  62.635036  62.635036   
GDM_g           1.0  43.762878  NaN  43.762878  43.762878  43.762878   

                    75%        max  
sample_id                           
Dry_Clover_g   0.000000   0.000000  
Dry_Dead_g    10.643952  10.643952  
Dry_Green_g   36.046967  36.046967  
Dry_Total_g   62.635036  62.635036  
GDM_g         43.762878  43.762878  

Total process takes: 113.24 seconds
'''