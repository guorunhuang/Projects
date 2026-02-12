'''
Number of unique images for feature extraction: 357

============================================================
Training target: Dry_Green_g
============================================================
Target: Dry_Green_g - Train R^2: 0.3517 - Val R^2: 0.3913

============================================================
Training target: Dry_Dead_g
============================================================
Target: Dry_Dead_g - Train R^2: 0.4560 - Val R^2: -0.0554

============================================================
Training target: Dry_Clover_g
============================================================
Target: Dry_Clover_g - Train R^2: 0.5832 - Val R^2: 0.6481

============================================================
Training target: GDM_g
============================================================
Target: GDM_g - Train R^2: 0.3524 - Val R^2: 0.5116

============================================================
Training target: Dry_Total_g
============================================================
Target: Dry_Total_g - Train R^2: 0.3898 - Val R^2: 0.2550

Model training completed!

Train takes: 126.72 seconds
Predicting: 0/5

Prediction completed! Submission file saved as submission.csv
Submission file shape: (5, 2)

Prediction statistics:
              count       mean  std        min        25%        50%  \
target_name                                                            
Dry_Clover_g    1.0   0.000000  NaN   0.000000   0.000000   0.000000   
Dry_Dead_g      1.0   9.138971  NaN   9.138971   9.138971   9.138971   
Dry_Green_g     1.0  35.033707  NaN  35.033707  35.033707  35.033707   
Dry_Total_g     1.0  67.335342  NaN  67.335342  67.335342  67.335342   
GDM_g           1.0  31.856451  NaN  31.856451  31.856451  31.856451   

                    75%        max  
target_name                         
Dry_Clover_g   0.000000   0.000000  
Dry_Dead_g     9.138971   9.138971  
Dry_Green_g   35.033707  35.033707  
Dry_Total_g   67.335342  67.335342  
GDM_g         31.856451  31.856451  

Total process takes: 127.10 seconds
'''