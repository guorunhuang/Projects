'''
Target: Dry_Green_g - Train R2: 0.9814 - Val R2: 0.6464
   feature  importance
     State    0.075527
 Height_cm    0.043590
resnet_491    0.037221
resnet_280    0.032701
 resnet_66    0.022075
resnet_239    0.019123
resnet_123    0.018226
resnet_502    0.017673
resnet_381    0.016511
ExG_median    0.016320
resnet_268    0.016262
resnet_448    0.015967
resnet_344    0.014031
resnet_211    0.013761
resnet_187    0.013299
resnet_120    0.013206
resnet_326    0.012864
resnet_171    0.012679
 resnet_95    0.012222
resnet_391    0.011254
Target: Dry_Dead_g - Train R2: 0.9882 - Val R2: 0.4329
     feature  importance
  resnet_431    0.033122
  resnet_192    0.026251
  resnet_402    0.025850
  resnet_311    0.024245
  resnet_355    0.023618
HSV_H_hist_4    0.018974
  resnet_299    0.016562
  resnet_279    0.015421
  resnet_266    0.014394
  resnet_460    0.013055
  resnet_441    0.011611
  ExG_hist_0    0.011575
  resnet_507    0.011524
  resnet_264    0.010803
  resnet_397    0.010599
  resnet_187    0.010598
  resnet_490    0.010369
       Month    0.009865
  resnet_336    0.009060
  resnet_450    0.008796
Target: Dry_Clover_g - Train R2: 0.9995 - Val R2: 0.5878
     feature  importance
   resnet_66    0.073513
  resnet_107    0.060403
  resnet_498    0.055200
  resnet_341    0.037036
  resnet_149    0.031031
  resnet_118    0.022089
     Species    0.022080
  resnet_434    0.019981
  resnet_169    0.019733
  resnet_227    0.018198
   CIVE_mean    0.018043
  resnet_354    0.014940
  resnet_316    0.014265
HSV_H_hist_4    0.013963
   resnet_93    0.013567
  resnet_473    0.012549
  resnet_342    0.012456
  resnet_314    0.012339
       State    0.012172
   resnet_26    0.011669
Target: GDM_g - Train R2: 0.9511 - Val R2: 0.6518
     feature  importance
       State    0.063525
   Height_cm    0.041637
  resnet_491    0.037689
  resnet_381    0.018535
  resnet_344    0.018063
        NDVI    0.017855
  ExR_hist_9    0.015839
   resnet_66    0.015582
   resnet_94    0.013997
HSV_H_hist_4    0.012987
  resnet_277    0.012081
  ExR_median    0.011173
  resnet_171    0.010749
  resnet_211    0.010190
   resnet_55    0.010066
   resnet_71    0.009820
  resnet_270    0.009690
  resnet_168    0.009398
  resnet_327    0.008840
    ExR_mean    0.008576
Target: Dry_Total_g - Train R2: 0.9999 - Val R2: 0.5423
   feature  importance
     State    0.112873
 Height_cm    0.046597
resnet_208    0.038386
resnet_344    0.036757
resnet_346    0.028005
 resnet_71    0.022970
resnet_491    0.018806
resnet_302    0.014942
resnet_207    0.014513
resnet_381    0.012840
  resnet_8    0.012040
resnet_448    0.011862
resnet_480    0.011823
resnet_305    0.011377
 resnet_66    0.010643
      NDVI    0.009935
resnet_379    0.009445
resnet_249    0.009128
resnet_120    0.007926
resnet_426    0.007698
Model training completed.

Train takes: 182.58 seconds
Predicting: 0/5

Prediction completed! Submission file saved as submission.csv
Submission file shape: (5, 2)

Prediction statistics:
              count       mean  std        min        25%        50%  \
target_name                                                            
Dry_Clover_g    1.0  13.773639  NaN  13.773639  13.773639  13.773639   
Dry_Dead_g      1.0  22.326494  NaN  22.326494  22.326494  22.326494   
Dry_Green_g     1.0  17.789762  NaN  17.789762  17.789762  17.789762   
Dry_Total_g     1.0  45.592762  NaN  45.592762  45.592762  45.592762   
GDM_g           1.0  27.252375  NaN  27.252375  27.252375  27.252375   

                    75%        max  
target_name                         
Dry_Clover_g  13.773639  13.773639  
Dry_Dead_g    22.326494  22.326494  
Dry_Green_g   17.789762  17.789762  
Dry_Total_g   45.592762  45.592762  
GDM_g         27.252375  27.252375  

Total process takes: 183.48 seconds
'''