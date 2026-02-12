'''
Loading training data...
Training data shape: (1785, 9)
Target variables: ['Dry_Clover_g' 'Dry_Dead_g' 'Dry_Green_g' 'Dry_Total_g' 'GDM_g']

==================================================
Extracting features from unique images (this will be reused for all targets)
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

==================================================
Training target: Dry_Green_g
==================================================
Training R²: 0.9997
Validation R²: 0.7613

All Feature Importance Scores:
                feature  importance
              Height_cm    0.195366
                  State    0.171623
             ExG_median    0.035714
             ExG_hist_9    0.033064
             ExG_hist_7    0.025618
             ExG_hist_8    0.025285
               ExG_mean    0.022831
             ExR_hist_2    0.019091
               B_hist_6    0.016018
           HSV_H_hist_2    0.015194
           HSV_H_hist_4    0.014647
               CIVE_std    0.014366
               B_hist_2    0.013824
           HSV_H_hist_0    0.013335
GLCM_dissimilarity_mean    0.013234
               R_hist_7    0.012498
              HSV_H_std    0.011571
                   NDVI    0.010780
           HSV_H_hist_1    0.009741
           HSV_H_hist_3    0.008043
             ExG_hist_4    0.007827
                ExG_max    0.007480
 GLCM_dissimilarity_std    0.007381
                Species    0.007322
           HSV_S_hist_3    0.007202
           HSV_V_hist_0    0.007017
               R_hist_4    0.006988
   GLCM_correlation_std    0.006584
          GLCM_ASM_mean    0.006310
               R_hist_2    0.006014
               G_hist_8    0.005977
             ExG_hist_5    0.005757
               G_hist_6    0.005737
           HSV_V_hist_2    0.005662
             ExG_hist_1    0.005465
           HSV_H_hist_6    0.005409
                 G_mean    0.005309
  GLCM_correlation_mean    0.005002
               R_hist_6    0.004994
                  Month    0.004917
               G_hist_1    0.004917
               R_hist_8    0.004813
             ExR_hist_4    0.004634
               ExR_mean    0.004433
           HSV_S_hist_2    0.004188
           HSV_H_hist_7    0.004188
              DayOfYear    0.003960
           HSV_V_hist_5    0.003846
               R_hist_5    0.003792
           HSV_H_hist_5    0.003685
              HSV_S_std    0.003632
       GLCM_energy_mean    0.003626
                  R_std    0.003621
  GLCM_homogeneity_mean    0.003615
     GLCM_contrast_mean    0.003597
              CIVE_mean    0.003577
             ExR_hist_1    0.003561
           HSV_S_median    0.003469
                ExG_min    0.003462
               G_hist_5    0.003368
             ExR_hist_6    0.003318
               B_hist_9    0.003205
             ExG_hist_3    0.003088
               R_hist_1    0.003055
      GLCM_contrast_std    0.003028
             ExR_hist_8    0.003025
             ExG_hist_0    0.002997
               B_hist_1    0.002949
             ExR_hist_0    0.002947
               G_hist_9    0.002926
               G_hist_4    0.002736
           HSV_V_hist_3    0.002726
             ExR_median    0.002707
           HSV_S_hist_1    0.002598
             ExG_hist_2    0.002545
           HSV_S_hist_0    0.002536
             ExR_hist_9    0.002517
               G_median    0.002516
               R_hist_3    0.002470
               R_hist_9    0.002464
               B_hist_0    0.002457
             HSV_H_mean    0.002447
               G_hist_0    0.002420
               B_hist_3    0.002407
               B_hist_8    0.002355
         green_coverage    0.002295
                ExG_std    0.002197
             ExR_hist_7    0.001976
              VARI_mean    0.001953
                ExR_std    0.001916
   GLCM_homogeneity_std    0.001781
               VARI_std    0.001750
             HSV_S_mean    0.001711
               B_hist_5    0.001704
               G_hist_2    0.001661
                NDI_std    0.001590
           HSV_V_hist_7    0.001489
               R_median    0.001483
             ExR_hist_3    0.001460
             ExG_hist_6    0.001458
                 R_mean    0.001408
               B_hist_4    0.001238
                 B_mean    0.001238
           HSV_S_hist_7    0.001099
                  G_std    0.001035
               G_hist_7    0.001005
               B_median    0.000976
        GLCM_energy_std    0.000972
           HSV_V_hist_1    0.000967
                  B_std    0.000905
           HSV_S_hist_6    0.000885
           HSV_H_median    0.000867
               B_hist_7    0.000852
           HSV_V_median    0.000843
             ExR_hist_5    0.000842
           HSV_S_hist_4    0.000784
               NDI_mean    0.000771
           HSV_V_hist_4    0.000738
           GLCM_ASM_std    0.000717
               G_hist_3    0.000712
               R_hist_0    0.000674
             HSV_V_mean    0.000434
           HSV_S_hist_5    0.000399
           HSV_V_hist_6    0.000363
              HSV_V_std    0.000230

==================================================
Training target: Dry_Dead_g
==================================================
Training R²: 0.9997
Validation R²: 0.5315

All Feature Importance Scores:
                feature  importance
             ExG_hist_0    0.061566
           HSV_H_hist_4    0.052781
                  Month    0.046602
              HSV_S_std    0.034325
             ExR_hist_9    0.033688
               ExR_mean    0.030469
              Height_cm    0.025336
           HSV_H_hist_5    0.024713
             ExG_hist_1    0.021714
                  State    0.020411
             HSV_H_mean    0.019679
             ExG_median    0.019110
      GLCM_contrast_std    0.018779
              DayOfYear    0.017329
             ExG_hist_7    0.017072
           HSV_H_hist_7    0.015259
               VARI_std    0.013377
                Species    0.012505
                   NDVI    0.012438
               G_hist_1    0.012304
             ExG_hist_3    0.011799
           HSV_H_hist_0    0.011178
              VARI_mean    0.011091
               G_hist_0    0.010686
               R_hist_9    0.010479
               B_hist_9    0.009820
           HSV_H_hist_2    0.009404
           HSV_S_hist_0    0.009346
           HSV_S_median    0.009098
             ExG_hist_5    0.008851
          GLCM_ASM_mean    0.008830
           GLCM_ASM_std    0.008766
             ExG_hist_2    0.008714
           HSV_S_hist_2    0.008384
             ExR_hist_0    0.008370
       GLCM_energy_mean    0.008365
               R_hist_2    0.007896
              HSV_H_std    0.007850
               B_hist_1    0.007765
           HSV_V_hist_0    0.007746
                ExR_std    0.007727
               G_hist_6    0.007602
           HSV_H_median    0.007400
             HSV_S_mean    0.007359
 GLCM_dissimilarity_std    0.007326
             ExR_hist_8    0.007017
               B_hist_5    0.006966
           HSV_H_hist_3    0.006954
               B_hist_8    0.006788
                  G_std    0.006103
                  R_std    0.005985
               R_hist_1    0.005947
                 B_mean    0.005940
   GLCM_correlation_std    0.005827
               B_hist_0    0.005770
             ExR_median    0.005756
               R_hist_3    0.005714
     GLCM_contrast_mean    0.005582
              CIVE_mean    0.005515
               B_hist_2    0.005155
               G_hist_2    0.005139
  GLCM_homogeneity_mean    0.005106
           HSV_H_hist_1    0.005095
              HSV_V_std    0.005037
               B_median    0.005029
             ExG_hist_9    0.004800
  GLCM_correlation_mean    0.004675
               G_median    0.004560
               R_hist_6    0.004222
           HSV_V_hist_6    0.004210
        GLCM_energy_std    0.004164
           HSV_H_hist_6    0.004108
               R_hist_0    0.004084
               CIVE_std    0.004077
             HSV_V_mean    0.003990
         green_coverage    0.003884
   GLCM_homogeneity_std    0.003677
                  B_std    0.003645
           HSV_V_hist_7    0.003450
                 R_mean    0.003305
             ExR_hist_1    0.003213
           HSV_S_hist_3    0.003153
               ExG_mean    0.003130
                ExG_min    0.003119
               B_hist_7    0.003037
           HSV_V_hist_3    0.003022
               R_hist_7    0.003006
             ExG_hist_4    0.002951
               R_hist_5    0.002946
               G_hist_9    0.002946
             ExG_hist_8    0.002866
             ExR_hist_4    0.002820
               R_hist_8    0.002777
           HSV_S_hist_4    0.002700
               G_hist_4    0.002525
           HSV_S_hist_7    0.002515
             ExG_hist_6    0.002454
                ExG_max    0.002425
               NDI_mean    0.002377
               B_hist_4    0.002355
               R_hist_4    0.002339
             ExR_hist_6    0.002289
                ExG_std    0.002041
               G_hist_7    0.002013
               R_median    0.001936
               G_hist_3    0.001829
           HSV_V_hist_1    0.001813
               B_hist_6    0.001773
                NDI_std    0.001758
           HSV_S_hist_6    0.001713
           HSV_S_hist_5    0.001491
               G_hist_8    0.001468
               G_hist_5    0.001439
               B_hist_3    0.001426
                 G_mean    0.001413
             ExR_hist_2    0.001368
GLCM_dissimilarity_mean    0.001268
           HSV_V_median    0.001194
           HSV_S_hist_1    0.001086
             ExR_hist_7    0.000910
             ExR_hist_3    0.000749
             ExR_hist_5    0.000737
           HSV_V_hist_4    0.000399
           HSV_V_hist_5    0.000321
           HSV_V_hist_2    0.000307

==================================================
Training target: Dry_Clover_g
==================================================
Training R²: 0.9653
Validation R²: 0.7264

All Feature Importance Scores:
                feature  importance
             ExG_hist_7    0.062191
                Species    0.054106
           HSV_S_hist_4    0.041192
  GLCM_homogeneity_mean    0.038672
             ExR_hist_9    0.033686
                   NDVI    0.032954
                  State    0.032321
              CIVE_mean    0.029009
             ExR_hist_4    0.028560
             ExG_hist_5    0.023344
              VARI_mean    0.020606
           GLCM_ASM_std    0.019394
               G_hist_3    0.018814
           HSV_H_hist_7    0.018497
             ExG_hist_8    0.017124
           HSV_H_hist_5    0.016651
           HSV_H_hist_4    0.016042
                  Month    0.015373
           HSV_V_median    0.012567
               R_hist_8    0.011390
              DayOfYear    0.011170
               B_hist_9    0.011158
           HSV_H_hist_3    0.011115
         green_coverage    0.010774
           HSV_H_hist_1    0.010660
             ExG_hist_6    0.010610
             HSV_V_mean    0.010602
   GLCM_homogeneity_std    0.010415
              HSV_H_std    0.009975
               G_hist_2    0.009949
             ExG_hist_0    0.009801
             ExG_median    0.009656
                 G_mean    0.009612
               R_hist_1    0.009556
               G_median    0.009155
           HSV_S_hist_0    0.008902
   GLCM_correlation_std    0.008336
           HSV_V_hist_5    0.008022
           HSV_S_hist_1    0.007842
             HSV_H_mean    0.007618
           HSV_S_hist_6    0.007574
             ExR_hist_5    0.007306
               CIVE_std    0.007124
                  R_std    0.006856
             ExR_hist_3    0.006825
                ExG_min    0.006314
               NDI_mean    0.006128
           HSV_V_hist_1    0.005991
               R_hist_2    0.005838
             ExG_hist_9    0.005770
               B_hist_5    0.005707
 GLCM_dissimilarity_std    0.005404
                ExG_max    0.005320
               G_hist_6    0.005242
           HSV_H_hist_2    0.005187
               R_median    0.005130
              Height_cm    0.004887
             ExR_hist_7    0.004796
               R_hist_3    0.004752
           HSV_S_hist_3    0.004599
           HSV_V_hist_7    0.004574
               ExR_mean    0.004495
             ExR_median    0.004360
          GLCM_ASM_mean    0.004330
             ExR_hist_0    0.004241
           HSV_H_hist_0    0.004181
                  B_std    0.004150
GLCM_dissimilarity_mean    0.004124
                 B_mean    0.003855
               ExG_mean    0.003815
               G_hist_9    0.003722
           HSV_V_hist_3    0.003704
             HSV_S_mean    0.003581
               VARI_std    0.003541
               G_hist_1    0.003096
           HSV_S_median    0.002999
      GLCM_contrast_std    0.002816
              HSV_V_std    0.002782
                ExG_std    0.002535
               B_hist_7    0.002449
           HSV_V_hist_0    0.002441
                  G_std    0.002396
               B_hist_3    0.002380
               B_hist_2    0.002344
               G_hist_0    0.002341
               R_hist_9    0.002331
               B_hist_0    0.002311
     GLCM_contrast_mean    0.002298
               B_hist_6    0.002292
             ExR_hist_2    0.002226
                ExR_std    0.002209
               R_hist_7    0.002121
       GLCM_energy_mean    0.002103
               G_hist_5    0.002084
           HSV_V_hist_2    0.002071
           HSV_V_hist_4    0.002068
        GLCM_energy_std    0.002033
           HSV_S_hist_5    0.001970
           HSV_H_hist_6    0.001928
               B_hist_8    0.001897
               R_hist_0    0.001689
  GLCM_correlation_mean    0.001686
             ExG_hist_1    0.001611
             ExG_hist_3    0.001572
               B_hist_1    0.001497
           HSV_H_median    0.001420
               G_hist_7    0.001413
               B_median    0.001342
                 R_mean    0.001283
               R_hist_6    0.001274
             ExR_hist_6    0.001269
           HSV_S_hist_7    0.001257
             ExR_hist_1    0.001226
               R_hist_4    0.001160
              HSV_S_std    0.001131
               R_hist_5    0.001052
                NDI_std    0.000971
               G_hist_8    0.000898
             ExG_hist_2    0.000815
               G_hist_4    0.000749
               B_hist_4    0.000736
             ExR_hist_8    0.000693
           HSV_S_hist_2    0.000660
           HSV_V_hist_6    0.000656
             ExG_hist_4    0.000572

==================================================
Training target: GDM_g
==================================================
Training R²: 0.9997
Validation R²: 0.7140

All Feature Importance Scores:
                feature  importance
              Height_cm    0.168814
                  State    0.111645
           HSV_H_hist_7    0.036288
                   NDVI    0.036227
  GLCM_homogeneity_mean    0.035392
             ExG_hist_9    0.035117
             ExR_median    0.034051
             ExG_hist_8    0.032511
               ExR_mean    0.024822
             ExR_hist_9    0.017990
  GLCM_correlation_mean    0.016433
               CIVE_std    0.016087
              HSV_H_std    0.015391
                  Month    0.014624
             ExG_median    0.014124
                ExG_std    0.013343
             ExG_hist_7    0.012998
              CIVE_mean    0.012214
          GLCM_ASM_mean    0.012032
              DayOfYear    0.010972
               B_hist_6    0.010497
               NDI_mean    0.010243
           HSV_H_hist_4    0.009205
               ExG_mean    0.008776
           HSV_H_hist_1    0.008576
               G_hist_1    0.008496
               R_hist_4    0.008303
           HSV_H_hist_0    0.007789
             ExG_hist_6    0.006942
   GLCM_correlation_std    0.006855
                Species    0.006766
               R_hist_6    0.006438
               R_hist_5    0.006292
         green_coverage    0.006264
               G_hist_0    0.006227
           HSV_S_hist_7    0.006041
               R_hist_1    0.006006
           HSV_S_hist_6    0.005564
                 G_mean    0.005017
             ExR_hist_8    0.004717
                ExG_max    0.004687
           HSV_V_hist_2    0.004645
             ExR_hist_7    0.004620
           HSV_V_hist_3    0.004447
               R_hist_2    0.004035
             HSV_H_mean    0.003962
           HSV_S_hist_2    0.003735
 GLCM_dissimilarity_std    0.003648
                  R_std    0.003613
               G_hist_8    0.003530
             ExR_hist_3    0.003494
               G_hist_5    0.003388
               B_hist_2    0.003351
             ExR_hist_4    0.003320
           HSV_V_hist_0    0.003167
      GLCM_contrast_std    0.003162
               G_median    0.003125
           HSV_S_median    0.003072
                ExR_std    0.003067
             ExG_hist_5    0.002999
               G_hist_4    0.002963
               G_hist_2    0.002890
           HSV_V_hist_5    0.002880
               R_hist_0    0.002862
               B_hist_9    0.002858
             ExG_hist_1    0.002799
           HSV_H_median    0.002786
              VARI_mean    0.002773
GLCM_dissimilarity_mean    0.002727
               B_hist_8    0.002718
           HSV_V_hist_7    0.002671
           GLCM_ASM_std    0.002649
              HSV_S_std    0.002549
               B_hist_3    0.002536
               R_median    0.002527
                  B_std    0.002506
               B_hist_1    0.002471
               B_hist_7    0.002393
           HSV_V_hist_6    0.002322
       GLCM_energy_mean    0.002306
               G_hist_9    0.002291
               B_hist_0    0.002277
             ExR_hist_6    0.002221
           HSV_S_hist_0    0.002142
               G_hist_7    0.002138
                 B_mean    0.002118
           HSV_H_hist_3    0.001905
           HSV_S_hist_1    0.001870
               R_hist_3    0.001821
                ExG_min    0.001817
                  G_std    0.001777
           HSV_V_hist_1    0.001765
     GLCM_contrast_mean    0.001667
             ExG_hist_0    0.001659
   GLCM_homogeneity_std    0.001640
           HSV_H_hist_5    0.001606
              HSV_V_std    0.001604
             ExG_hist_2    0.001570
           HSV_H_hist_6    0.001526
           HSV_S_hist_4    0.001478
               R_hist_7    0.001448
             HSV_S_mean    0.001406
           HSV_S_hist_3    0.001371
           HSV_H_hist_2    0.001313
               B_hist_5    0.001297
               VARI_std    0.001292
             ExR_hist_5    0.001088
               B_median    0.001039
                NDI_std    0.000960
                 R_mean    0.000861
             HSV_V_mean    0.000811
             ExR_hist_1    0.000799
               B_hist_4    0.000739
           HSV_S_hist_5    0.000725
               R_hist_9    0.000715
               G_hist_6    0.000693
               R_hist_8    0.000643
             ExR_hist_0    0.000626
               G_hist_3    0.000581
        GLCM_energy_std    0.000555
             ExG_hist_4    0.000434
             ExR_hist_2    0.000430
           HSV_V_median    0.000413
             ExG_hist_3    0.000321
           HSV_V_hist_4    0.000209

==================================================
Training target: Dry_Total_g
==================================================
Training R²: 0.9998
Validation R²: 0.6618

All Feature Importance Scores:
                feature  importance
              Height_cm    0.155413
                  State    0.142168
             ExG_hist_8    0.074864
           HSV_H_hist_7    0.034965
             ExG_hist_9    0.024906
   GLCM_correlation_std    0.020962
             ExG_median    0.020223
           HSV_H_hist_0    0.018592
              DayOfYear    0.016633
           HSV_H_hist_4    0.015985
              HSV_H_std    0.015385
                   NDVI    0.014663
               R_hist_2    0.012169
                  Month    0.012090
           HSV_S_median    0.011907
  GLCM_correlation_mean    0.010615
             ExG_hist_7    0.010396
             ExR_hist_8    0.010351
           HSV_V_median    0.009734
  GLCM_homogeneity_mean    0.009178
             HSV_H_mean    0.009085
               CIVE_std    0.008974
             ExR_hist_9    0.008930
                Species    0.008734
               G_hist_5    0.008623
                ExG_std    0.008013
               R_hist_4    0.007197
             ExG_hist_0    0.006992
           HSV_V_hist_2    0.006503
           HSV_S_hist_3    0.006292
               R_hist_6    0.006014
               ExG_mean    0.005753
           HSV_V_hist_5    0.005685
               G_hist_2    0.005390
           HSV_V_hist_0    0.005350
               G_hist_1    0.005348
                 G_mean    0.005278
           HSV_S_hist_7    0.005268
               B_hist_8    0.005221
           HSV_S_hist_6    0.005101
             HSV_S_mean    0.004990
               G_hist_8    0.004915
             ExR_hist_7    0.004893
               VARI_std    0.004830
           HSV_S_hist_2    0.004593
               G_hist_4    0.004466
               G_hist_3    0.004336
               R_hist_5    0.004268
               G_hist_6    0.004242
           HSV_H_median    0.004197
           HSV_H_hist_2    0.004169
             ExG_hist_3    0.004152
               R_hist_1    0.004146
             ExG_hist_2    0.004096
               G_hist_9    0.003950
                 B_mean    0.003933
               B_hist_6    0.003901
              HSV_S_std    0.003669
                ExR_std    0.003658
               G_hist_7    0.003498
              VARI_mean    0.003461
        GLCM_energy_std    0.003393
             ExR_median    0.003370
               B_hist_1    0.003366
                  G_std    0.003350
               R_median    0.003226
             ExG_hist_4    0.003212
                NDI_std    0.003196
       GLCM_energy_mean    0.003171
               B_hist_9    0.003161
               B_hist_0    0.003156
         green_coverage    0.003121
           HSV_V_hist_7    0.003120
               G_hist_0    0.003049
 GLCM_dissimilarity_std    0.003039
           HSV_H_hist_1    0.002958
           HSV_V_hist_1    0.002930
               R_hist_7    0.002906
             ExG_hist_5    0.002899
           HSV_H_hist_5    0.002785
           HSV_S_hist_1    0.002751
                  R_std    0.002682
           HSV_V_hist_3    0.002681
           GLCM_ASM_std    0.002665
   GLCM_homogeneity_std    0.002628
           HSV_H_hist_6    0.002602
     GLCM_contrast_mean    0.002596
               B_hist_7    0.002552
               ExR_mean    0.002298
               B_hist_4    0.002213
               R_hist_3    0.002163
                ExG_max    0.002089
               B_hist_5    0.002066
               R_hist_0    0.001997
           HSV_S_hist_0    0.001991
GLCM_dissimilarity_mean    0.001952
             ExG_hist_1    0.001861
               B_hist_3    0.001851
               G_median    0.001849
               B_median    0.001848
               R_hist_8    0.001769
              HSV_V_std    0.001756
               B_hist_2    0.001724
              CIVE_mean    0.001689
           HSV_V_hist_4    0.001588
             ExR_hist_4    0.001587
                ExG_min    0.001569
               R_hist_9    0.001551
               NDI_mean    0.001512
          GLCM_ASM_mean    0.001511
             ExR_hist_6    0.001495
           HSV_H_hist_3    0.001439
           HSV_V_hist_6    0.001392
                  B_std    0.001327
      GLCM_contrast_std    0.001318
           HSV_S_hist_5    0.001270
           HSV_S_hist_4    0.001202
             ExR_hist_0    0.001157
             ExG_hist_6    0.001106
                 R_mean    0.001004
             ExR_hist_1    0.000814
             HSV_V_mean    0.000744
             ExR_hist_3    0.000732
             ExR_hist_5    0.000471
             ExR_hist_2    0.000220

Model training completed!

Train takes: 298.77 seconds
Loading test data...
Number of test images: 1
Extracting features...
Processing: 0/1
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_47/4137158527.py in <cell line: 0>()
    410 
    411     # Predict test set
--> 412     submission = predictor.predict('/kaggle/input/csiro-biomass/test.csv', image_dir='/kaggle/input/csiro-biomass/test')
    413 
    414     # Save submission file

/tmp/ipykernel_47/4137158527.py in predict(self, test_csv_path, image_dir)
    357 
    358         # Extract features for each image (only once)
--> 359         X_test = self.prepare_features(unique_images, is_training=False)
    360 
    361         # Predict for each target

/tmp/ipykernel_47/4137158527.py in prepare_features(self, df, is_training)
    242             self.feature_names = feature_df.columns.tolist()
    243 
--> 244         return feature_df[self.feature_names].values
    245 
    246     def train(self, train_csv_path, image_dir='/kaggle/input/csiro-biomass/train'):

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4106             if is_iterator(key):
   4107                 key = list(key)
-> 4108             indexer = self.columns._get_indexer_strict(key, "columns")[1]
   4109 
   4110         # take() does not accept boolean indexers

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _get_indexer_strict(self, key, axis_name)
   6198             keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
   6199 
-> 6200         self._raise_if_missing(keyarr, indexer, axis_name)
   6201 
   6202         keyarr = self.take(indexer)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _raise_if_missing(self, key, indexer, axis_name)
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())
-> 6252             raise KeyError(f"{not_found} not in index")
   6253 
   6254     @overload

KeyError: "['State', 'Species', 'Month', 'DayOfYear'] not in index"
'''