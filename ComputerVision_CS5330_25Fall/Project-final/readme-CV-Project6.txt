README 
======

Name: Guorun Huang

// Project 6

In this project, I explored multiple data sources and modeling approaches to predict pasture biomass for the CSIRO dataset. The work covered data preparation, feature engineering, remote‑sensing integration, and the development of five different models.

1. Data Preparation
I first processed the original image dataset, which contains ground‑level photos of pasture with varying grass density, color, and condition. From these images, I extracted a wide range of hand‑crafted features, including:

Color indices (ExG, ExR, VARI, CIVE, NDI) to distinguish green and dry grass

RGB/HSV histograms and hue‑based color distributions

Texture features using GLCM/Haralick metrics

Green‑coverage estimation as a global vegetation indicator

Metadata such as height, state, and species was also incorporated.

2. Sentinel‑2 Remote Sensing Data
To enrich the information beyond ground‑level images, I integrated Sentinel‑2 satellite data. Since no paddock boundaries were provided, I approximated locations using state‑level coordinates with controlled variation. I used DEA’s Analysis Ready Data (Level‑2A), applied cloud masking, resampled bands to a uniform 20 m resolution, and extracted the median reflectance for each band within the selected region.

This added a second, independent view of vegetation conditions, captured from space and including infrared information.

3. Model Development
I built five models to compare different data sources and modeling strategies:

ResNet (image‑only) Raw images were fed into a ResNet model, and the learned embeddings were used to predict the five biomass targets.

XGBoost (hand‑crafted features + metadata) All engineered features and metadata were combined into a single vector for each image and used to train an XGBoost model.

MLP (hand‑crafted features + metadata) The same feature set was passed into a multilayer perceptron to evaluate performance differences between tree‑based and neural models.

MLP with Sentinel‑2 (S2 median bands + hand‑crafted features + metadata) Sentinel‑2 reflectance values were added to the feature set and used as input to an MLP.

Fusion Model (ResNet + XGBoost) This model combined deep image embeddings from ResNet with the structured‑feature predictions from XGBoost.

4. Feature Importance and Model Insights
Using XGBoost’s feature importance scores, I evaluated which engineered features contributed most to prediction accuracy. This helped identify redundant or low‑impact features and guided feature selection for later models.

5. Overall Contribution
This project demonstrates a full pipeline for biomass prediction:

Ground‑image processing

Remote‑sensing data extraction

Multi‑modal feature engineering

Five modeling strategies, including deep learning and fusion models

It also shows how combining close‑range imagery with satellite data can provide a richer understanding of pasture conditions.


// Environment

Kaggle Notebook

// Programs & File formats

sentinel2_dea_kaggle-NSW.csv

Date	Latitude	Longitude	nbart_blue	nbart_green	nbart_red	nbart_red_edge_1	nbart_red_edge_2	nbart_red_edge_3	nbart_nir_1	nbart_nir_2	nbart_swir_2	nbart_swir_3
2015/7/31	-31.2	145.9	362	568	758	1123	1651	1740	1803	1910	2410	1693
2015/8/10	-31.2	145.9	338	578	808	1164	1696	1812	1874	1981	2547	1819
2015/8/20	-31.2	145.9	372	594	856	1214	1740	1856	1927	2024	2603	1908
2015/8/30	-31.2	145.9	392	627	890	1272	1824	1942	2027	2111	2665	1954
2015/9/9	-31.2	145.9	430	654	937	1305	1840	1990	2067	2161	2747	2015
2015/9/19	-31.2	145.9	442	664	975	1320	1782	1963	2038	2131	2767	2032
2015/9/29	-31.2	145.9	464	686	1055	1361	1764	1981	2051	2154	2867	2111
2015/11/18	-31.2	145.9	432	658	1078	1349	1709	1974	2057	2145	2971	2260
2015/11/28	-31.2	145.9	463	708	1162	1444	1796	2060	2156	2234	3092	2371
2015/12/28	-31.2	145.9	434	660	1104	1375	1700	1957	2057	2141	3011	2255



