README 
======

Name: Guorun Huang

//Project Summary

This repository implements a simple content-based image retrieval system in C++ / OpenCV. The system extracts and compares visual features (color histograms, Sobel texture histograms, HOG, Laplacian variance, and ResNet embeddings) to retrieve visually similar images. Two main workflows are provided:
Car-shape retrieval (shape-focused): HOG + Laplacian variance, match by weighted Euclidean distance.
Banana (object) retrieval: color histogram + Sobel texture + ResNet embeddings, match by weighted combination of histogram-intersection, SSD and cosine distance.

//Time-off / Time Travel Days

I am using 2 time travel days. (2 days)

//Environment

Primary tested environment: Windows, g++, msys64, VS Code, OpenCV 4.

//Programs & File formats

Compile and run example
$ g++ -std=c++17 P2-texture-v3.cpp -o P2-texture-v3.exe `pkg-config --cflags --libs opencv4`
$ g++ -std=c++17 P2-texture-match-v3.cpp -o P2-texture-match-v3.exe `pkg-config --cflags --libs opencv4`
$ ./P2-texture-v3 ./olympus P2_textures.txt
$ ./P2-texture-match-v3 ./P2_textures.txt pic.0948.jpg ./olympus

