README 
======

Name: Guorun Huang

// Project Summary

This project implements a real-time shape-based object recognition system in C++ with OpenCV. Adaptive HSV/gray hybrid thresholding is used for robust object/background separation under varying illumination, followed by connected-component region filtering. A feature vector is computed for each detected region, including centroid, area, orientation from central moments, oriented minimum bounding box. The system supports interactive one-shot learning — pressing ‘n’ stores the current object’s feature vector with a user-provided label into a lightweight text-based database.
For customized extension experiments, the system is evaluated on objects such as scissors, money, sunglasses, pens, phone camera modules, and cards. It performs well on objects with distinctive geometric signatures, but struggles on color/texture-dominated items (e.g. fruits, boxes, cables). A one-shot ResNet18 embedding was also tested.
Overall, the results indicate that segmentation quality and region filtering thresholds are critical, and that shape-based descriptors are highly effective for structurally distinctive objects, while more general categories may require color/texture or deep embeddings to achieve stable recognition.

// Time-off / Time Travel Days

I am using 1 time travel day. (1 day)

// Environment

Primary tested environment: Windows, g++, msys64, VS Code, OpenCV 4.

// Programs & File formats

Compile and run example
$ g++ -std=c++17 xxx.cpp -o xxx.exe `pkg-config --cflags --libs opencv4`


