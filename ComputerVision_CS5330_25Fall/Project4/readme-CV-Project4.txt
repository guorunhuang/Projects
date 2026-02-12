README 
======

Name: Guorun Huang

// Project 4 Summary

In this project, I implemented a complete camera calibration and augmented reality pipeline using OpenCV. 
The system first detects and extracts chessboard corners from video frames, allowing the user to select specific frames for calibration. The selected 2D corner positions are stored along with their corresponding 3D world coordinates, enabling computation of the camera matrix, distortion coefficients, and pose (rotation and translation) using solvePnP. The pose results behaved as expected: moving the chessboard sideways changed the X translation, moving it forward or backward affected Z, and tilting it resulted in rotation changes. Because the camera was fixed and the chessboard was handheld and slightly curved, minor inaccuracies occurred and distortion coefficients increased due to modeling assumptions being violated. 
Using the estimated pose, I projected 3D axes and later created a virtual object—a tree and a rotating windmill—anchored to the chessboard in real time. I also experimented with robust feature detection using Harris, GFTT, FAST, ORB, and SIFT, and observed that adjusting thresholds and limiting feature count greatly improved detection quality. 
Finally, I explored visual extensions, concluding that texture replacement on the detected board produces the best visual results compared to other attempted effects.

// Time-off / Time Travel Days

NA

// Environment

Primary tested environment: Windows, g++, msys64, VS Code, OpenCV 4.

// Programs & File formats

Compile and run example
$ g++ -std=c++17 xxx.cpp -o xxx.exe `pkg-config --cflags --libs opencv4`


