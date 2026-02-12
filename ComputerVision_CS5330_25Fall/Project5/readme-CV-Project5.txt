README 
======

Name: Guorun Huang

// Project 5 Summary

In this project, I built and trained a convolutional neural network for MNIST digit recognition using PyTorch, including convolution, pooling, dropout, and fully connected layers. The trained model was saved and later reused for evaluation. I analyzed the first convolutional layer by visualizing its learned filters and applying them to sample images using OpenCV’s filter2D, confirming that the filters captured meaningful edge and stroke patterns.
I then performed transfer learning on Greek letters and examined misclassification behavior by relating errors to the extracted filter responses. Finally, I designed a series of controlled experiments, exploring variations in network depth, filter sizes, dropout rates, activations, and optimizers. I also replaced the first layer with handcrafted filters—Sobel, Laplacian, Gaussian, and Gabor—and retrained only the remaining layers. Gaussian produced the strongest results, with 0° Gabor performing second best, demonstrating how different low-level feature extractors influence network performance.

// Handwritten letters url
https://drive.google.com/drive/folders/16_ZpK8AYvdqIXvZbYMKHkmh87pnBLF9S?usp=drive_link

// Time-off / Time Travel Days

NA

// Environment

Primary tested environment: Windows(virtual environment), VS Code, OpenCV 4.

// Programs & File formats

Compile and run example
run .py in Windows cmd


