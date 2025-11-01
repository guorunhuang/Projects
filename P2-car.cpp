/*
功能: 遍历指定文件夹中的图像，为每张图像提取 HOG 描述子和 Laplacian 方差纹理值，
并将结果以 CSV 格式保存（每行：filename, texture, h0,h1,...）。
For each image, compute a HOG feature vector (using a 128×128 window) and a texture score (Laplacian standard deviation), 
then write filename, texture, and all HOG float values into car_features.csv.

参数与运行（代码中为硬编码，可修改变量）：
imgDir (string): 图像文件夹路径，示例 "./olympus"
输出文件: car_features.csv（主程序中已写死）

Usage (as implemented):
直接执行程序（main 中 imgDir 与输出文件已硬编码）

实现细节与生成文件说明
HOG 设置（HOGDescriptor 构造参数）:
winSize = 128×128, blockSize = 16×16, blockStride = 8×8, cellSize = 8×8, nbins = 9
输入图像先转为灰度并 resize 到 128×128，再调用 hog.compute。
返回的 descriptors 被封装为 1×N 的 Mat（行向量），N 为 HOG 特征长度（程序假定为 3780 并在 CSV 表头中预留该列）。
纹理特征:
计算灰度图的 Laplacian（CV_64F），对结果做 meanStdDev，使用 sigma[0]（标准差）作为纹理强度/锐利度指标。
输出 CSV 格式（car_features.csv）:
首行为表头: filename,texture,h0,h1,...,h3780
每行对应一张图像: filename,texture, then HOG floats comma-separated
filename 使用 entry.path().filename().string()（仅文件名，不含路径）
*/

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <fstream>
#include <iostream>
#include <filesystem>
using namespace cv;
using namespace std;
namespace fs = std::filesystem;

// HOG 特征
Mat computeHOG(const Mat& img) {
    HOGDescriptor hog(
        Size(128, 128), // winSize
        Size(16, 16),   // blockSize
        Size(8, 8),     // blockStride
        Size(8, 8),     // cellSize
        9               // nbins
    );
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    resize(gray, gray, Size(128, 128));
    vector<float> descriptors;
    hog.compute(gray, descriptors);
    return Mat(descriptors).t(); // 1 x N
}

// Laplacian 方差作为纹理特征
double computeTexture(const Mat& img) {
    Mat gray, lap;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    Laplacian(gray, lap, CV_64F);
    Scalar mu, sigma;
    meanStdDev(lap, mu, sigma);
    return sigma[0];
}

int main() {
    string imgDir = "./olympus";  // 汽车图片文件夹
    ofstream fout("car_features.csv");
    fout << "filename,texture";
    for (int i = 0; i < 3780; i++) fout << ",h" << i;
    fout << endl;

    for (const auto& entry : fs::directory_iterator(imgDir)) {
        Mat img = imread(entry.path().string());
        if (img.empty()) continue;

        Mat hogFeat = computeHOG(img);
        double texture = computeTexture(img);

        fout << entry.path().filename().string() << "," << texture;
        for (int i = 0; i < hogFeat.cols; i++)
            fout << "," << hogFeat.at<float>(0, i);
        fout << endl;
    }

    fout.close();
    cout << "Car features saved to car_features.csv" << endl;
    return 0;
}