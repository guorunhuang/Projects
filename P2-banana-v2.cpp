/*
功能: 遍历指定图像文件夹，提取每张图像的 HSV 颜色直方图（8×8×8 = 512 bins）、Sobel 梯度幅值纹理直方图（256 bins），
并从预先生成的 DNN CSV 中读取对应图像的深度特征，最终将 filename_with_path + color + texture + DNN 特征按 CSV 行写入输出文件（默认 banana_features.csv）。
For each image, compute an L1-normalized 8×8×8 HSV color histogram (512 floats), an L1-normalized 256-bin Sobel magnitude histogram (256 floats), 
read the precomputed DNN feature vector by matching filename (no path) from a DNN CSV, and write filename_with_path followed by all features as a single CSV row.

参数与运行（代码中为硬编码，可根据需要修改变量）：
imgDir (string): 图像文件夹路径，示例 "./olympus"
dnnCsv (string): 预先计算的 DNN 特征 CSV，示例 "./ResNet18_olym.csv"

Usage (as implemented):直接运行程序（main 内硬编码路径）
若需要其它输入/输出，请在代码中修改 imgDir、dnnCsv 或输出文件名

生成文件示意与说明
输出文件: banana_features.csv（可在代码中修改）
每行对应一张成功读取且在 DNN CSV 中能找到 DNN 特征的图像；
行格式：<absolute_or_relative_path>, c0,c1,...,c511, t0,t1,...,t255, d0,d1,...,dM
colorHist: HSV 3D 直方图重塑为 512 个浮点数（范围由 H:[0,180), S:[0,256), V:[0,256)），已用 NORM_L1 归一化；
textureHist: Sobel magnitude 的 256-bin 直方图，已用 NORM_L1 归一化；
dnnFeat: 从 dnnCsv 中按行匹配 filename（仅文件名，不含路径）读取的预计算 DNN 特征（长度由 CSV 决定）；
字段以逗号分隔，行尾换行。
*/

#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <numeric>
using namespace std;
using namespace cv;
namespace fs = std::filesystem;

// Function to compute color histogram (8 bins per channel)
vector<float> computeColorHist(const Mat& img) {
    Mat hsv;
    cvtColor(img, hsv, COLOR_BGR2HSV);
    int histSize[] = {8, 8, 8};
    float h_ranges[] = {0, 180}, s_ranges[] = {0, 256}, v_ranges[] = {0, 256};
    const float* ranges[] = {h_ranges, s_ranges, v_ranges};
    int channels[] = {0, 1, 2};
    Mat hist;
    calcHist(&hsv, 1, channels, Mat(), hist, 3, histSize, ranges, true, false);
    normalize(hist, hist, 1.0, 0.0, NORM_L1); // Use NORM_L1 for better comparison
    return hist.reshape(1,1);
}

// Function to compute texture histogram (Sobel magnitude)
vector<float> computeTextureHist(const Mat& img) {
    Mat gray, grad_x, grad_y, mag;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    Sobel(gray, grad_x, CV_32F, 1, 0);
    Sobel(gray, grad_y, CV_32F, 0, 1);
    magnitude(grad_x, grad_y, mag);
    int histSize = 256;
    float range[] = {0, 255};
    const float* histRange = {range};
    Mat hist;
    calcHist(&mag, 1, 0, Mat(), hist, 1, &histSize, &histRange);
    normalize(hist, hist, 1.0, 0.0, NORM_L1); // Use NORM_L1 for better comparison
    return hist.reshape(1,1);
}

// Read DNN features (precomputed)
vector<float> readDNNFeature(const string& dnnFile, const string& filename) {
    ifstream file(dnnFile);
    string line;
    vector<float> features;
    while (getline(file, line)) {
        stringstream ss(line);
        string name;
        getline(ss, name, ',');
        // Correctly match the filename without the path
        if (name == filename) {
            float val;
            while (ss >> val) {
                features.push_back(val);
                if (ss.peek() == ',') ss.ignore();
            }
            break;
        }
    }
    return features;
}

int main() {
    string imgDir = "./olympus";
    string dnnCsv = "./ResNet18_olym.csv";
    ofstream out("banana_features.csv");

    for (const auto& entry : fs::directory_iterator(imgDir)) {
        string filename_with_path = entry.path().string();
        string filename_only = entry.path().filename().string();
        Mat img = imread(filename_with_path);
        if (img.empty()) continue;

        vector<float> colorHist = computeColorHist(img);
        vector<float> textureHist = computeTextureHist(img);
        vector<float> dnnFeat = readDNNFeature(dnnCsv, filename_only);

        // Check if DNN feature was found
        if (dnnFeat.empty()) {
            cerr << "Warning: DNN feature for " << filename_only << " not found. Skipping." << endl;
            continue;
        }

        // Use the original filename with path for consistency in the output CSV
        out << filename_with_path;
        for (float v : colorHist) out << "," << v;
        for (float v : textureHist) out << "," << v;
        for (float v : dnnFeat) out << "," << v;
        out << "\n";
    }
    cout << " Feature extraction done." << endl;
}