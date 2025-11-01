/*
功能: 遍历指定文件夹的图像，提取每张图像的 RGB 颜色直方图（每通道 8 bins，共 24 值，L1 归一化）和
 Sobel 梯度幅值的纹理直方图（16 bins，L1 归一化），将文件名与拼接后的特征向量以 CSV 行写出。

Brief English description: For each image in a folder, compute an L1-normalized 8-bin per-channel RGB histogram (24 floats)
 and an L1-normalized 16-bin Sobel magnitude histogram (16 floats), concatenate them (total 40 floats) and save as CSV rows: filename,f0,f1,...,f39.

Parameters to Pass 运行时需传入 2 个位置参数，顺序固定： 参数顺序 参数含义 示例值 
第 1 个 图像文件夹路径 ./images 第 2 个 输出 CSV 文件路径 ./features_color_texture.csv

Usage: ./build_features_color_texture <image_folder> <output_csv> 
Example: ./build_features_color_texture ./olympus features.csv
*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <filesystem>
using namespace std;
using namespace cv;
namespace fs = std::filesystem;

// ===== 计算颜色直方图 (RGB, 8 bins each) =====
// Compute color histogram with 8 bins per channel
vector<float> computeColorHist(const Mat& img) {
    vector<Mat> channels;
    split(img, channels);

    int histSize = 8;
    float range[] = {0, 256};
    const float* histRange = {range};

    vector<float> feature;
    for (int i = 0; i < 3; ++i) {
        Mat hist;
        calcHist(&channels[i], 1, 0, Mat(), hist, 1, &histSize, &histRange);
        normalize(hist, hist, 1, 0, NORM_L1);
        for (int j = 0; j < histSize; ++j)
            feature.push_back(hist.at<float>(j));
    }
    return feature;
}

// ===== 计算纹理特征 (Sobel 梯度幅值直方图) =====
// Compute texture histogram using Sobel gradient magnitude
vector<float> computeTextureHist(const Mat& imgGray) {
    Mat gx, gy;
    Sobel(imgGray, gx, CV_32F, 1, 0, 3);
    Sobel(imgGray, gy, CV_32F, 0, 1, 3);

    Mat mag;
    magnitude(gx, gy, mag);

    int histSize = 16;
    float range[] = {0, 256};
    const float* histRange = {range};

    Mat hist;
    calcHist(&mag, 1, 0, Mat(), hist, 1, &histSize, &histRange);
    normalize(hist, hist, 1, 0, NORM_L1);

    vector<float> feature;
    for (int i = 0; i < histSize; ++i)
        feature.push_back(hist.at<float>(i));

    return feature;
}

// ===== 主程序：遍历目录提取特征 =====
int main(int argc, char** argv) {
    if (argc < 3) {
        cout << "Usage: ./build_features_color_texture <image_folder> <output_csv>\n";
        return -1;
    }

    string folder = argv[1];
    string output_csv = argv[2];
    ofstream fout(output_csv);
    if (!fout.is_open()) {
        cerr << "Failed to open output file.\n";
        return -1;
    }

    for (const auto& entry : fs::directory_iterator(folder)) {
        string path = entry.path().string();
        Mat img = imread(path);
        if (img.empty()) continue;

        Mat gray;
        cvtColor(img, gray, COLOR_BGR2GRAY);

        vector<float> colorFeat = computeColorHist(img);
        vector<float> textureFeat = computeTextureHist(gray);

        // 合并特征 / concatenate color and texture
        vector<float> feat;
        feat.insert(feat.end(), colorFeat.begin(), colorFeat.end());
        feat.insert(feat.end(), textureFeat.begin(), textureFeat.end());

        fout << fs::path(path).filename().string();
        for (float v : feat) fout << "," << v;
        fout << "\n";
        cout << "Processed: " << path << endl;
    }

    fout.close();
    cout << " Feature CSV saved to " << output_csv << endl;
    return 0;
}