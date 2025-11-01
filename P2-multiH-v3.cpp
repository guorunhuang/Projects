/* 
功能: 对每张图像计算未归一化的 8-bin RGB 直方图（每通道 8 个 bin），构成全图（24 值）与中心区域（24 值）
的拼接向量后对整个向量做 L1 归一化，输出文件名 + 48 个浮点特征（总和约为 1.0）。
Compute raw 8-bin per-channel RGB histograms (3*8=24) for the full image and for a center ROI (center 25%×25%), 
concatenate them into a 48-length vector, then L1-normalize the whole vector so the sum ≈ 1.0. Save each image's filename and the 48 normalized floats as a CSV row.
颜色直方图+梯度幅值直方图

Parameters to Pass 运行时需传入 2 个位置参数，顺序固定： 参数顺序 参数含义 示例值 
第 1 个 图像文件夹路径 ./images 第 2 个 输出 CSV 文件路径 ./features_multiH.csv

Usage: ./P2-multiH-v3 <image_folder> <output_csv> 
Example: ./P2-multiH-v3 ./images features.csv

实现与生成文件示意
直方图计算：对每个通道使用 8 个 bin，范围 [0,256)。computeRGBHistRaw 返回按通道拼接的原始 bin 值（长度 24），此处不对单通道做归一化。
中心区域：中心 ROI 大小为图像宽高的 1/4（cw = w/4, ch = h/4），并做边界夹紧以保证合法矩形。
特征拼接与归一化：
拼接顺序为：full-image(24) + center-ROI(24) -> 48。
对整个 48 维向量做 L1 归一化（所有绝对值之和 = 1.0，若和为 0 则保持原向量除以 1 防止除 0）。
输出 CSV 行格式： filename,f0,f1,...,f47 其中 filename 为文件名（不含路径），每个 f 是归一化后的浮点数，字段用逗号分隔。
*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <vector>
using namespace std;
using namespace cv;
namespace fs = std::filesystem;

// Extract single RGB histogram with 8 bins per channel, then return concatenated vector
// IMPORTANT: returns concatenated vector (3*8 = 24) and DOES NOT normalize per-channel.
// We'll normalize the whole combined vector (or combined full+center) later.
vector<float> computeRGBHistRaw(const Mat& img) {
    vector<Mat> channels;
    split(img, channels);
    int histSize = 8;
    float range[] = {0, 256};
    const float* histRange = {range};
    vector<float> feature;
    feature.reserve(3 * histSize);
    for (int i = 0; i < 3; ++i) {
        Mat hist;
        calcHist(&channels[i], 1, 0, Mat(), hist, 1, &histSize, &histRange);
        // push raw bins (not normalized here)
        for (int j = 0; j < histSize; ++j) feature.push_back(hist.at<float>(j));
    }
    return feature; // length = 24
}

// compute combined feature: full-image hist (24) + center hist (24) -> 48
// then L1-normalize the entire concatenated vector so sum = 1.0
vector<float> computeCombinedFeature(const Mat& img) {
    vector<float> fullHist = computeRGBHistRaw(img);

    // center ROI - we use a center rectangle sized 25% x 25% of image (as before)
    int w = img.cols, h = img.rows;
    int cw = max(1, w / 4), ch = max(1, h / 4);
    int sx = (w - cw) / 2, sy = (h - ch) / 2;
    // clamp to image
    if (sx < 0) sx = 0;
    if (sy < 0) sy = 0;
    if (sx + cw > w) cw = w - sx;
    if (sy + ch > h) ch = h - sy;
    Rect centerRect(sx, sy, cw, ch);
    Mat centerROI = img(centerRect);
    vector<float> centerHist = computeRGBHistRaw(centerROI);

    // concatenate
    vector<float> combined;
    combined.reserve(fullHist.size() + centerHist.size());
    combined.insert(combined.end(), fullHist.begin(), fullHist.end());
    combined.insert(combined.end(), centerHist.begin(), centerHist.end());

    // L1 normalize the whole combined vector so total sum == 1.0
    double s = 0.0;
    for (float v : combined) s += fabs(v);
    if (s <= 0) s = 1.0; // avoid div0
    for (float &v : combined) v = float(v / s);

    return combined; // length 48, sum ~= 1.0
}

int main(int argc, char** argv) {
    if (argc < 3) {
        cout << " Usage: ./P2-multiH-v3 <image_folder> <output_csv>\n";
        return -1;
    }

    string folder = argv[1];
    string output_csv = argv[2];
    ofstream fout(output_csv);
    if (!fout.is_open()) {
        cerr << " Failed to open output file.\n";
        return -1;
    }

    for (const auto& entry : fs::directory_iterator(folder)) {
        string path = entry.path().string();
        Mat img = imread(path);
        if (img.empty()) continue;

        vector<float> feat = computeCombinedFeature(img);

        fout << fs::path(path).filename().string();
        for (float v : feat) fout << "," << v;
        fout << "\n";
        cout << "Processed: " << path << endl;
    }

    fout.close();
    cout << "Feature CSV saved to " << output_csv << endl;
    return 0;
}