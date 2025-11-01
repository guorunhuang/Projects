/*
功能: 遍历指定文件夹里的图像，提取每张图像中心 7×7 区域的 B,G,R 值（ B,G,R 顺序、行优先），并将“图像绝对路径 + 147 个整数特征”写入。

Parameters to Pass 运行时需传入 2 个位置参数，顺序固定： 
参数顺序 参数含义 示例值 第 1 个 图像文件夹路径 ./images 第 2 个 输出特征文件路径 ./features_center7x7.txt

Usage: ./build_features <image_directory> <output_feature_file>
Example: ./build_features ./images features.txt

生成文件示意与说明
每行对应一张成功读取的图像；
行格式：<absolute_path>\t v0 v1 v2 ... v146
路径为绝对路径，特征为 147 个 0–255 整数（7×7×3），通道顺序为 B G R，值间以空格分隔，路径与特征间用 tab 分隔。
*/

// cv::Vec3b pix = patch.at<cv::Vec3b>(y, x);为什么不用指针方式访问，.at<>() 明确表示访问的是 (y,x) 位置的像素，类型安全
// 用指针访问时需要手动计算偏移量：ptr[x*3 + c]，不如 Vec3b 直观。对于小 patch 7×7，性能差异可以忽略

#include <opencv2/opencv.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

namespace fs = std::filesystem;

// Extract center 7x7 patch feature as vector<int> (B,G,R per pixel, row-major)
std::vector<int> extract_center7x7(const cv::Mat& input_img) {
    cv::Mat img = input_img;
    // Ensure 3-channel BGR
    if (img.channels() == 4) cv::cvtColor(img, img, cv::COLOR_BGRA2BGR);
    if (img.channels() == 1) cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);

    int h = img.rows, w = img.cols;
    // If too small, resize up to at least 7x7
    if (h < 7 || w < 7) {
        int new_h = std::max(7, h);
        int new_w = std::max(7, w);
        cv::Mat resized;
        cv::resize(img, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
        img = resized;
        h = img.rows; w = img.cols;
    }

    int cx = w / 2;
    int cy = h / 2;
    int sx = cx - 3;
    int sy = cy - 3;
    if (sx < 0) sx = 0;
    if (sy < 0) sy = 0;
    if (sx + 7 > w) sx = w - 7;
    if (sy + 7 > h) sy = h - 7;

    cv::Rect roi(sx, sy, 7, 7);
    cv::Mat patch = img(roi);

    std::vector<int> feat;
    feat.reserve(7 * 7 * 3);
    for (int y = 0; y < 7; ++y) {
        for (int x = 0; x < 7; ++x) {
            cv::Vec3b pix = patch.at<cv::Vec3b>(y, x);
            // B, G, R order (OpenCV default). Keep consistent across programs.
            feat.push_back(static_cast<int>(pix[0]));
            feat.push_back(static_cast<int>(pix[1]));
            feat.push_back(static_cast<int>(pix[2]));
        }
    }
    return feat;
}

bool has_image_ext(const std::string& ext) {
    std::string e = ext;
    std::transform(e.begin(), e.end(), e.begin(), ::tolower);
    return e==".jpg" || e==".jpeg" || e==".png" || e==".bmp" || e==".tif" || e==".tiff";
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: ./build_features <image_directory> <output_feature_file>\n";
        return 1;
    }
    std::string dir = argv[1];
    std::string out_file = argv[2];

    std::ofstream ofs(out_file);
    if (!ofs.is_open()) {
        std::cerr << "Failed to open output file: " << out_file << "\n";
        return 1;
    }

    int processed = 0;
    for (const auto &entry : fs::directory_iterator(dir)) {
        if (!entry.is_regular_file()) continue;
        std::string path = entry.path().string();
        if (!has_image_ext(entry.path().extension().string())) continue;

        cv::Mat img = cv::imread(path, cv::IMREAD_UNCHANGED);
        if (img.empty()) {
            std::cerr << "Warning: cannot read image: " << path << "\n";
            continue;
        }
        std::vector<int> feat = extract_center7x7(img);
        // Write line: <absolute_path>\tv0 v1 v2 ... v146\n
        std::string abs_path = fs::absolute(entry.path()).string();
        ofs << abs_path << '\t';
        for (size_t i = 0; i < feat.size(); ++i) {
            if (i) ofs << ' ';
            ofs << feat[i];
        }
        ofs << '\n';
        ++processed;
    }

    ofs.close();
    std::cout << "Processed " << processed << " images. Features saved to " << out_file << "\n";
    return 0;
}
