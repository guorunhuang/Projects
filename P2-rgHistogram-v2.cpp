/*
功能:扫描指定文件夹中的所有 JPG 图像，计算每个图像的 16×16 R-G 通道归一化直方图，并将「图像路径 + 直方图数据」保存到特征文件中

Parameters to Pass运行时需传入 2 个位置参数，顺序固定：
参数顺序	参数含义	示例值	说明
第 1 个	图像文件夹路径	./images	存放所有待处理 JPG 图像的文件夹路径
第 2 个	输出特征文件路径	./feature_rgHistogram.txt	保存特征数据的文本文件（自动生成）

# 运行命令
feature_extractor.exe ./images ./feature_rgHistogram.txt

程序结束时生成的文件：feature_rgHistogram.txt（特征文件），格式如下：
每一行对应一个 JPG 图像的特征；
每行第 1 列是图像的标准化路径（如 ./images/pic.0164.jpg）；
每行第 2 列及以后是 256 个归一化直方图数据（16×16 个 bin 的值，用逗号分隔）
示例文件内容（简化版）：
./images/pic.0164.jpg,0.0021,0.0035,0.0012,...,0.0043
./images/pic.0080.jpg,0.0018,0.0029,0.0031,...,0.0027
./images/pic.1032.jpg,0.0042,0.0015,0.0023,...,0.0039
*/

#include <opencv2/opencv.hpp>
#include <filesystem>  // 替代glob.h，跨平台文件系统操作
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

namespace fs = std::filesystem;  // 简化文件系统操作的命名空间

// Computes 2D RGB histogram (R and G channels) - MANUAL IMPLEMENTATION
// 计算2D RGB直方图（R和G通道）- 手动实现版本
cv::Mat computeRGHistogram(const cv::Mat& image, int bins = 16) {
    if (image.empty() || image.channels() != 3) {
        return cv::Mat();
    }

    // Convert to RGB color space (OpenCV reads images as BGR by default)
    // 转换为RGB颜色空间（OpenCV默认以BGR格式读取图像）
    cv::Mat rgb;
    cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);

    // Create 2D histogram matrix initialized to zero
    // 创建2D直方图矩阵，初始化为0
    cv::Mat hist = cv::Mat::zeros(bins, bins, CV_32F);
    
    // Calculate bin size for mapping pixel values [0-255] to bins [0-bins-1]
    // 计算bin大小，用于将像素值[0-255]映射到bin索引[0-bins-1]
    float binSize = 256.0f / bins;
    
    // Manually compute histogram by iterating through all pixels
    // 手动遍历所有像素计算直方图
    for (int y = 0; y < rgb.rows; y++) {
        for (int x = 0; x < rgb.cols; x++) {
            // Get RGB values at current pixel
            // 获取当前像素的RGB值
            cv::Vec3b pixel = rgb.at<cv::Vec3b>(y, x);
            int r_value = pixel[0];  // R channel / R通道
            int g_value = pixel[1];  // G channel / G通道
            
            // Calculate which bin this pixel belongs to
            // 计算该像素属于哪个bin
            int r_bin = static_cast<int>(r_value / binSize);
            int g_bin = static_cast<int>(g_value / binSize);
            
            // Handle edge case: value 255 should go to bin (bins-1), not bins
            // 处理边界情况：值255应该放在bin(bins-1)中，而不是bins
            if (r_bin >= bins) r_bin = bins - 1;
            if (g_bin >= bins) g_bin = bins - 1;
            
            // Increment the corresponding histogram bin
            // 增加对应直方图bin的计数
            hist.at<float>(r_bin, g_bin) += 1.0f;
        }
    }
    
    // Normalize histogram manually (L1 normalization)
    // 手动归一化直方图（L1归一化）
    float sum = 0.0f;
    for (int r = 0; r < bins; r++) {
        for (int g = 0; g < bins; g++) {
            sum += hist.at<float>(r, g);
        }
    }
    
    // Divide each bin by sum to normalize
    // 将每个bin除以总和进行归一化
    if (sum > 0) {
        for (int r = 0; r < bins; r++) {
            for (int g = 0; g < bins; g++) {
                hist.at<float>(r, g) /= sum;
            }
        }
    }
    
    return hist;
}

// Finds all JPG files in a directory using std::filesystem
// 使用std::filesystem查找目录中所有JPG文件（跨平台兼容）
std::vector<std::string> findJPGFiles(const std::string& directory) {
    std::vector<std::string> jpgFiles;

    // Check if the directory exists
    // 检查目录是否存在
    if (!fs::exists(directory) || !fs::is_directory(directory)) {
        std::cerr << "Directory does not exist or is not a directory: " << directory << std::endl;
        // 目录不存在或不是一个有效目录: 
        return jpgFiles;
    }

    // Iterate through all entries in the directory
    // 遍历目录中所有条目
    for (const auto& entry : fs::directory_iterator(directory)) {
        // Check if it's a regular file (not a directory)
        // 检查是否为普通文件（非目录）
        if (entry.is_regular_file()) {
            std::string filename = entry.path().filename().string();
            // Convert filename to lowercase to handle .JPG and .jpg
            // 转换文件名到小写，以同时处理.JPG和.jpg后缀
            std::transform(filename.begin(), filename.end(), filename.begin(), ::tolower);
            
            // Check if the file has a .jpg extension
            // 检查文件是否有.jpg后缀
            if (filename.size() >= 4 && filename.substr(filename.size() - 4) == ".jpg") {
                jpgFiles.push_back(entry.path().string());
            }
        }
    }

    return jpgFiles;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <image directory> <output feature file>" << std::endl;
        // 用法:  <图像目录> <输出特征文件>
        return -1;
    }

    std::string imageDir = argv[1];
    std::string outputFile = argv[2];
    const int bins = 16;

    // Find all JPG files in the directory using std::filesystem
    // 使用std::filesystem查找目录中所有JPG文件
    std::vector<std::string> imageFiles = findJPGFiles(imageDir);
    if (imageFiles.empty()) {
        std::cerr << "No JPG files found in directory: " << imageDir << std::endl;
        // 目录中未找到JPG文件: 
        return -1;
    }

    std::cout << "Found " << imageFiles.size() << " JPG files. Processing..." << std::endl;
    // 找到 个JPG文件。正在处理...

    // Open output file for writing features
    // 打开输出文件用于写入特征
    std::ofstream outFile(outputFile);
    if (!outFile.is_open()) {
        std::cerr << "Failed to open output file: " << outputFile << std::endl;
        // 无法打开输出文件: 
        return -1;
    }

    // Process each image
    // 处理每个图像
    for (const std::string& filePath : imageFiles) {
        // Read image
        // 读取图像
        cv::Mat image = cv::imread(filePath);
        if (image.empty()) {
            std::cerr << "Failed to read image: " << filePath << std::endl;
            // 无法读取图像: 
            continue;
        }

        // Compute histogram
        // 计算直方图
        cv::Mat hist = computeRGHistogram(image, bins);
        if (hist.empty()) {
            std::cerr << "Failed to compute histogram for image: " << filePath << std::endl;
            // 无法计算图像的直方图: 
            continue;
        }

        // Write image path and histogram to file
        // 将图像路径和直方图写入文件
        outFile << filePath;
        
        for (int r = 0; r < bins; ++r) {
            for (int g = 0; g < bins; ++g) {
                outFile << "," << hist.at<float>(r, g);
            }
        }
        
        outFile << std::endl;
    }

    outFile.close();
    std::cout << "Feature extraction completed. Results saved to: " << outputFile << std::endl;
    // 特征提取完成。结果保存至: 

    return 0;
}