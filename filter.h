//if not define, then define the following func
#ifndef FILTER_H
#define FILTER_H

#include <opencv2/opencv.hpp>

// & 表示引用传参，不会复制整个图像，提高效率。
// const 表示这个参数在函数内部不能被修改，保护原始图像
// 在 C++ 中，函数可以返回任何类型（比如 int, bool, void, std::string, 结构体、类、指针、引用等）
// 返回值可能表示执行结果，比如 0 表示成功，其他值表示错误或状态码

// 任务4：自定义灰度转换（三通道平均值取反）
int customGray(const cv::Mat& src, cv::Mat& dst);

// 任务5：棕褐色调滤镜（含渐晕效果）
int brownFilter(const cv::Mat& src, cv::Mat& dst);

// 任务6.1：基础版5×5模糊（朴素实现）
int blur5x5_v1(const cv::Mat& src, cv::Mat& dst);

// 任务6.2：优化版5×5模糊（可分离滤波）
int blur5x5_v2(const cv::Mat& src, cv::Mat& dst);

// 任务7：Sobel X/Y 滤镜（3×3可分离）
int sobelX3x3(const cv::Mat& src, cv::Mat& dst);
int sobelY3x3(const cv::Mat& src, cv::Mat& dst);

// 任务8：梯度幅度图像生成
int magnitude(const cv::Mat& sx, const cv::Mat& sy, cv::Mat& dst);

// 任务9：模糊量化函数
int blurQuantize(const cv::Mat& src, cv::Mat& dst, int levels = 10);

// 任务12.1：浮雕效果：(SobelX * cosθ + SobelY * sinθ) + 128，θ=45°（cosθ=sinθ≈0.7071）
int embossEffect(const cv::Mat& src, cv::Mat& dst);

// 任务12.2：亮度/对比度调整：dst = alpha * src + beta（alpha>1增强对比度，beta>0增强亮度）
int adjustBrightnessContrast(const cv::Mat& src, cv::Mat& dst, float alpha = 1.0f, int beta = 0);

// 任务12.3：强色保留：保留指定颜色（此处为红色），其他区域转为灰度
int retainRedColor(const cv::Mat& src, cv::Mat& dst, int red_threshold = 120, int diff_threshold = 50);

#endif