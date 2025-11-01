#include "filter.h"
#include <cmath>

// 任务4：自定义灰度转换（三通道平均值 → 255 - 平均值）
int customGray(const cv::Mat& src, cv::Mat& dst) {
    if (src.empty() || src.channels() != 3) return -1; // 仅处理3通道彩色图，否则 cv::Vec3b 访问会出错。
    // CV_8U：每个像素是 8位无符号整数（uchar）C1：表示 1个通道（灰度图）
    dst.create(src.size(), CV_8UC1); // 创建单通道灰度图输出

    for (int i = 0; i < src.rows; i++) {
        // cv::Vec3b 是 OpenCV 的一个结构，表示一个包含 3 个 uchar 的向量：[B, G, R]
        // uchar 是 unsigned char，无符号的字符，占用 1 个字节（8 位），数值范围是 0 到 255
        // char 是有符号的，范围是 -128 到 127，不适合表示颜色值.用 uchar 可以节省内存（1 字节 vs 4 字节的 int）
        const cv::Vec3b* src_row = src.ptr<cv::Vec3b>(i); // 行首指针（BGR通道）
              uchar* dst_row = dst.ptr<uchar>(i);              // 灰度图行首指针
        for (int j = 0; j < src.cols; j++) {
            // 计算三通道平均值
            // 指针定在某一行然后按列遍历，本质上是：指针 + 下标访问 = 指针偏移
            // ptr[j] ≡ *(ptr + j)
            uchar b = src_row[j][0];
            uchar g = src_row[j][1];
            uchar r = src_row[j][2];
            uchar avg = (b + g + r) / 3;
            dst_row[j] = 255 - avg; // 取反
        }
    }
    return 0;
}

// 任务5：棕褐色调滤镜（含渐晕效果）
int brownFilter(const cv::Mat& src, cv::Mat& dst) {
    if (src.empty() || src.channels() != 3) return -1;
    dst.create(src.size(), src.type()); // 保持与输入相同的类型（3通道）

    // 渐晕效果参数：中心到边缘的衰减系数（0~1）
    int center_col = src.cols / 2;
    int center_row = src.rows / 2;
    double max_dist = sqrt(center_row*center_row + center_col*center_col); // 中心到角落的最大距离

    for (int i = 0; i < src.rows; i++) {
        const cv::Vec3b* src_row = src.ptr<cv::Vec3b>(i);
              cv::Vec3b* dst_row = dst.ptr<cv::Vec3b>(i);
        
        // 计算当前行到中心的y距离（用于渐晕）
        double dist_row = abs(i - center_row);
        for (int j = 0; j < src.cols; j++) {
            // 1. 计算原始BGR值
            uchar b = src_row[j][0];
            uchar g = src_row[j][1];
            uchar r = src_row[j][2];

            // 2. 棕褐色通道计算（使用原始BGR值）
            // 必须用double，因为这里系数是double（小数）
            double new_b = 0.272 * r + 0.534 * g + 0.131 * b;
            double new_g = 0.349 * r + 0.686 * g + 0.168 * b;
            double new_r = 0.393 * r + 0.769 * g + 0.189 * b;

            // 3. 数值范围校验（截断到0~255）
            new_b = std::min(255.0, std::max(0.0, new_b));
            new_g = std::min(255.0, std::max(0.0, new_g));
            new_r = std::min(255.0, std::max(0.0, new_r));

            // 4. 渐晕效果：计算当前像素到中心的距离，生成衰减系数
            double dist = sqrt(pow(j - center_col, 2) + pow(dist_row, 2));
            // dist小，则接近中心，不衰减。dist大，则在边缘，需要衰减
            double fadeFactor = 1.0 - (dist / max_dist) * 0.5; // 边缘最多衰减50%
            fadeFactor = std::min(1.0, std::max(0.0, fadeFactor)); // 确保系数合法

            // 5. 应用渐晕并赋值
            dst_row[j][0] = static_cast<uchar>(new_b * fadeFactor);
            dst_row[j][1] = static_cast<uchar>(new_g * fadeFactor);
            dst_row[j][2] = static_cast<uchar>(new_r * fadeFactor);
        }
    }
    return 0;
}

// 任务6.1：基础版5×5模糊（朴素实现，高斯核整数近似）
int blur5x5_v1(const cv::Mat& src, cv::Mat& dst) {
    if (src.empty() || (src.channels() != 1 && src.channels() != 3)) return -1;
    src.copyTo(dst); // 先复制原图，外层2行2列保持不变

    // 5×5高斯核（整数近似）
    // Python：使用 中括号 [] 表示列表（数组）类型和维度可以动态变化，不需要提前声明
    const int kernel[5][5] = {
        {1, 2, 4, 2, 1}
        ,{2, 4, 8, 4, 2}
        ,{4, 8, 16, 8, 4}
        ,{2, 4, 8, 4, 2}
        ,{1, 2, 4, 2, 1}
    };
    const int kernel_sum = 100; // 核元素总和（用于归一化）

    // 遍历除外层2行2列的区域
    for (int i = 2; i < src.rows - 2; i++) {
        for (int j = 2; j < src.cols - 2; j++) {
            if (src.channels() == 3) { // 彩色图（BGR通道分别处理）
                int sum_b = 0, sum_g = 0, sum_r = 0;

                // 遍历5×5核
                for (int ky = -2; ky <= 2; ky++) {
                    for (int kx = -2; kx <= 2; kx++) {
                        int weight = kernel[ky + 2][kx + 2]; // 核索引转换（0~4）
                        cv::Vec3b pixel = src.at<cv::Vec3b>(i + ky, j + kx); // 获取像素值
                        sum_b += pixel[0] * weight;
                        sum_g += pixel[1] * weight;
                        sum_r += pixel[2] * weight;
                    }
                }
                // 归一化并赋值
                dst.at<cv::Vec3b>(i, j)[0] = sum_b / kernel_sum;
                dst.at<cv::Vec3b>(i, j)[1] = sum_g / kernel_sum;
                dst.at<cv::Vec3b>(i, j)[2] = sum_r / kernel_sum;

            } else { // 灰度图
                int sum_gray = 0;
                for (int ky = -2; ky <= 2; ky++) {
                    for (int kx = -2; kx <= 2; kx++) {
                        int weight = kernel[ky + 2][kx + 2];
                        sum_gray += src.at<uchar>(i + ky, j + kx) * weight;
                    }
                }
                dst.at<uchar>(i, j) = sum_gray / kernel_sum;
            }
        }
    }
    return 0;
}


// at	src.at<cv::Vec3b>(i, j)	安全、易读、自动边界检查
// ptr	src.ptr<cv::Vec3b>(i)[j]	快速、直接访问内存、无边界检查


// 任务6.2：优化版5×5模糊（可分离滤波：先垂直1×5，再水平1×5）
int blur5x5_v2(const cv::Mat& src, cv::Mat& dst) {
    if (src.empty() || src.channels() != 3) return -1; // 仅处理彩色图
    cv::Mat temp;
    src.copyTo(temp); // 临时图像（存储垂直滤波结果）
    src.copyTo(dst);  // 最终输出图像（存储水平滤波结果）

    // 可分离核（1×5，垂直/水平通用，总和20）
    const int kernel_1d[5] = {1, 2, 4, 2, 1};
    const int kernel_sum = 10; // 1×5核总和（用于归一化）

    // 第一步：垂直滤波（对每列应用1×5核，遍历除外层2行）
    for (int i = 2; i < src.rows - 2; i++) {
        cv::Vec3b* temp_row = temp.ptr<cv::Vec3b>(i);
        for (int j = 0; j < src.cols; j++) {
            int sum_b = 0, sum_g = 0, sum_r = 0;

            // 垂直方向遍历5个像素（y-2到y+2）
            for (int ky = -2; ky <= 2; ky++) {
                int weight = kernel_1d[ky + 2];
                cv::Vec3b pixel = src.ptr<cv::Vec3b>(i + ky)[j];
                sum_b += pixel[0] * weight;
                sum_g += pixel[1] * weight;
                sum_r += pixel[2] * weight;
            }

            // 归一化并赋值到临时图像
            temp_row[j][0] = sum_b / kernel_sum;
            temp_row[j][1] = sum_g / kernel_sum;
            temp_row[j][2] = sum_r / kernel_sum;
        }
    }

    // 第二步：水平滤波（对每行应用1×5核，遍历除外层2列）
    for (int i = 0; i < src.rows; i++) {
              cv::Vec3b* dst_row = dst.ptr<cv::Vec3b>(i);
        const cv::Vec3b* temp_row = temp.ptr<cv::Vec3b>(i);
        for (int j = 2; j < src.cols - 2; j++) {
            int sum_b = 0, sum_g = 0, sum_r = 0;
            // 水平方向遍历5个像素（x-2到x+2）
            for (int kx = -2; kx <= 2; kx++) {
                int weight = kernel_1d[kx + 2];
                cv::Vec3b pixel = temp_row[j + kx];
                sum_b += pixel[0] * weight;
                sum_g += pixel[1] * weight;
                sum_r += pixel[2] * weight;
            }
            // 归一化并赋值到最终输出
            dst_row[j][0] = sum_b / kernel_sum;
            dst_row[j][1] = sum_g / kernel_sum;
            dst_row[j][2] = sum_r / kernel_sum;
        }
    }

    return 0;
}

// 任务7：3×3 Sobel X / Sobel Y 滤镜（可分离实现）
// Sobel X 3x3（水平方向，向右为正）：可分离为 [1,2,1]（垂直核）和 [-1,0,1]（水平核）
int sobelX3x3(const cv::Mat& src, cv::Mat& dst) {
    if (src.empty() || src.channels() != 3) return -1;
    // 输出为 3通道带符号短整型（16SC3）：每个通道是16位short 类型，避免溢出
    // Sobel 运算后：梯度值可能是 负数（例如 -300）也可能是 大于 255 的正数（例如 +400）
    // 如果直接输出为 CV_8U：所有负数会被截断为 0，所有超过 255 的值会被截断为 255，导致信息丢失和边缘失真
    dst.create(src.size(), CV_16SC3);

    // 步骤1：应用垂直1×3核 [1, 2, 1]（先平滑垂直方向）
    cv::Mat temp;
    temp.create(src.size(), CV_16SC3); // 临时存储垂直滤波结果

    for (int i = 1; i < src.rows - 1; i++) { // 外层1行不处理
        const cv::Vec3b* src_row_prev = src.ptr<cv::Vec3b>(i - 1);
        const cv::Vec3b* src_row_curr = src.ptr<cv::Vec3b>(i);
        const cv::Vec3b* src_row_next = src.ptr<cv::Vec3b>(i + 1);
        cv::Vec3s* temp_row = temp.ptr<cv::Vec3s>(i);

        for (int j = 0; j < src.cols; j++) {
            // 垂直方向加权求和（BGR三通道分别处理）
            int b = src_row_prev[j][0] * 1 + src_row_curr[j][0] * 2 + src_row_next[j][0] * 1;
            int g = src_row_prev[j][1] * 1 + src_row_curr[j][1] * 2 + src_row_next[j][1] * 1;
            int r = src_row_prev[j][2] * 1 + src_row_curr[j][2] * 2 + src_row_next[j][2] * 1;
            // 存储的时候，把32位int改成16位short更节省空间
            temp_row[j] = cv::Vec3s(static_cast<short>(b), static_cast<short>(g), static_cast<short>(r));
        }
    }

    // 步骤2：应用水平1×3核 [-1, 0, 1]（检测水平边缘，向右为正）
    for (int i = 0; i < src.rows; i++) {
        const cv::Vec3s* temp_row = temp.ptr<cv::Vec3s>(i);
              cv::Vec3s* dst_row = dst.ptr<cv::Vec3s>(i);

        for (int j = 1; j < src.cols - 1; j++) { // 外层1列不处理
            // 水平方向加权差分（使用垂直滤波后的结果）
            int b = temp_row[j-1][0] * (-1) + temp_row[j][0] * 0 + temp_row[j+1][0] * 1;
            int g = temp_row[j-1][1] * (-1) + temp_row[j][1] * 0 + temp_row[j+1][1] * 1;
            int r = temp_row[j-1][2] * (-1) + temp_row[j][2] * 0 + temp_row[j+1][2] * 1;
            dst_row[j] = cv::Vec3s(static_cast<short>(b), static_cast<short>(g), static_cast<short>(r));
        }
    }

    // 边缘填充（外层行/列设为0，简化处理）
    dst.row(0).setTo(cv::Vec3s(0, 0, 0));
    dst.row(dst.rows - 1).setTo(cv::Vec3s(0, 0, 0));
    dst.col(0).setTo(cv::Vec3s(0, 0, 0));
    dst.col(dst.cols - 1).setTo(cv::Vec3s(0, 0, 0));

    return 0;
}

// Sobel Y 3x3（垂直方向，向上为正）：可分离为 [-1,0,1]（垂直核）和 [1,2,1]（水平核）
int sobelY3x3(const cv::Mat& src, cv::Mat& dst) {
    if (src.empty() || src.channels() != 3) return -1;
    dst.create(src.size(), CV_16SC3); // 输出为16SC3类型

    // 步骤1：应用水平1×3核 [1, 2, 1]（先平滑水平方向）
    cv::Mat temp;
    temp.create(src.size(), CV_16SC3);

    for (int i = 0; i < src.rows; i++) {
        const cv::Vec3b* src_row = src.ptr<cv::Vec3b>(i);
        cv::Vec3s* temp_row = temp.ptr<cv::Vec3s>(i);

        for (int j = 1; j < src.cols - 1; j++) { // 外层1列不处理
            // 水平方向加权求和
            int b = src_row[j-1][0] * 1 + src_row[j][0] * 2 + src_row[j+1][0] * 1;
            int g = src_row[j-1][1] * 1 + src_row[j][1] * 2 + src_row[j+1][1] * 1;
            int r = src_row[j-1][2] * 1 + src_row[j][2] * 2 + src_row[j+1][2] * 1;
            temp_row[j] = cv::Vec3s(static_cast<short>(b), static_cast<short>(g), static_cast<short>(r));
        }
    }

    // 步骤2：应用垂直1×3核 [-1, 0, 1]（检测垂直边缘，向上为正）
    for (int i = 1; i < src.rows - 1; i++) { // 外层1行不处理
        const cv::Vec3s* temp_row_prev = temp.ptr<cv::Vec3s>(i - 1);
        const cv::Vec3s* temp_row_curr = temp.ptr<cv::Vec3s>(i);
        const cv::Vec3s* temp_row_next = temp.ptr<cv::Vec3s>(i + 1);
        cv::Vec3s* dst_row = dst.ptr<cv::Vec3s>(i);

        for (int j = 0; j < src.cols; j++) {
            // 垂直方向加权差分（向上为正：上一行 - 下一行）
            int b = temp_row_prev[j][0] * 1 + temp_row_curr[j][0] * 0 + temp_row_next[j][0] * (-1);
            int g = temp_row_prev[j][1] * 1 + temp_row_curr[j][1] * 0 + temp_row_next[j][1] * (-1);
            int r = temp_row_prev[j][2] * 1 + temp_row_curr[j][2] * 0 + temp_row_next[j][2] * (-1);
            dst_row[j] = cv::Vec3s(static_cast<short>(b), static_cast<short>(g), static_cast<short>(r));
        }
    }

    // 边缘填充（外层行/列设为0）
    dst.row(0).setTo(cv::Vec3s(0, 0, 0));
    dst.row(dst.rows - 1).setTo(cv::Vec3s(0, 0, 0));
    dst.col(0).setTo(cv::Vec3s(0, 0, 0));
    dst.col(dst.cols - 1).setTo(cv::Vec3s(0, 0, 0));

    return 0;
}

// 任务8：梯度幅度图像生成
// 将两个方向的梯度图（CV_16SC3）合并为一个梯度幅度图（CV_8UC3），用于显示图像边缘强度。
int magnitude(const cv::Mat& sx, const cv::Mat& sy, cv::Mat& dst) {
    // 检查输入是否为16SC3类型且尺寸一致
    if (sx.empty() || sy.empty() || sx.type() != CV_16SC3 || sy.type() != CV_16SC3 ||
        sx.size() != sy.size() || sx.channels() != 3) {
        return -1;
    }

    dst.create(sx.size(), CV_8UC3); // 输出为8位无符号彩色图（适合显示）

    for (int y = 0; y < sx.rows; y++) {
        const cv::Vec3s* sx_row = sx.ptr<cv::Vec3s>(y);
        const cv::Vec3s* sy_row = sy.ptr<cv::Vec3s>(y);
              cv::Vec3b* dst_row = dst.ptr<cv::Vec3b>(y);

        for (int x = 0; x < sx.cols; x++) {
            // 对每个通道分别计算梯度幅度：sqrt(sx² + sy²)
            for (int c = 0; c < 3; c++) { // c=0(B), 1(G), 2(R)
                short sx_val = sx_row[x][c];
                short sy_val = sy_row[x][c];
                // 计算欧几里得距离，结果截断到0~255
                double mag = sqrt(static_cast<double>(sx_val * sx_val) + static_cast<double>(sy_val * sy_val));
                dst_row[x][c] = static_cast<uchar>(std::min(255.0, std::max(0.0, mag)));
            }
        }
    }

    return 0;
}

// 任务9：模糊量化函数，将0-255数值映射成10个离散桶对应的10个值
int blurQuantize(const cv::Mat& src, cv::Mat& dst, int levels) {
    if (src.empty() || src.channels() != 3 || levels < 1 || levels > 255) return -1;

    // 步骤1：先对输入图像进行模糊处理（调用之前实现的优化版5×5模糊）
    cv::Mat blurred;
    blur5x5_v2(src, blurred); // 使用blur5x5_2减少细节，为量化做准备

    // 步骤2：计算量化桶大小（255/levels，确保桶大小为整数）
    int bucket_size = 255 / levels;
    if (bucket_size == 0) bucket_size = 1; // 避免levels=255时桶大小为0

    dst.create(blurred.size(), CV_8UC3); // 输出为8位彩色图

    // 对每个像素的每个通道进行量化
    for (int y = 0; y < blurred.rows; y++) {
        const cv::Vec3b* blurred_row = blurred.ptr<cv::Vec3b>(y);
        cv::Vec3b* dst_row = dst.ptr<cv::Vec3b>(y);

        for (int x = 0; x < blurred.cols; x++) {
            for (int c = 0; c < 3; c++) { // 遍历B、G、R通道
                uchar pixel_val = blurred_row[x][c];
                // 量化逻辑：x → xt = x//bucket_size → xf = xt * bucket_size
                int quant_level = pixel_val / bucket_size;
                uchar quant_val = static_cast<uchar>(quant_level * bucket_size);
                dst_row[x][c] = quant_val;
            }
        }
    }

    return 0;
}

// 任务12.1：浮雕效果：(SobelX * cosθ + SobelY * sinθ) + 128，θ=45°（cosθ=sinθ≈0.7071）
int embossEffect(const cv::Mat& src, cv::Mat& dst) {
    if (src.empty() || src.channels() != 3) return -1;
    dst.create(src.size(), CV_8UC3);

    // 1. 计算Sobel X和Y（16SC3类型）
    cv::Mat sx, sy;
    sobelX3x3(src, sx);
    sobelY3x3(src, sy);

    // 2. 计算浮雕值：(sx * 0.7071 + sy * 0.7071) + 128（偏移到0~255）
    const float cos_theta = 0.7071f;
    const float sin_theta = 0.7071f;

    for (int y = 0; y < src.rows; y++) {
        const cv::Vec3s* sx_row = sx.ptr<cv::Vec3s>(y);
        const cv::Vec3s* sy_row = sy.ptr<cv::Vec3s>(y);
        cv::Vec3b* dst_row = dst.ptr<cv::Vec3b>(y);

        for (int x = 0; x < src.cols; x++) {
            for (int c = 0; c < 3; c++) {
                float emboss_val = sx_row[x][c] * cos_theta + sy_row[x][c] * sin_theta;
                emboss_val += 128.0f; // 偏移到0~255范围
                // 截断到合法范围
                dst_row[x][c] = static_cast<uchar>(std::min(255.0f, std::max(0.0f, emboss_val)));
            }
        }
    }

    return 0;
}

// 任务12.2：亮度/对比度调整：dst = alpha * src + beta（alpha>1增强对比度，beta>0增强亮度）
int adjustBrightnessContrast(const cv::Mat& src, cv::Mat& dst, float alpha, int beta) {
    if (src.empty()) return -1;
    src.convertTo(dst, -1, alpha, beta); // OpenCV convertTo函数实现线性变换
    return 0;
}

// 任务12.3：强色保留：保留指定颜色（此处为红色），其他区域转为灰度
// 逻辑：判断像素是否为“强红色”（R通道值远大于G、B通道，且超过阈值），是则保留原彩色，否则转为灰度
int retainRedColor(const cv::Mat& src, cv::Mat& dst, int red_threshold, int diff_threshold) {
    if (src.empty() || src.channels() != 3) return -1;
    dst.create(src.size(), src.type()); // 输出为彩色图像

    // 先创建全图的灰度版本（用于非红色区域）
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(gray, gray, cv::COLOR_GRAY2BGR); // 转为3通道灰度图（与src格式一致）

    for (int y = 0; y < src.rows; y++) {
        const cv::Vec3b* src_row = src.ptr<cv::Vec3b>(y); // 原彩色图像行指针
        const cv::Vec3b* gray_row = gray.ptr<cv::Vec3b>(y); // 灰度图像行指针
        cv::Vec3b* dst_row = dst.ptr<cv::Vec3b>(y); // 输出图像行指针

        for (int x = 0; x < src.cols; x++) {
            // 提取当前像素的B、G、R通道值（OpenCV默认BGR顺序）
            uchar b = src_row[x][0];
            uchar g = src_row[x][1];
            uchar r = src_row[x][2];

            // 判断是否为“强红色”：R > 阈值，且R与G、B的差值均大于差异阈值
            bool is_strong_red = (r > red_threshold) && 
                                 (r - g > diff_threshold) && 
                                 (r - b > diff_threshold);

            // 强红色区域保留原彩色，否则使用灰度图像素
            if (is_strong_red) {
                dst_row[x] = src_row[x];
            } else {
                dst_row[x] = gray_row[x];
            }
        }
    }

    return 0;
}