#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // 创建一个 480x640 的 8-bit 3通道图像，所有像素设为蓝色 (B,G,R) = (255,0,0)
    cv::Mat img(480, 640, CV_8UC3, cv::Scalar(255, 0, 0));

    // 显示窗口
    cv::imshow("Blue Image", img);

    // 等待用户按键（0 表示无限等待）
    int k = cv::waitKey(0);

    // 打印按键码并退出
    std::cout << "Key pressed: " << k << std::endl;
    return 0;
}

// 终于看见蓝色图片了，cpp runs successfully!!!