#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include "filter.h"
#include "faceDetect.h"

using namespace cv;
using namespace std;

int main() {
    // 打开默认摄像头（索引0）通常是笔记本内置摄像头
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cout << "Error: Could not open the camera." << endl;
        return -1;
    }

    // 创建窗口
    namedWindow("Real-Time Video", WINDOW_AUTOSIZE);

    // 状态变量：记录当前滤镜模式
    // 定义一个名为 FilterMode 的枚举类型，它包含 5 个滤镜模式
    enum FilterMode { NORMAL, GRAY, CUSTOM_GRAY, BROWN, BLUR,
                SOBEL_X, SOBEL_Y, MAGNITUDE, BLUR_QUANTIZE, FACE_DETECT, EMBOSS,
                RETAIN_RED };
    FilterMode current_mode = NORMAL;
    // const std::string CASCADE_PATH = "haarcascade_frontalface_default.xml"; // 替换为实际路径

    Mat frame, processed_frame;
    int save_count = 0; // 保存图像的计数（避免文件名重复）
    // 新增Sobel结果存储变量（16SC3类型，用于梯度幅度计算）
    cv::Mat sobel_x_result, sobel_y_result; 
    float contrast_alpha = 1.0f; 
    int brightness_beta = 0;

    cout << "Available keys:\n"
            << "  q - Quit\n"
            << "  s - Save frame\n"
            << "  g - Toggle OpenCV Gray\n"
            << "  h - Toggle Custom Gray\n"
            << "  p - Toggle Brown Filter\n"
            << "  b - Toggle Blur\n"
            << "  f - Toggle Face Detection\n";

    while (true) {
        // 捕获一帧
        cap >> frame;
        if (frame.empty()) {
            cout << "Error: Could not capture frame." << endl;
            break;
        }

        // 根据当前模式处理帧
        switch (current_mode) {
            case NORMAL:
                frame.copyTo(processed_frame); // 原始彩色
                break;
            case GRAY:
                cvtColor(frame, processed_frame, COLOR_BGR2GRAY); // OpenCV灰度
                break;
            case CUSTOM_GRAY:
                customGray(frame, processed_frame); // 自定义灰度（filter.cpp）
                break;
            case BROWN:
                brownFilter(frame, processed_frame); // 棕褐色调（filter.cpp）
                break;
            case BLUR:
                blur5x5_v2(frame, processed_frame); // 优化版5×5模糊（filter.cpp）
                break;
            case SOBEL_X:
                sobelX3x3(frame, sobel_x_result); // 计算Sobel X（16SC3）
                cv::convertScaleAbs(sobel_x_result, processed_frame); // 转换为8UC3用于显示
                break;
            case SOBEL_Y:
                sobelY3x3(frame, sobel_y_result); // 计算Sobel Y（16SC3）
                cv::convertScaleAbs(sobel_y_result, processed_frame); // 转换为8UC3
                break;
            case MAGNITUDE:
                // 先计算Sobel X和Y，再生成梯度幅度
                sobelX3x3(frame, sobel_x_result);
                sobelY3x3(frame, sobel_y_result);
                magnitude(sobel_x_result, sobel_y_result, processed_frame); // 直接输出8UC3
                break;
            case BLUR_QUANTIZE:
                blurQuantize(frame, processed_frame, 10); // 量化级别默认10
                break;
            case FACE_DETECT: {
                    cv::Mat gray;
                    cvtColor(frame, gray, COLOR_BGR2GRAY);
    
                    vector<Rect> faces;
                    detectFaces(gray, faces);      // 调用 faceDetect.cpp 里的函数
                    frame.copyTo(processed_frame);
                    drawBoxes(processed_frame, faces);  // 画框
                    break;
                }
            case EMBOSS:
                embossEffect(frame, processed_frame);
                break;
            case RETAIN_RED:
                retainRedColor(frame, processed_frame, 120, 50); // 使用默认阈值
                break;
            default:
                frame.copyTo(processed_frame);
                break;
        }

        // 显示处理后的帧
        if (contrast_alpha != 1.0f || brightness_beta != 0) {
            cv::Mat tmp;
            adjustBrightnessContrast(processed_frame, tmp, contrast_alpha, brightness_beta);
            processed_frame = tmp;
        }
        imshow("Real-Time Video", processed_frame);

        // 处理按键（等待30ms，避免卡顿）
        char key = waitKey(30);
        switch (key) {
            case 'q': // 退出程序
                cout << "Exiting program..." << endl;
                goto exit_loop; // 跳出循环
            case 's': // 保存当前帧
            {
                string filename = "captured_frame_" + to_string(++save_count) + ".png"; // 前置自增,所以文件命名从captured_frame_1开始而不是0
                imwrite(filename, processed_frame); // 保存处理后带滤镜的帧（如果要原始则传入frame）
                cout << "Frame saved as: " << filename << endl; // imwrite会将图像保存到当前工作目录，也就是程序运行时所在的文件夹
                break;
            }
            case 'g': // 切换：彩色 ↔ OpenCV灰度
                current_mode = (current_mode == GRAY) ? NORMAL : GRAY;
                cout << "Switched to " << (current_mode == GRAY ? "OpenCV Gray" : "Normal Color") << endl;
                break;
            case 'h': // 切换：彩色 ↔ 自定义灰度
                current_mode = (current_mode == CUSTOM_GRAY) ? NORMAL : CUSTOM_GRAY;
                cout << "Switched to " << (current_mode == CUSTOM_GRAY ? "Custom Gray" : "Normal Color") << endl;
                break;
            case 'p': // 切换：彩色 ↔ 棕褐色调
                current_mode = (current_mode == BROWN) ? NORMAL : BROWN;
                cout << "Switched to " << (current_mode == BROWN ? "Brown Filter" : "Normal Color") << endl;
                break;
            case 'b': // 切换：彩色 ↔ 5×5模糊
                current_mode = (current_mode == BLUR) ? NORMAL : BLUR;
                cout << "Switched to " << (current_mode == BLUR ? "5x5 Blur" : "Normal Color") << endl;
                break;
            case 'x': // 切换Sobel X
                current_mode = (current_mode == SOBEL_X) ? NORMAL : SOBEL_X;
                cout << "Switched to " << (current_mode == SOBEL_X ? "Sobel X" : "Normal Color") << endl;
                break;
            case 'y': // 切换Sobel Y
                current_mode = (current_mode == SOBEL_Y) ? NORMAL : SOBEL_Y;
                cout << "Switched to " << (current_mode == SOBEL_Y ? "Sobel Y" : "Normal Color") << endl;
                break;
            case 'm': // 切换梯度幅度
                current_mode = (current_mode == MAGNITUDE) ? NORMAL : MAGNITUDE;
                cout << "Switched to " << (current_mode == MAGNITUDE ? "Gradient Magnitude" : "Normal Color") << endl;
                break;
            case 'l': // 切换模糊量化（任务9）
                current_mode = (current_mode == BLUR_QUANTIZE) ? NORMAL : BLUR_QUANTIZE;
                cout << "Switched to " << (current_mode == BLUR_QUANTIZE ? "Blur Quantize" : "Normal Color") << endl;
                break;
            case 'f': // 检测人脸
                current_mode = (current_mode == FACE_DETECT) ? NORMAL : FACE_DETECT;
                cout << "Switched to " << (current_mode == FACE_DETECT ? "Face Detection" : "Normal Color") << endl;
                break;
            case 'e': // 浮雕效果
                current_mode = (current_mode == EMBOSS) ? NORMAL : EMBOSS;
                cout << "Switched to " << (current_mode == EMBOSS ? "Emboss" : "Normal Color") << endl;
                break;
            case '+' : // 增加亮度（每次+10）
                brightness_beta = std::min(100, brightness_beta + 10);
                std::cout << "Brightness: " << brightness_beta << std::endl;
                break;
            case '-' : // 减少亮度（每次-10）
                brightness_beta = std::max(-100, brightness_beta - 10);
                std::cout << "Brightness: " << brightness_beta << std::endl;
                break;
            case '=' : // 增加对比度（每次+0.1）
                contrast_alpha = std::min(3.0f, contrast_alpha + 0.1f);
                std::cout << "Contrast (alpha): " << contrast_alpha << std::endl;
                break;
            case '_' : // 减少对比度（每次-0.1）
                contrast_alpha = std::max(0.1f, contrast_alpha - 0.1f);
                std::cout << "Contrast (alpha): " << contrast_alpha << std::endl;
                break;
            case 'c' : // 重置亮度对比度
                contrast_alpha = 1.0f;
                brightness_beta = 0;
                std::cout << "Brightness/Contrast reset to default." << std::endl;
                break;
            case 'r': // 切换“保留红色”效果
                current_mode = (current_mode == RETAIN_RED) ? NORMAL : RETAIN_RED;
                std::cout << "Switched to " << (current_mode == RETAIN_RED ? "Retain Red Color" : "Normal Color") << std::endl;
                break;            
        }
    }

exit_loop: //配合前面的 goto exit_loop
    // 释放资源
    cap.release();
    destroyAllWindows();
    return 0;
}