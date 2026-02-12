#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

// g++ imgDisplay.cpp -o imgDisplay `pkg-config --cflags --libs opencv4`
// ./imgDisplay lena.jpg
// 那么：argc == 2, argv[0] == "./imgDisplay"（程序名）, argv[1] == "lena.jpg"（你传入的图像路径）


int main(int argc, char** argv) {
    // 检查参数（需传入图像路径）
    if (argc != 2) {
        cout << "Usage: ./imgDisplay <image_path>" << endl;
        return -1;
    }

    // 读取图像（彩色模式）
    Mat img = imread(argv[1], IMREAD_COLOR);
    if (img.empty()) {
        cout << "Error: image empty." << endl;
        return -1;
    }

    // 创建窗口并显示图像
    namedWindow("Image Display", WINDOW_AUTOSIZE);
    imshow("Image Display", img);

    // 等待按键：按'q'退出
    while (true) {
        char key = waitKey(30); // 动态图像展示
        if (key == 'q') { 
            break;
        }
    }

    // 释放资源
    destroyAllWindows();
    return 0;
}