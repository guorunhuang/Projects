/*
  Bruce A. Maxwell
  Spring 2024
  CS 5330 Computer Vision

  Include file for faceDetect.cpp, face detection and drawing functions
*/
#ifndef FACEDETECT_H
#define FACEDETECT_H

// put the path to the haar cascade file here
#define FACE_CASCADE_FILE "./haarcascade_frontalface_alt2.xml"

#include <opencv2/opencv.hpp>

// prototypes
// 传入的灰度图，输出检测到的人脸矩形（cv::Rect）
int detectFaces( cv::Mat &grey, std::vector<cv::Rect> &faces );
// minWidth (默认 50):过滤太小检测框的阈值（像素），避免误检
// scale (默认 1.0): 如果需要把检测坐标按某个比例缩放到显示图像上（例如检测在缩放后的图像上运行，绘制时放回原始尺寸）
int drawBoxes( cv::Mat &frame, std::vector<cv::Rect> &faces, int minWidth = 50, float scale = 1.0  );

#endif

// 级联（cascade） 指多个分类器按“层级”串联，早期快速舍弃多数负样本，只把疑似区域交给更复杂的分类器继续判断
// 目标检测时，大部分候选窗口是背景。级联思想是先用非常快、计算量小的检测器快速剔除大部分窗口；
// 留下的少量候选再由更复杂的检测器精判。这样总体既快又能保持较好准确率

// OpenCV 把训练好的检测器参数以 XML（或 YAML）文件保存，一种用文本描述结构化数据的格式（标签式），可读可编辑。

// Haar 特征：一类简单的矩形亮度差分特征，用若干个白/黑矩形区域之和差来衡量局部亮暗结构（例如眼窝比额头暗、鼻梁比两侧亮）
// Boosting / AdaBoost：一种把很多“弱分类器”（性能略优于随机）组合成一个“强分类器”的集成学习算法。通过反复训练弱分类器并调整样本权重，最终得到加权投票的强分类器

// 宏（macro）是由预处理器在编译前进行文本替换的“符号”或“模板”
// #define PI 3.14159
// #define SQR(x) ((x)*(x))
// PI 在源码中出现的每处都会被替换成 3.14159。
// SQR(x) 是一个带参数的宏，SQR(a+1) 会被展开为 ((a+1)*(a+1))
// #define FACE_CASCADE_FILE "./haarcascade_frontalface_alt2.xml"
// 就是一个字符串宏（路径宏）。含义：在源代码中每次出现 FACE_CASCADE_FILE，预处理器会把它替换成 "./haarcascade_frontalface_alt2.xml"。
// 典型用途是把外部资源（如模型文件、配置文件）的路径以宏形式集中定义，方便修改和复用。