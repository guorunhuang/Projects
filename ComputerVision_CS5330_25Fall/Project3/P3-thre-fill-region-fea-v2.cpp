/*
调整后dynamicThreshold变成使用HSV后，尝试了剪刀、眼镜，虽然能被识别出来，
但是画出来的bounding box还是偏小，只画了一半，或者只识别了剪刀柄
又改成HSV和灰度同时使用之后，青色方形识别不出来了

然后又改了morphologicalCleanup，这时候剪刀识别比较好

还改了threshold的参数，放宽了一些

minArea = 500效果不好容易忽略小的物体
*/
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <map>

using namespace cv;
using namespace std;

struct RegionFeatures {
    int regionId;
    Point2d centroid;
    double area;
    Rect boundingBox;
    double angle;
    double percentFilled;
    double aspectRatio;
    double huMoments[7];
    Vec3b color;
};

Mat preprocessImage(const Mat& src) {
    Mat blurred;
    GaussianBlur(src, blurred, Size(5, 5), 1.0);
    return blurred;
}

// Mat dynamicThreshold(const Mat& src) {
//     Mat hsv;
//     cvtColor(src, hsv, COLOR_BGR2HSV);

//     vector<Mat> chans;
//     split(hsv, chans);
//     Mat s = chans[1];  // Saturation 通道最适合区分“彩色 vs 白背景”

//     // Threshold on Saturation (彩色牙膏的饱和度 > 白背景极低饱和度)
//     Mat binary;
//     threshold(s, binary, 50, 255, THRESH_BINARY);  // 50 可调

//     return binary;
// }

Mat dynamicThreshold(const Mat& src) {
    Mat hsv, gray;
    cvtColor(src, hsv, COLOR_BGR2HSV);
    cvtColor(src, gray, COLOR_BGR2GRAY);

    vector<Mat> chans;
    split(hsv, chans);
    Mat s = chans[1];  // Saturation

    // 核心改进：彩色 OR 暗像素都当前景
    Mat darkMask, colorMask, binary;
    threshold(gray, darkMask, 140, 255, THRESH_BINARY_INV);  // 暗色物体→前景，尝试了120识别效果不够好
    threshold(s,    colorMask, 30,  255, THRESH_BINARY);     // 彩色物体→前景，尝试了50识别效果不够好改成30

    binary = darkMask | colorMask; // 合并两种类型前景

    return binary;
}

Mat morphologicalCleanup(const Mat& binary) {
    // Mat cleaned;
    // // 1. Closing：先填补小孔洞，不会吃掉细长结构
    // Mat kernelClose = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
    // morphologyEx(binary, cleaned, MORPH_CLOSE, kernelClose);

    // // 2. 轻微膨胀一下，确保细节不缺失
    // Mat kernelDilate = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
    // dilate(cleaned, cleaned, kernelDilate);

    // return cleaned;
    return binary;
}

vector<Vec3b> generateColorPalette(int n) {
    vector<Vec3b> palette;
    RNG rng(12345);
    for (int i = 0; i < n; i++) {
        palette.push_back(Vec3b(rng.uniform(50, 255), 
                                rng.uniform(50, 255), 
                                rng.uniform(50, 255)));
    }
    return palette;
}

Mat segmentRegions(const Mat& binary, Mat& labels, Mat& stats, Mat& centroids, 
                   int minArea = 300) {
    // 1) 连通域分析：把白色前景分成若干个“区域”（label = 0 是背景）
    int nLabels = connectedComponentsWithStats(binary, labels, stats, centroids, 8);
    // 2) 为可视化准备一组随机颜色
    vector<Vec3b> palette = generateColorPalette(256);
    // 3) 初始化一个彩色图，用于画出每个区域的颜色
    Mat coloredRegions = Mat::zeros(binary.size(), CV_8UC3);
    // 4) oldToNew：从“原始 label 编号” → “新的紧凑编号（1..K）”的映射
    map<int, int> oldToNew;
    int newLabel = 1;

    // 5) 遍历每个连通区域（0 是背景，所以从 1 开始）
    for (int i = 1; i < nLabels; i++) {
        int area   = stats.at<int>(i, CC_STAT_AREA);   // 像素数（区域面积）
        int left   = stats.at<int>(i, CC_STAT_LEFT);   // 外接矩形左上角 x
        int top    = stats.at<int>(i, CC_STAT_TOP);    // 外接矩形左上角 y
        int width  = stats.at<int>(i, CC_STAT_WIDTH);  // 外接矩形宽
        int height = stats.at<int>(i, CC_STAT_HEIGHT); // 外接矩形高
        
        // 5.2) 触边判断：只要外接矩形贴到图像边缘，就认为 touchesBorder = true
        bool touchesBorder = (left <= 1 || top <= 1 || 
                              left + width  >= binary.cols - 1 || 
                              top  + height >= binary.rows - 1);
        
        // 5.3) 筛选条件：面积要 >= minArea，且不能触边
        //      会把“小而靠边”的、以及“所有靠边”的区域都丢掉
        if (area >= minArea && !touchesBorder) {
            oldToNew[i] = newLabel++;   // 通过筛选的区域，赋予新编号 1..K
        }
    }
    
    // 6) 第二遍扫描：根据 oldToNew 重新标号，并给 coloredRegions 上色
    for (int y = 0; y < labels.rows; y++) {
        for (int x = 0; x < labels.cols; x++) {
            int label = labels.at<int>(y, x);   // 原始 label
            if (oldToNew.count(label)) {
                int newId = oldToNew[label];    // 压缩后的新 label
                coloredRegions.at<Vec3b>(y, x) = palette[newId % 256];
                labels.at<int>(y, x) = newId;   // 写回：把原 labels 改成新编号
            } else {
                labels.at<int>(y, x) = 0;       // 未通过筛选的像素全部当背景
            }
        }
    }
    
    // 7) 返回可视化颜色图（labels/stats/centroids 在参数里已经被填好）
    return coloredRegions;
}

RegionFeatures computeRegionFeatures(const Mat& labels, int regionId, 
                                     const Vec3b& color) {
    RegionFeatures features;
    features.regionId = regionId;
    features.color = color;
    
    Mat regionMask = (labels == regionId);
    
    Moments m = moments(regionMask, true);
    features.area = m.m00;
    features.centroid = Point2d(m.m10 / m.m00, m.m01 / m.m00);
    
    // Angle of least central moment
    double mu20 = m.mu20 / m.m00;
    double mu02 = m.mu02 / m.m00;
    double mu11 = m.mu11 / m.m00;
    features.angle = 0.5 * atan2(2 * mu11, mu20 - mu02) * 180 / CV_PI;
    
    // Oriented bounding box
    vector<Point> points;
    findNonZero(regionMask, points);
    RotatedRect rotRect = minAreaRect(points);
    features.boundingBox = rotRect.boundingRect();
    
    // Percent filled
    double bbArea = rotRect.size.width * rotRect.size.height;
    features.percentFilled = (bbArea > 0) ? (features.area / bbArea) : 0;
    
    // Aspect ratio (scale invariant)
    double w = max(rotRect.size.width, rotRect.size.height);
    double h = min(rotRect.size.width, rotRect.size.height);
    features.aspectRatio = (h > 0) ? (w / h) : 0;
    
    // Hu Moments (rotation, scale, translation invariant)
    HuMoments(m, features.huMoments);
    
    return features;
}

void drawRegionFeatures(Mat& display, const RegionFeatures& features, 
                        const Mat& labels) {
    Mat regionMask = (labels == features.regionId);
    vector<Point> points;
    findNonZero(regionMask, points);
    
    if (points.empty()) return;
    
    RotatedRect rotRect = minAreaRect(points);
    Point2f vertices[4];
    rotRect.points(vertices);
    
    // Draw oriented bounding box
    for (int i = 0; i < 4; i++) {
        line(display, vertices[i], vertices[(i + 1) % 4], Scalar(0, 255, 0), 2);
    }
    
    // Draw centroid
    circle(display, features.centroid, 5, Scalar(255, 0, 0), -1);
    
    // Draw axis of least central moment
    double length = 50;
    double rad = features.angle * CV_PI / 180.0;
    Point2d endpoint(features.centroid.x + length * cos(rad),
                     features.centroid.y + length * sin(rad));
    arrowedLine(display, features.centroid, endpoint, Scalar(0, 0, 255), 2);
    
    // Display feature values
    int y_offset = 30;
    putText(display, "Region " + to_string(features.regionId), 
            Point(10, y_offset), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 2);
    y_offset += 25;
    
    putText(display, "Area: " + to_string((int)features.area), 
            Point(10, y_offset), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
    y_offset += 20;
    
    putText(display, "Angle: " + to_string((int)features.angle) + " deg", 
            Point(10, y_offset), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
    y_offset += 20;
    
    char buf[100];
    sprintf(buf, "Fill: %.2f", features.percentFilled);
    putText(display, buf, Point(10, y_offset), 
            FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
    y_offset += 20;
    
    sprintf(buf, "Aspect: %.2f", features.aspectRatio);
    putText(display, buf, Point(10, y_offset), 
            FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
}

void printFeatureVector(const RegionFeatures& features) {
    cout << "\n=== Region " << features.regionId << " Features ===" << endl;
    cout << "Centroid: (" << features.centroid.x << ", " 
         << features.centroid.y << ")" << endl;
    cout << "Area: " << features.area << endl;
    cout << "Angle: " << features.angle << " degrees" << endl;
    cout << "Percent Filled: " << features.percentFilled << endl;
    cout << "Aspect Ratio: " << features.aspectRatio << endl;
    cout << "Hu Moments: [";
    for (int i = 0; i < 7; i++) {
        cout << features.huMoments[i];
        if (i < 6) cout << ", ";
    }
    cout << "]" << endl;
}

int main(int argc, char** argv) {
    VideoCapture cap;
    
    if (argc > 1) {
        cap.open(argv[1]);
    } else {
        cap.open(0);
    }
    
    if (!cap.isOpened()) {
        cerr << "Error: Cannot open video source" << endl;
        return -1;
    }
    
    namedWindow("Original", WINDOW_AUTOSIZE);
    namedWindow("Cleaned Binary", WINDOW_AUTOSIZE);
    namedWindow("Segmented Regions", WINDOW_AUTOSIZE);
    namedWindow("Features", WINDOW_AUTOSIZE);
    
    cout << "Press 's' to save and print features, 'q' to quit" << endl;
    int imgCount = 0;
    
    while (true) {
        Mat frame;
        cap >> frame;
        if (frame.empty()) break;
        
        Mat preprocessed = preprocessImage(frame);
        Mat thresholded = dynamicThreshold(preprocessed);
        Mat cleaned = morphologicalCleanup(thresholded);
        
        Mat labels, stats, centroids;
        Mat coloredRegions = segmentRegions(cleaned, labels, stats, centroids, 500);
        
        Mat featureDisplay = frame.clone();
        
        int maxLabel = 0;
        for (int y = 0; y < labels.rows; y++) {
            for (int x = 0; x < labels.cols; x++) {
                maxLabel = max(maxLabel, labels.at<int>(y, x));
            }
        }
        
        vector<RegionFeatures> allFeatures;
        for (int i = 1; i <= maxLabel; i++) {
            Vec3b color = coloredRegions.at<Vec3b>(centroids.at<double>(i, 1), 
                                                     centroids.at<double>(i, 0));
            RegionFeatures features = computeRegionFeatures(labels, i, color);
            allFeatures.push_back(features);
            drawRegionFeatures(featureDisplay, features, labels);
        }
        
        imshow("Original", frame);
        imshow("Cleaned Binary", cleaned);
        imshow("Segmented Regions", coloredRegions);
        imshow("Features", featureDisplay);
        
        char key = waitKey(30);
        if (key == 'q') break;
        
        if (key == 's') {
            string prefix = "task34_" + to_string(imgCount);
            imwrite(prefix + "_original.jpg", frame);
            imwrite(prefix + "_cleaned.jpg", cleaned);
            imwrite(prefix + "_regions.jpg", coloredRegions);
            imwrite(prefix + "_features.jpg", featureDisplay);
            
            cout << "\n========== Image " << imgCount << " ==========" << endl;
            for (const auto& f : allFeatures) {
                printFeatureVector(f);
            }
            imgCount++;
        }
    }
    
    cap.release();
    destroyAllWindows();
    
    return 0;
}