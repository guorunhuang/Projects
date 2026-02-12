// 显示了我的左眼睛和左半边嘴巴。而且土豆贴在了脸上，显示成了jpg而不是png
#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;

// 把 RGBA 前景图贴到 BGR 背景图上
void overlayImage(const Mat &background, const Mat &foreground, Mat &output, Point2i location) {
    background.copyTo(output);

    for (int y = max(location.y, 0); y < background.rows; ++y) {
        int fY = y - location.y; // foreground row

        if (fY >= foreground.rows) break;

        for (int x = max(location.x, 0); x < background.cols; ++x) {
            int fX = x - location.x; // foreground col
            if (fX >= foreground.cols) break;

            Vec4b fgPixel = foreground.at<Vec4b>(fY, fX);
            Vec3b &bgPixel = output.at<Vec3b>(y, x);

            float alpha = fgPixel[3] / 255.0; // alpha 通道
            for (int c = 0; c < 3; ++c) {
                bgPixel[c] = bgPixel[c] * (1.0 - alpha) + fgPixel[c] * alpha;
            }
        }
    }
}

int main() {
    // 加载人脸、眼睛、嘴巴分类器
    CascadeClassifier face_cascade, eye_cascade, mouth_cascade;
    if (!face_cascade.load("haarcascade_frontalface_alt2.xml")) {
        cout << "Error loading face cascade\n"; return -1;
    }
    if (!eye_cascade.load("haarcascade_eye.xml")) {
        cout << "Error loading eye cascade\n"; return -1;
    }
    if (!mouth_cascade.load("haarcascade_mcs_mouth.xml")) {
        cout << "Error loading mouth cascade\n"; return -1;
    }

    // 打开摄像头
    VideoCapture cap(0);
    if (!cap.isOpened()) return -1;

    // 读取土豆图（带透明背景 PNG）
    Mat potato_rgba = imread("potato.png", IMREAD_UNCHANGED);
    if (potato_rgba.empty()) {
        cout << "Cannot load potato.png\n"; return -1;
    }

    Mat frame, gray;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        cvtColor(frame, gray, COLOR_BGR2GRAY);
        equalizeHist(gray, gray);

        vector<Rect> faces;
        face_cascade.detectMultiScale(gray, faces);

        Mat display = frame.clone();

        for (auto &face : faces) {
            // 人脸中心点
            Point face_center(face.x + face.width/2, face.y + face.height/2);

            // 缩放土豆大小（与人脸宽度差不多）
            Mat potato_resized;
            double scale = face.width * 1.5 / potato_rgba.cols;
            resize(potato_rgba, potato_resized, Size(), scale, scale);

            // 在人脸区域检测眼睛和嘴巴
            Mat faceROI = gray(face);
            vector<Rect> eyes, mouths;
            eye_cascade.detectMultiScale(faceROI, eyes, 1.1, 3, 0, Size(30,30));
            mouth_cascade.detectMultiScale(faceROI, mouths, 1.1, 5, 0, Size(40,40));

            Mat potato_with_face = potato_resized.clone();

            // 土豆内部位置（预定义锚点）
            int eye_y = potato_resized.rows / 3;
            int mouth_y = potato_resized.rows * 2 / 3;

            // 贴眼睛
            if (!eyes.empty()) {
                Rect eye_rect(face.x + eyes[0].x, face.y + eyes[0].y, eyes[0].width, eyes[0].height);
                Mat eyeROI = frame(eye_rect).clone();
                resize(eyeROI, eyeROI, Size(potato_resized.cols/5, potato_resized.rows/8));
                cvtColor(eyeROI, eyeROI, COLOR_BGR2BGRA);
                overlayImage(potato_with_face, eyeROI, potato_with_face,
                             Point(potato_resized.cols/4, eye_y));
            }

            // 贴嘴巴
            if (!mouths.empty()) {
                Rect mouth_rect(face.x + mouths[0].x, face.y + mouths[0].y, mouths[0].width, mouths[0].height);
                Mat mouthROI = frame(mouth_rect).clone();
                resize(mouthROI, mouthROI, Size(potato_resized.cols/3, potato_resized.rows/6));
                cvtColor(mouthROI, mouthROI, COLOR_BGR2BGRA);
                overlayImage(potato_with_face, mouthROI, potato_with_face,
                             Point(potato_resized.cols/3, mouth_y));
            }

            // 把整颗土豆贴到画面上（锚点 = 人脸中心）
            int x = face_center.x - potato_with_face.cols/2;
            int y = face_center.y - potato_with_face.rows/2;
            overlayImage(display, potato_with_face, display, Point(x,y));
        }

        imshow("Original", frame);
        imshow("Potato Human", display);

        if (waitKey(10) == 'q') break;
    }

    return 0;
}
