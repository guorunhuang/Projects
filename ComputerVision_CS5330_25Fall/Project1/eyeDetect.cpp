#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;

int main() {
    // 加载分类器
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

    VideoCapture cap(0);
    if (!cap.isOpened()) return -1;

    Mat frame, gray;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        cvtColor(frame, gray, COLOR_BGR2GRAY);
        equalizeHist(gray, gray);

        vector<Rect> faces;
        face_cascade.detectMultiScale(gray, faces);

        for (auto &face : faces) {
            rectangle(frame, face, Scalar(255,0,0), 2);

            // ROI = 人脸区域
            Mat faceROI = gray(face);

            // 检测眼睛
            vector<Rect> eyes;
            eye_cascade.detectMultiScale(faceROI, eyes, 1.1, 3, 0, Size(30,30));
            for (auto &eye : eyes) {
                Rect eye_rect(face.x + eye.x, face.y + eye.y, eye.width, eye.height);
                rectangle(frame, eye_rect, Scalar(0,255,0), 2);
            }

            // 检测嘴巴（一般在脸下半部分）
            vector<Rect> mouths;
            mouth_cascade.detectMultiScale(faceROI, mouths, 1.1, 5, 0, Size(40,40));
            for (auto &mouth : mouths) {
                Rect mouth_rect(face.x + mouth.x, face.y + mouth.y, mouth.width, mouth.height);
                rectangle(frame, mouth_rect, Scalar(0,0,255), 2);
            }
        }

        imshow("Potato Human", frame);
        if (waitKey(10) == 'q') break;
    }

    return 0;
}
