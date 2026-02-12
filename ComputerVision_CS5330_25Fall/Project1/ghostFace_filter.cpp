#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;

// 透明叠加函数
void overlayImage(Mat &background, const Mat &foreground, Point2i location) {
    for (int y = max(location.y, 0); y < background.rows; ++y) {
        int fY = y - location.y;
        if (fY >= foreground.rows) break;

        for (int x = max(location.x, 0); x < background.cols; ++x) {
            int fX = x - location.x;
            if (fX >= foreground.cols) break;

            Vec4b fgPixel = foreground.at<Vec4b>(fY, fX);
            Vec3b &bgPixel = background.at<Vec3b>(y, x);

            float alpha = fgPixel[3] / 255.0;
            for (int c = 0; c < 3; ++c) {
                bgPixel[c] = bgPixel[c] * (1.0 - alpha) + fgPixel[c] * alpha;
            }
        }
    }
}

int main() {
    CascadeClassifier face_cascade, eye_cascade;
    if (!face_cascade.load("haarcascade_frontalface_alt2.xml")) {
        cout << "Error loading face cascade\n"; return -1;
    }
    if (!eye_cascade.load("haarcascade_eye.xml")) {
        cout << "Error loading eye cascade\n"; return -1;
    }

    VideoCapture cap(0);
    if (!cap.isOpened()) return -1;

    // 读取鬼脸 PNG（必须有透明背景）
    Mat ghost_rgba = imread("ghostSheet3.png", IMREAD_UNCHANGED);
    if (ghost_rgba.empty()) {
        cout << "Cannot load ghost.png\n"; return -1;
    }

    Mat frame, gray;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        cvtColor(frame, gray, COLOR_BGR2GRAY);
        equalizeHist(gray, gray);

        vector<Rect> faces;
        face_cascade.detectMultiScale(gray, faces);

        for (auto &face : faces) {
            // 缩放鬼脸到人脸大小
            Mat ghost_resized;
            // resize(ghost_rgba, ghost_resized, face.size());
            resize(ghost_rgba, ghost_resized, Size(face.width * 2, face.height * 2));

            // 在人脸 ROI 内检测眼睛
            Mat faceROI = gray(face);
            vector<Rect> eyes;
            eye_cascade.detectMultiScale(faceROI, eyes, 1.1, 3, 0, Size(30, 30));

            // 按 y 坐标排序，取最上面的两个矩形（左右眼）
            sort(eyes.begin(), eyes.end(), [](Rect a, Rect b) { return a.y < b.y; });

            if (eyes.size() >= 2) {
                for (int i = 0; i < 2; i++) {
                    Rect eye = eyes[i];
                    Rect eye_rect(face.x + eye.x, face.y + eye.y, eye.width, eye.height);

                    // 把原始眼睛抠出来，覆盖到鬼脸对应区域
                    if (eye_rect.x >= 0 && eye_rect.y >= 0 &&
                        eye_rect.x + eye_rect.width <= frame.cols &&
                        eye_rect.y + eye_rect.height <= frame.rows) 
                    {
                        Mat eyeROI = frame(eye_rect).clone();
                        resize(eyeROI, eyeROI, Size(eye.width, eye.height));
                        cvtColor(eyeROI, eyeROI, COLOR_BGR2BGRA);

                        for (int y = 0; y < eyeROI.rows; y++) {
                            for (int x = 0; x < eyeROI.cols; x++) {
                                Vec4b &dstPixel = ghost_resized.at<Vec4b>(eye.y + y, eye.x + x);
                                Vec4b srcPixel = eyeROI.at<Vec4b>(y, x);
                                dstPixel = srcPixel; // 用眼睛替换鬼脸对应区域
                            }
                        }
                    }
                }
            }

            // 叠加鬼脸到原始画面
            overlayImage(frame, ghost_resized, face.tl());
        }

        imshow("Ghost Filter", frame);

        if (waitKey(10) == 'q') break;
    }

    return 0;
}
