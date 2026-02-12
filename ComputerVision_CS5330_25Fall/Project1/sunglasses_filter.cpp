#include <opencv2/opencv.hpp>
#include <iostream>
#include <ctime>
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

// 获取时间戳字符串
string getTimestamp() {
    time_t now = time(0);
    tm *ltm = localtime(&now);

    char buf[32];
    sprintf(buf, "%04d%02d%02d_%02d%02d%02d",
            1900 + ltm->tm_year,
            1 + ltm->tm_mon,
            ltm->tm_mday,
            ltm->tm_hour,
            ltm->tm_min,
            ltm->tm_sec);
    return string(buf);
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

    // 读取墨镜 PNG（必须有透明背景）
    Mat sunglasses = imread("sunglasses1.png", IMREAD_UNCHANGED);
    if (sunglasses.empty()) {
        cout << "Cannot load sunglasses1.png\n"; return -1;
    }

    Mat frame, gray;
    bool recording = false;
    VideoWriter writer;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        cvtColor(frame, gray, COLOR_BGR2GRAY);
        equalizeHist(gray, gray);

        vector<Rect> faces;
        face_cascade.detectMultiScale(gray, faces);

        for (auto &face : faces) {
            Mat faceROI = gray(face);
            vector<Rect> eyes;
            eye_cascade.detectMultiScale(faceROI, eyes, 1.1, 3, 0, Size(30, 30));

            if (eyes.size() >= 2) {
                sort(eyes.begin(), eyes.end(), [](Rect a, Rect b) { return a.x < b.x; });

                Rect leftEye(face.x + eyes[0].x, face.y + eyes[0].y, eyes[0].width, eyes[0].height);
                Rect rightEye(face.x + eyes[1].x, face.y + eyes[1].y, eyes[1].width, eyes[1].height);

                Point2i midPoint((leftEye.x + leftEye.width/2 + rightEye.x + rightEye.width/2) / 2,
                                 (leftEye.y + leftEye.height/2 + rightEye.y + rightEye.height/2) / 2);

                int targetWidth = static_cast<int>(face.width * 0.9);
                int targetHeight = (int)((float)targetWidth / sunglasses.cols * sunglasses.rows);

                Mat sunglass_resized;
                resize(sunglasses, sunglass_resized, Size(targetWidth, targetHeight));

                Point2i topLeft(midPoint.x - targetWidth/2, midPoint.y - targetHeight/2);
                overlayImage(frame, sunglass_resized, topLeft);
            }
        }

        imshow("Sunglasses Filter", frame);

        // 写入视频（如果在录制中）
        if (recording) {
            writer.write(frame);
        }

        char key = (char)waitKey(10);
        if (key == 'q') break;
        else if (key == 's') {
            if (!recording) {
                // 开始录制
                string filename = "sunglasses_" + getTimestamp() + ".avi";
                int fourcc = VideoWriter::fourcc('M','J','P','G'); 
                double fps = cap.get(CAP_PROP_FPS);
                if (fps <= 1.0) fps = 30.0; // 默认30fps
                Size size((int)cap.get(CAP_PROP_FRAME_WIDTH), (int)cap.get(CAP_PROP_FRAME_HEIGHT));
                writer.open(filename, fourcc, fps, size, true);

                if (!writer.isOpened()) {
                    cout << "Error: Cannot open video writer\n";
                } else {
                    recording = true;
                    cout << "Started recording: " << filename << endl;
                }
            } else {
                // 停止录制
                recording = false;
                writer.release();
                cout << "Stopped recording." << endl;
            }
        }
    }

    cap.release();
    if (writer.isOpened()) writer.release();
    destroyAllWindows();
    return 0;
}
