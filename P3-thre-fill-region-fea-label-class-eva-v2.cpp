#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <map>
#include <fstream>
#include <cmath>

#include <fstream>
#include <iomanip>

using namespace cv;
using namespace std;

// evaluate
const int N_CLASSES = 5; // change if needed
const vector<string> CLASS_NAMES = {"class1","class2","class3","class4","class5"}; // set real names

// confusion matrix: rows=true, cols=pred
static Mat confusion = Mat::zeros(N_CLASSES, N_CLASSES, CV_32S);
static int evalMode = 0; // 0 = normal, 1 = evaluation
static int evalCount = 0;
static int evalTargetSamplesPerClass = 3; // need >= 3 per class (adjust)

void addConfusion(int trueLabelIdx, int predLabelIdx) {
    if (trueLabelIdx < 0 || trueLabelIdx >= N_CLASSES) return;
    if (predLabelIdx < 0 || predLabelIdx >= N_CLASSES) return;
    confusion.at<int>(trueLabelIdx, predLabelIdx)++;
    evalCount++;
}

void saveConfusionCSV(const string &filename) {
    ofstream ofs(filename);
    ofs << "true/pred,";
    for (int c = 0; c < N_CLASSES; ++c) {
        ofs << CLASS_NAMES[c] << (c+1==N_CLASSES? "\n" : ",");
    }
    for (int r = 0; r < N_CLASSES; ++r) {
        ofs << CLASS_NAMES[r] << ",";
        for (int c = 0; c < N_CLASSES; ++c) {
            ofs << confusion.at<int>(r,c) << (c+1==N_CLASSES? "\n" : ",");
        }
    }
    ofs.close();
}

// label
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

struct TrainingExample {
    string label;
    vector<double> features;
};

class ObjectDatabase {
private:
    vector<TrainingExample> database;
    vector<double> featureStdDev;
    string dbFilename;
    
    void computeStdDev() {
        if (database.empty()) return;
        
        int numFeatures = database[0].features.size();
        vector<double> means(numFeatures, 0.0);
        vector<double> variances(numFeatures, 0.0);
        
        for (const auto& example : database) {
            for (int i = 0; i < numFeatures; i++) {
                means[i] += example.features[i];
            }
        }
        
        for (int i = 0; i < numFeatures; i++) {
            means[i] /= database.size();
        }
        
        for (const auto& example : database) {
            for (int i = 0; i < numFeatures; i++) {
                double diff = example.features[i] - means[i];
                variances[i] += diff * diff;
            }
        }
        
        featureStdDev.resize(numFeatures);
        for (int i = 0; i < numFeatures; i++) {
            featureStdDev[i] = sqrt(variances[i] / database.size());
            if (featureStdDev[i] < 1e-6) featureStdDev[i] = 1.0;
        }
    }
    
public:
    ObjectDatabase(const string& filename) : dbFilename(filename) {
        loadFromFile();
    }
    
    void addTrainingExample(const string& label, const vector<double>& features) {
        TrainingExample example;
        example.label = label;
        example.features = features;
        database.push_back(example);
        computeStdDev();
        saveToFile();
        cout << "Added training example: " << label << endl;
    }
    
    pair<string, double> classify(const vector<double>& features, double& confidence) {
        if (database.empty()) {
            return make_pair("UNKNOWN", 0.0);
        }
        
        double minDistance = numeric_limits<double>::max();
        string bestLabel = "UNKNOWN";
        
        for (const auto& example : database) {
            double distance = scaledEuclideanDistance(features, example.features);
            if (distance < minDistance) {
                minDistance = distance;
                bestLabel = example.label;
            }
        }
        
        confidence = 1.0 / (1.0 + minDistance);
        
        double unknownThreshold = 0.3;
        if (confidence < unknownThreshold) {
            return make_pair("UNKNOWN", confidence);
        }
        
        return make_pair(bestLabel, confidence);
    }
    
    double scaledEuclideanDistance(const vector<double>& f1, const vector<double>& f2) {
        if (f1.size() != f2.size() || featureStdDev.empty()) {
            return numeric_limits<double>::max();
        }
        
        double sum = 0.0;
        for (size_t i = 0; i < f1.size(); i++) {
            double diff = (f1[i] - f2[i]) / featureStdDev[i];
            sum += diff * diff;
        }
        return sqrt(sum);
    }
    
    void saveToFile() {
        ofstream file(dbFilename);
        if (!file.is_open()) {
            cerr << "Error: Cannot save database to " << dbFilename << endl;
            return;
        }
        
        file << database.size() << endl;
        for (const auto& example : database) {
            file << example.label << " ";
            file << example.features.size() << " ";
            for (double feat : example.features) {
                file << feat << " ";
            }
            file << endl;
        }
        file.close();
        cout << "Database saved: " << database.size() << " examples" << endl;
    }
    
    void loadFromFile() {
        ifstream file(dbFilename);
        if (!file.is_open()) {
            cout << "No existing database found. Starting fresh." << endl;
            return;
        }
        
        int numExamples;
        file >> numExamples;
        
        for (int i = 0; i < numExamples; i++) {
            TrainingExample example;
            file >> example.label;
            
            int numFeatures;
            file >> numFeatures;
            example.features.resize(numFeatures);
            
            for (int j = 0; j < numFeatures; j++) {
                file >> example.features[j];
            }
            database.push_back(example);
        }
        file.close();
        
        computeStdDev();
        cout << "Database loaded: " << database.size() << " examples" << endl;
    }
    
    void printDatabase() {
        cout << "\n=== Object Database ===" << endl;
        cout << "Total examples: " << database.size() << endl;
        
        map<string, int> labelCounts;
        for (const auto& example : database) {
            labelCounts[example.label]++;
        }
        
        for (const auto& pair : labelCounts) {
            cout << "  " << pair.first << ": " << pair.second << " examples" << endl;
        }
    }
    
    int size() const { return database.size(); }
};

Mat preprocessImage(const Mat& src) {
    Mat blurred;
    GaussianBlur(src, blurred, Size(5, 5), 1.0);
    return blurred;
}

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
    // Mat kernel1 = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
    // morphologyEx(binary, cleaned, MORPH_OPEN, kernel1);
    
    // Mat kernel2 = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
    // morphologyEx(cleaned, cleaned, MORPH_CLOSE, kernel2);
    
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
        // bool touchesBorder = (left <= 1 || top <= 1 || 
        //                       left + width  >= binary.cols - 1 || 
        //                       top  + height >= binary.rows - 1);
        
        // 5.3) 筛选条件：面积要 >= minArea，且不能触边
        //      会把“小而靠边”的、以及“所有靠边”的区域都丢掉
        // if (area >= minArea && !touchesBorder) {
        if (area >= minArea) {
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

RegionFeatures computeRegionFeatures(const Mat& labels, int regionId) {
    RegionFeatures features;
    features.regionId = regionId;
    
    Mat regionMask = (labels == regionId);
    
    Moments m = moments(regionMask, true);
    features.area = m.m00;
    features.centroid = Point2d(m.m10 / m.m00, m.m01 / m.m00);
    
    double mu20 = m.mu20 / m.m00;
    double mu02 = m.mu02 / m.m00;
    double mu11 = m.mu11 / m.m00;
    features.angle = 0.5 * atan2(2 * mu11, mu20 - mu02) * 180 / CV_PI;
    
    vector<Point> points;
    findNonZero(regionMask, points);
    RotatedRect rotRect = minAreaRect(points);
    features.boundingBox = rotRect.boundingRect();
    
    double bbArea = rotRect.size.width * rotRect.size.height;
    features.percentFilled = (bbArea > 0) ? (features.area / bbArea) : 0;
    
    double w = max(rotRect.size.width, rotRect.size.height);
    double h = min(rotRect.size.width, rotRect.size.height);
    features.aspectRatio = (h > 0) ? (w / h) : 0;
    
    HuMoments(m, features.huMoments);
    
    return features;
}

vector<double> extractFeatureVector(const RegionFeatures& features) {
    vector<double> fv;
    fv.push_back(features.percentFilled);
    fv.push_back(features.aspectRatio);
    
    for (int i = 0; i < 7; i++) {
        fv.push_back(log(abs(features.huMoments[i]) + 1e-10));
    }
    
    return fv;
}

void drawClassification(Mat& display, const RegionFeatures& features, 
                       const string& label, double confidence, const Mat& labels) {
    Mat regionMask = (labels == features.regionId);
    vector<Point> points;
    findNonZero(regionMask, points);
    
    if (points.empty()) return;
    
    RotatedRect rotRect = minAreaRect(points);
    Point2f vertices[4];
    rotRect.points(vertices);
    
    Scalar color = (label == "UNKNOWN") ? Scalar(0, 0, 255) : Scalar(0, 255, 0);
    
    for (int i = 0; i < 4; i++) {
        line(display, vertices[i], vertices[(i + 1) % 4], color, 2);
    }
    
    circle(display, features.centroid, 5, Scalar(255, 0, 0), -1);
    
    char text[100];
    sprintf(text, "%s (%.2f)", label.c_str(), confidence);
    
    int baseline = 0;
    Size textSize = getTextSize(text, FONT_HERSHEY_SIMPLEX, 0.8, 2, &baseline);
    
    Point textPos(features.centroid.x - textSize.width / 2, 
                  features.centroid.y - 20);
    
    rectangle(display, 
              Point(textPos.x - 5, textPos.y - textSize.height - 5),
              Point(textPos.x + textSize.width + 5, textPos.y + 5),
              Scalar(0, 0, 0), -1);
    
    putText(display, text, textPos, FONT_HERSHEY_SIMPLEX, 0.8, 
            Scalar(255, 255, 255), 2);
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
    
    ObjectDatabase objDB("object_database.txt");
    
    namedWindow("Classification", WINDOW_AUTOSIZE);
    
    cout << "\n=== Object Recognition System ===" << endl;
    cout << "Commands:" << endl;
    cout << "  'n' - Train new object (enter label)" << endl;
    cout << "  'd' - Display database contents" << endl;
    cout << "  's' - Save current frame with classification" << endl;
    cout << "  'q' - Quit" << endl;
    cout << "===================================\n" << endl;
    
    objDB.printDatabase();
    
    int imgCount = 0;
    
    while (true) {
        Mat frame;
        cap >> frame;
        if (frame.empty()) break;
        
        Mat preprocessed = preprocessImage(frame);
        Mat thresholded = dynamicThreshold(preprocessed);
        Mat cleaned = morphologicalCleanup(thresholded);
        
        Mat labels, stats, centroids;
        segmentRegions(cleaned, labels, stats, centroids, 500);
        
        Mat display = frame.clone();
        
        int maxLabel = 0;
        for (int y = 0; y < labels.rows; y++) {
            for (int x = 0; x < labels.cols; x++) {
                maxLabel = max(maxLabel, labels.at<int>(y, x));
            }
        }
        
        vector<RegionFeatures> allFeatures;
        for (int i = 1; i <= maxLabel; i++) {
            RegionFeatures features = computeRegionFeatures(labels, i);
            allFeatures.push_back(features);
            
            vector<double> featureVec = extractFeatureVector(features);
            
            double confidence;
            pair<string, double> result = objDB.classify(featureVec, confidence);
            
            drawClassification(display, features, result.first, result.second, labels);
        }
        
        imshow("Classification", display);
        
        char key = waitKey(100);
        
        if (key == 'q') break;
        
        if (key == 'n' && !allFeatures.empty()) {
            cout << "\nEnter object label: ";
            string label;
            cin >> label;
            
            RegionFeatures features = allFeatures[0];
            vector<double> featureVec = extractFeatureVector(features);
            
            objDB.addTrainingExample(label, featureVec);
            
            cout << "Feature vector: [";
            for (size_t i = 0; i < featureVec.size(); i++) {
                cout << featureVec[i];
                if (i < featureVec.size() - 1) cout << ", ";
            }
            cout << "]" << endl;
        }
        
        if (key == 'd') {
            objDB.printDatabase();
        }
        
        if (key == 's') {
            string filename = "classified_" + to_string(imgCount) + ".jpg";
            imwrite(filename, display);
            cout << "Saved: " << filename << endl;
            imgCount++;
        }

        if (key == 'e') { // toggle evaluation mode
            evalMode = 1 - evalMode;
            cout << "Evaluation mode = " << evalMode << endl;
        }
        
        if (evalMode) {
            if (!allFeatures.empty()) {
                auto features = allFeatures[0];
                vector<double> featVec = extractFeatureVector(features);
                double confidence;
                pair<string,double> result = objDB.classify(featVec, confidence);
        
                int predIdx = -1;
                for (int i=0;i<N_CLASSES;++i)
                    if (CLASS_NAMES[i] == result.first)
                        predIdx = i;
        
                cout << "Predicted: " << result.first << " (" << predIdx << ")";
                cout << "  Press 1.." << N_CLASSES << " for TRUE label, 0 to skip.\n";
        
                
                int key = waitKey(0);  
                if (key == 'q') {       // 全局退出
                    return 0;
                }
            
                if (key == 'e') {       // 仅退出评估模式, 回主循环继续训练/识别
                    evalMode = 0;
                    continue;           // 回到 while(true)，继续下一帧，不退出程序
                }
        
                if (key >= '1' && key <= char('0' + N_CLASSES)) {
                    int trueIdx = key - '1';
                    if (predIdx >= 0) addConfusion(trueIdx, predIdx);
                    cout << "Added: true=" << CLASS_NAMES[trueIdx]
                         << " pred=" << (predIdx>=0?CLASS_NAMES[predIdx]:"<none>") << "\n";
                } else if (key == '0') {
                    cout << "Skipped sample.\n";
                } else if (key == 'r') {
                    confusion = Mat::zeros(N_CLASSES, N_CLASSES, CV_32S);
                    cout << "Confusion reset.\n";
                } else if (key == 'w') {
                    saveConfusionCSV("confusion.csv");
                    cout << "Saved confusion.csv\n";
                }
                
            }
        }
    }
    cap.release();
    destroyAllWindows();
    
    return 0;
    
}