/*
One-shot CNN training
*/
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <filesystem>

using namespace cv;
using namespace std;

struct RegionParams {
    // 已有的区域特征：主轴中心(cx, cy)，旋转角度(度)，以及旋转后轴对齐的宽高
    // 若拿不到这些参数，set valid=false，代码会退化为中心正方形 ROI。
    double cx=0, cy=0;
    double angleDeg=0;
    double width=0, height=0;
    bool valid=false;
};

// 可在这里接入已计算好的区域特征
// 返回 true 则使用区域对齐与裁剪；返回 false 则使用中心正方形 ROI。
static bool getRegionParamsFromYourPipeline(const Mat& frame, RegionParams& rp) {
    // TODO: 把这段替换为你真实的特征获取逻辑：
    // rp.cx = ...; rp.cy = ...; rp.angleDeg = ...; rp.width = ...; rp.height = ...; rp.valid = true;
    // 这里默认 false，表示走“中心裁剪”的回退逻辑。
    (void)frame;
    rp.valid = false;
    return false;
}

// 旋转图像（绕图像中心）
static Mat rotateImage(const Mat& src, double angleDeg) {
    Point2f center(src.cols/2.f, src.rows/2.f);
    Mat R = getRotationMatrix2D(center, angleDeg, 1.0);
    Mat dst;
    warpAffine(src, dst, R, src.size(), INTER_LINEAR, BORDER_REPLICATE);
    return dst;
}

// 从给定中心与宽高裁剪轴对齐 ROI（自动裁边）
static Mat cropROI(const Mat& img, double cx, double cy, double w, double h) {
    Rect roi(
        int(round(cx - w/2.0)),
        int(round(cy - h/2.0)),
        int(round(w)),
        int(round(h))
    );
    roi &= Rect(0,0,img.cols,img.rows);
    if (roi.width <= 0 || roi.height <= 0) return Mat();
    return img(roi).clone();
}

// 居中正方形裁剪（回退用）
static Mat centerSquareCrop(const Mat& img) {
    int side = std::min(img.cols, img.rows);
    int x = (img.cols - side)/2;
    int y = (img.rows - side)/2;
    Rect r(x,y,side,side);
    return img(r).clone();
}

// 归一化到 [0,1] 并做 ImageNet 均值方差标准化，转为 NCHW blob（224x224）
static Mat makeInputBlob(const Mat& imgBGR, const Size target=Size(224,224)) {
    Mat img;
    // BGR->RGB
    cvtColor(imgBGR, img, COLOR_BGR2RGB);
    // resize 到 CNN 输入大小
    resize(img, img, target, 0, 0, INTER_LINEAR);
    img.convertTo(img, CV_32F, 1.0/255.0);
    // ImageNet mean/std
    Scalar mean(0.485, 0.456, 0.406);
    Scalar stdv(0.229, 0.224, 0.225);
    img = (img - mean) / stdv;
    // HWC -> NCHW
    Mat blob = dnn::blobFromImage(img); // already RGB float, NCHW
    return blob;
}

// 从 net 中选择一个“倒数第二层/特征层”作为输出；若找不到则用默认输出
static string pickFeatureLayerOrEmpty(const dnn::Net& net) {
    // 常见候选层名（不同导出工具可能不同）
    // 这些名字是“包含关系”匹配（见下）
    vector<string> prefer = {
        "globalaveragepool", "avgpool", "pool0", "flatten", "features", "resnetv1b0_pool",
        "resnetv24_pool", "resnetv1d_pool", "Gemm_pre", "fc_pre", "penultimate"
    };
    vector<string> layerNames = net.getLayerNames();
    // 遍历末尾到前面，找更接近输出端的中间层
    for (int i = int(layerNames.size())-1; i >= 0; --i) {
        string name = layerNames[i];
        string lname = name;
        std::transform(lname.begin(), lname.end(), lname.begin(), ::tolower);
        for (auto& key : prefer) {
            string k = key;
            std::transform(k.begin(), k.end(), k.begin(), ::tolower);
            if (lname.find(k) != string::npos) {
                return name; // 命中一个候选层
            }
        }
    }
    return ""; // 返回空串表示使用默认输出（logits）
}

// 将输出张量展平成向量
static vector<float> flatten(const Mat& blob) {
    // blob 是 1xCxHxW 或 1xN
    Mat f = blob.reshape(1, 1); // 变成 1x(total)
    vector<float> v;
    f.copyTo(v);
    return v;
}

// L2 归一化（可选）
static void l2normalize(vector<float>& v) {
    double s=0;
    for (float x : v) s += double(x)*x;
    s = std::sqrt(std::max(1e-12, s));
    for (auto& x : v) x = float(x / s);
}

// SSD 距离
static double ssd(const vector<float>& a, const vector<float>& b) {
    if (a.size() != b.size()) return std::numeric_limits<double>::infinity();
    double s=0;
    for (size_t i=0;i<a.size();++i){
        double d = double(a[i]) - double(b[i]);
        s += d*d;
    }
    return s;
}

// 计算单帧 embedding（含预处理）
static vector<float> computeEmbedding(
    const Mat& frameBGR,
    dnn::Net& net,
    const string& featureLayer /* empty -> 默认输出 */
) {
    // 1) 区域对齐（若已完成分割/主轴/尺度）
    RegionParams rp;
    bool hasRegion = getRegionParamsFromYourPipeline(frameBGR, rp);
    Mat work = frameBGR;

    if (hasRegion && rp.valid) {
        // 旋转使主轴与 X 轴对齐（注意是旋转 -theta）
        Mat rotated = rotateImage(work, -rp.angleDeg);
        // 旋转后按宽高裁剪
        Mat roi = cropROI(rotated, rp.cx, rp.cy, rp.width, rp.height);
        if (!roi.empty()) work = roi;
        else work = centerSquareCrop(work); // 容错回退
    } else {
        // 没有区域信息则中心正方形裁剪
        work = centerSquareCrop(work);
    }

    // 2) 形成输入 blob 
    Mat blob = makeInputBlob(work, Size(224,224));
    net.setInput(blob);

    // 3) 前向推理到特征层 / 默认输出 
    Mat out;
    if (!featureLayer.empty()) {
        // 指定中间层作为输出
        vector<String> outNames = { featureLayer };
        vector<Mat> outs;
        net.forward(outs, outNames);
        out = outs[0];
    } else {
        out = net.forward();
    }

    // 4) 展平 & 归一化（建议做）
    vector<float> feat = flatten(out);
    l2normalize(feat);
    return feat;
}

// 简单的嵌入库（内存 + 文本持久化）
struct EmbeddingStore {
    vector<pair<string, vector<float>>> items; // (label, embedding)

    bool load(const string& path) {
        items.clear();
        ifstream fin(path);
        if (!fin.is_open()) return false;
        string line;
        while (std::getline(fin, line)) {
            if (line.empty()) continue;
            // 格式：label,val1,val2,...,valN   （逗号分隔）
            stringstream ss(line);
            string label;
            if (!std::getline(ss, label, ',')) continue;
            vector<float> v;
            string tok;
            while (std::getline(ss, tok, ',')) {
                try {
                    v.push_back(stof(tok));
                } catch(...) {}
            }
            if (!label.empty() && !v.empty()) items.emplace_back(label, std::move(v));
        }
        return true;
    }

    bool append(const string& path, const string& label, const vector<float>& v) {
        // 先写文件
        ofstream fout(path, ios::app);
        if (!fout.is_open()) return false;
        fout << label;
        for (size_t i=0;i<v.size();++i) fout << "," << std::setprecision(8) << v[i];
        fout << "\n";
        fout.close();
        // 再放到内存
        items.emplace_back(label, v);
        return true;
    }

    bool empty() const { return items.empty(); }

    // 返回 (bestLabel, bestDist)
    pair<string, double> predict(const vector<float>& q) const {
        string bestLabel = "UNKNOWN";
        double bestD = std::numeric_limits<double>::infinity();
        for (auto& it : items) {
            double d = ssd(q, it.second);
            if (d < bestD) {
                bestD = d;
                bestLabel = it.first;
            }
        }
        return {bestLabel, bestD};
    }
};

static void ensureDir(const string& dir) {
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
}

int main(int argc, char** argv) {
    (void)argc; (void)argv;

    const string onnxPath = "resnet18-v2-7.onnx";
    const string dbPath   = "P3-embeddings.txt";
    const string imgDir   = "P3-train_imgs";

    // 1) 载入网络
    dnn::Net net = dnn::readNet(onnxPath);
    if (net.empty()) {
        cerr << "[ERROR] Failed to load ONNX: " << onnxPath << endl;
        return -1;
    }
    // 优先使用 CPU，避免无 GPU 时崩溃
    net.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(dnn::DNN_TARGET_CPU);

    // 2) 选择一个合适的中间层（若找不到则用默认输出）
    string featureLayer = pickFeatureLayerOrEmpty(net);
    if (!featureLayer.empty()) {
        cout << "[INFO] Using intermediate layer as embedding: " << featureLayer << endl;
    } else {
        cout << "[INFO] Using final output (logits) as embedding." << endl;
    }

    // 3) 载入已有样本
    EmbeddingStore store;
    store.load(dbPath);
    cout << "[INFO] Loaded samples: " << store.items.size() << " from " << dbPath << endl;

    // 4) 打开摄像头
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "[ERROR] Cannot open camera 0" << endl;
        return -2;
    }

    // 5) 运行模式：0=训练, 1=分类（默认先训练）
    int mode = 0;
    cout << "Select mode: 0=Train, 1=Classify  [default 0]: ";
    {
        string line; std::getline(cin, line);
        if (!line.empty() && (line=="1")) mode = 1;
    }
    cout << (mode==0 ? "[MODE] TRAIN" : "[MODE] CLASSIFY") << endl;

    ensureDir(imgDir);
    const string winName = "One-Shot Classification (Press Q to quit, M to switch)";
    namedWindow(winName, WINDOW_NORMAL);

    auto lastTick = std::chrono::steady_clock::now();

    while (true) {
        Mat frame;
        if (!cap.read(frame) || frame.empty()) {
            cerr << "[WARN] Empty frame." << endl;
            continue;
        }

        // 分类模式：每 0.5s 执行一次
        if (mode == 1) {
            auto now = std::chrono::steady_clock::now();
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastTick).count();
            if (ms >= 500) {
                lastTick = now;

                vector<float> q = computeEmbedding(frame, net, featureLayer);
                if (!store.empty()) {
                    auto [pred, dist] = store.predict(q);
                    // 在图像左上角显示
                    string msg = "Pred: " + pred + " (dist=" + to_string(dist).substr(0,8) + ")";
                    putText(frame, msg, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0,255,0), 2, LINE_AA);
                    // 终端打印
                    cout << "[CLASSIFY] " << msg << endl;
                } else {
                    string msg = "No samples. Press M to switch to Train.";
                    putText(frame, msg, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0,255,255), 2, LINE_AA);
                    cout << "[CLASSIFY] " << msg << endl;
                }
            }
        } else {
            // 训练模式也在画面上给一点提示
            putText(frame, "TRAIN MODE: Press N to capture sample", Point(10,30),
                    FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255,255,0), 2, LINE_AA);
        }

        imshow(winName, frame);

        int key = waitKeyEx(1);
        key &= 0xFF; // 只保留低 8 位，避免特殊键值干扰
        if (key == 'q' || key == 'Q') {
            cout << "[EXIT] Quit." << endl;
            break;
        } else if (key == 'm' || key == 'M') {
            mode = 1 - mode;
            cout << (mode==0 ? "[MODE] TRAIN" : "[MODE] CLASSIFY") << endl;
        } else if (mode == 0 && (key == 'n' || key == 'N')) {
            // 训练采样
            Mat snap = frame.clone();
            // 计算 embedding
            vector<float> feat = computeEmbedding(snap, net, featureLayer);
            if (feat.empty()) {
                cerr << "[ERROR] Empty embedding. Skip." << endl;
                continue;
            }
            // 控制台输入标签
            cout << "[TRAIN] Enter label for this sample: ";
            string label;
            // 注意：保证控制台有焦点时输入；若在 IDE 里，先点击控制台
            std::getline(cin, label);
            if (label.empty()) {
                // 如果上一行被空读了，再读一次
                std::getline(cin, label);
            }
            if (label.empty()) {
                cout << "[TRAIN] Empty label. Skipped." << endl;
                continue;
            }
            // 保存 embedding
            if (!store.append(dbPath, label, feat)) {
                cerr << "[ERROR] Failed to append embedding to " << dbPath << endl;
            } else {
                // 保存图片
                auto tnow = std::chrono::system_clock::now();
                auto tsecs = std::chrono::duration_cast<std::chrono::seconds>(tnow.time_since_epoch()).count();
                string imgPath = imgDir + "/" + label + "_" + to_string(tsecs) + ".jpg";
                imwrite(imgPath, snap);

                // 视觉提示
                Mat disp = snap.clone();
                putText(disp, "Saved: " + label, Point(10,30), FONT_HERSHEY_SIMPLEX, 0.9, Scalar(0,200,255), 2, LINE_AA);
                imshow(winName, disp);
                cout << "[TRAIN] Saved label=" << label << "  (embedding_dim=" << feat.size()
                     << ")  -> " << dbPath << "  and image -> " << imgPath << endl;
                // 小停顿以便看到提示
                waitKey(300);
            }
        }
    }

    destroyAllWindows();
    return 0;
}
