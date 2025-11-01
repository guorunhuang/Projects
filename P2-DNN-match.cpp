/* 
功能: 从二进制索引文件加载图像特征（每条记录为 uint32 filename_length + filename bytes + 512 floats），
查找目标图像的特征，使用 SSD 或 Cosine 两种度量计算与数据库中所有向量的相似度/距离，排序并输出 Top-N 结果，同时把结果写入 match_result.txt。
Load binary index of (filename + 512-dim float feature), locate target feature (by exact filename or basename), 
compute distances using either SSD or cosine distance, sort ascending, print and save top-N matches.

Parameters to Pass 运行时需传入 4 个位置参数，顺序固定： 参数顺序 参数含义 示例值 
第 1 个 索引文件路径 ./features.idx 第 2 个 目标文件名或路径片段 target.jpg 第 3 个 Top-N 数量 10 第 4 个 度量方式 ssd 或 cosine

Usage: match <index_file> <target_filename> <topN> <metric> 
Example: match features.idx target.jpg 10 ssd
*/

#include <bits/stdc++.h>
using namespace std;

struct Entry {
    string filename;
    vector<float> feat; // length 512
};

bool load_index(const string& idx_path, vector<Entry>& out_entries) {
    ifstream ifs(idx_path, ios::binary);
    if (!ifs.is_open()) return false;
    while (ifs.peek() != EOF) {
        uint32_t fn_len;
        ifs.read(reinterpret_cast<char*>(&fn_len), sizeof(fn_len));
        if (!ifs) break;
        string fname(fn_len, '\0');
        ifs.read(&fname[0], fn_len);
        vector<float> feat(512);
        ifs.read(reinterpret_cast<char*>(feat.data()), sizeof(float) * 512);
        if (!ifs) break;
        out_entries.push_back({fname, move(feat)});
    }
    ifs.close();
    return true;
}

double ssd_distance(const vector<float>& a, const vector<float>& b) {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double d = double(a[i]) - double(b[i]);
        sum += d * d;
    }
    return sum;
}

double cosine_distance(const vector<float>& a, const vector<float>& b) {
    // compute dot and norms
    double dot = 0.0;
    double na = 0.0, nb = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += double(a[i]) * double(b[i]);
        na += double(a[i]) * double(a[i]);
        nb += double(b[i]) * double(b[i]);
    }
    if (na <= 0 || nb <= 0) return 1.0; // fallback
    double cosv = dot / (sqrt(na) * sqrt(nb));
    // numeric stability clamp
    if (cosv > 1.0) cosv = 1.0;
    if (cosv < -1.0) cosv = -1.0;
    return 1.0 - cosv; // cosine distance
}

int main(int argc, char** argv) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    if (argc < 5) {
        cerr << "Usage): match <index_file> <target_filename> <topN> <metric>\n";
        cerr << "Example): match features.idx target.jpg 10 ssd\n";
        return 1;
    }
    string idx = argv[1];
    string target = argv[2];
    int topN = stoi(argv[3]);
    string metric = argv[4];
    for (auto & c: metric) c = tolower(c);

    vector<Entry> db;
    if (!load_index(idx, db)) {
        cerr << "Cannot load index): " << idx << "\n";
        return 1;
    }
    cout << db.size() << " Loaded " << db.size() << " features).\n";

    // 找到 target 向量
    int target_idx = -1;
    for (size_t i = 0; i < db.size(); ++i) {
        if (db[i].filename == target) {
            target_idx = int(i);
            break;
        }
    }
    if (target_idx == -1) {
        // 也尝试绝对路径或文件名匹配的简单后缀
        for (size_t i = 0; i < db.size(); ++i) {
            // compare basename
            string base = db[i].filename;
            auto pos = base.find_last_of("/\\");
            if (pos != string::npos) base = base.substr(pos+1);
            if (base == target) { target_idx = int(i); break; }
        }
    }

    if (target_idx == -1) {
        cerr << "Error(target not found in index): " << target << "\n";
        return 1;
    }

    const auto& tfeat = db[target_idx].feat;
    vector<pair<double, string>> scores;
    scores.reserve(db.size());

    for (size_t i = 0; i < db.size(); ++i) {
        const auto &e = db[i];
        double d;
        if (metric == "ssd") d = ssd_distance(tfeat, e.feat);
        else d = cosine_distance(tfeat, e.feat);
        scores.emplace_back(d, e.filename);
    }

    sort(scores.begin(), scores.end(), [](const auto& a, const auto& b){
        if (a.first != b.first) return a.first < b.first;
        return a.second < b.second;
    });

    cout << "Top-" << topN << " matches):\n";
    for (int i = 0; i < min(topN, (int)scores.size()); ++i) {
        cout << i+1 << ". " << scores[i].second << "    distance): " << scores[i].first << "\n";
    }

    // 写结果到文件
    string outtxt = "match_result.txt";
    ofstream ofs(outtxt);
    ofs << "Target: " << target << "\n";
    ofs << "Metric: " << metric << "\n";
    ofs << "Top-" << topN << " matches:\n";
    for (int i = 0; i < min(topN, (int)scores.size()); ++i) {
        ofs << i+1 << "\t" << scores[i].second << "\t" << scores[i].first << "\n";
    }
    ofs.close();
    cout  << outtxt << " (results written).\n";
    return 0;
}
