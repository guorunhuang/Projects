/*
xfeatures2d 模块不在OpenCV的主库中，而是在 opencv_contrib 扩展模块里,所以报错了
*/
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <vector>

class FeatureDetector {
public:
    enum DetectorType { HARRIS, GFTT, FAST, ORB, SIFT}; //main()无法访问类内部 private 的枚举成员，所以改成了public、
private:
    // enum DetectorType { HARRIS, GFTT, FAST, ORB, SIFT}; //另一种解决方法是提供 public setter
    DetectorType current_detector;
    
    // Harris parameters
    int harris_block_size = 2;
    int harris_aperture = 3;
    double harris_k = 0.04;
    double harris_threshold = 200.0;
    
    // GFTT parameters
    int gftt_max_corners = 100;
    double gftt_quality = 0.01;
    double gftt_min_distance = 10;
    
    // FAST parameters
    int fast_threshold = 20;
    
    // ORB parameters
    int orb_features = 500;
    
    // SIFT parameters
    int sift_features = 100;
    double sift_contrast = 0.04;
    
    cv::Mat reference_image;
    std::vector<cv::KeyPoint> reference_keypoints;
    cv::Mat reference_descriptors;
    bool has_reference = false;
    
public:
    FeatureDetector() : current_detector(HARRIS) {}
    
    void detectHarrisCorners(const cv::Mat& gray, std::vector<cv::KeyPoint>& keypoints) {
        cv::Mat dst, dst_norm;
        
        // Detect Harris corners
        cv::cornerHarris(gray, dst, harris_block_size, harris_aperture, harris_k);
        cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1);
        
        // Convert to keypoints
        keypoints.clear();
        for (int i = 0; i < dst_norm.rows; i++) {
            for (int j = 0; j < dst_norm.cols; j++) {
                if (dst_norm.at<float>(i, j) > harris_threshold) {
                    keypoints.push_back(cv::KeyPoint(j, i, 5));
                }
            }
        }
        
        std::cout << "\r[HARRIS] Corners: " << keypoints.size() 
                  << " | Threshold: " << harris_threshold << "   " << std::flush;
    }
    
    void detectGFTT(const cv::Mat& gray, std::vector<cv::KeyPoint>& keypoints) {
        std::vector<cv::Point2f> corners;
        
        cv::goodFeaturesToTrack(gray, corners, gftt_max_corners, 
                               gftt_quality, gftt_min_distance);
        
        keypoints.clear();
        for (const auto& corner : corners) {
            keypoints.push_back(cv::KeyPoint(corner, 5));
        }
        
        std::cout << "\r[GFTT] Corners: " << keypoints.size() 
                  << " | Quality: " << gftt_quality 
                  << " | MinDist: " << gftt_min_distance << "   " << std::flush;
    }
    
    void detectFAST(const cv::Mat& gray, std::vector<cv::KeyPoint>& keypoints) {
        cv::Ptr<cv::FastFeatureDetector> detector = 
            cv::FastFeatureDetector::create(fast_threshold);
        
        detector->detect(gray, keypoints);
        
        std::cout << "\r[FAST] Features: " << keypoints.size() 
                  << " | Threshold: " << fast_threshold << "   " << std::flush;
    }
    
    void detectORB(const cv::Mat& gray, std::vector<cv::KeyPoint>& keypoints,
                   cv::Mat& descriptors) {
        cv::Ptr<cv::ORB> detector = cv::ORB::create(orb_features);
        
        detector->detectAndCompute(gray, cv::noArray(), keypoints, descriptors);
        
        std::cout << "\r[ORB] Features: " << keypoints.size() 
                  << " | Max: " << orb_features << "   " << std::flush;
    }
    
    void detectSIFT(const cv::Mat& gray, std::vector<cv::KeyPoint>& keypoints,
                    cv::Mat& descriptors) {
        cv::Ptr<cv::SIFT> detector = cv::SIFT::create(
            sift_features, 3, sift_contrast);
        
        detector->detectAndCompute(gray, cv::noArray(), keypoints, descriptors);
        
        std::cout << "\r[SIFT] Features: " << keypoints.size() 
                  << " | Contrast: " << sift_contrast << "   " << std::flush;
    }
    
    
    void detectFeatures(const cv::Mat& frame, std::vector<cv::KeyPoint>& keypoints,
                       cv::Mat& descriptors) {
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        
        keypoints.clear();
        descriptors = cv::Mat();
        
        switch (current_detector) {
            case HARRIS:
                detectHarrisCorners(gray, keypoints);
                break;
            case GFTT:
                detectGFTT(gray, keypoints);
                break;
            case FAST:
                detectFAST(gray, keypoints);
                break;
            case ORB:
                detectORB(gray, keypoints, descriptors);
                break;
            case SIFT:
                detectSIFT(gray, keypoints, descriptors);
                break;
        }
    }
    
    void drawFeatures(cv::Mat& frame, const std::vector<cv::KeyPoint>& keypoints) {
        // Draw keypoints with different colors based on response
        for (const auto& kp : keypoints) {
            // Color based on response strength
            int intensity = std::min(255, (int)(kp.response * 100));
            cv::Scalar color(0, 255 - intensity/2, intensity);
            
            // Draw circle
            cv::circle(frame, kp.pt, 3, color, -1);
            cv::circle(frame, kp.pt, 5, cv::Scalar(0, 255, 0), 1);
            
            // Draw orientation if available
            if (kp.angle >= 0) {
                float angle_rad = kp.angle * CV_PI / 180.0f;
                cv::Point2f end(kp.pt.x + 10 * std::cos(angle_rad),
                               kp.pt.y + 10 * std::sin(angle_rad));
                cv::line(frame, kp.pt, end, cv::Scalar(255, 0, 0), 1);
            }
        }
    }
    
    void saveReference(const cv::Mat& frame, const std::vector<cv::KeyPoint>& keypoints,
                      const cv::Mat& descriptors) {
        reference_image = frame.clone();
        reference_keypoints = keypoints;
        reference_descriptors = descriptors.clone();
        has_reference = true;
        
        std::cout << "\n[SAVED] Reference pattern with " << keypoints.size() 
                  << " features" << std::endl;
    }
    
    void matchAndDraw(cv::Mat& frame, const std::vector<cv::KeyPoint>& query_keypoints,
                     const cv::Mat& query_descriptors) {
        if (!has_reference || query_descriptors.empty() || reference_descriptors.empty()) {
            return;
        }
        
        // Match features
        cv::Ptr<cv::DescriptorMatcher> matcher;
        std::vector<std::vector<cv::DMatch>> knn_matches;
        
        if (current_detector == ORB) {
            matcher = cv::BFMatcher::create(cv::NORM_HAMMING);
        } else {
            matcher = cv::BFMatcher::create(cv::NORM_L2);
        }
        
        matcher->knnMatch(query_descriptors, reference_descriptors, knn_matches, 2);
        
        // Apply ratio test (Lowe's ratio test)
        std::vector<cv::DMatch> good_matches;
        for (const auto& match : knn_matches) {
            if (match.size() >= 2 && match[0].distance < 0.7f * match[1].distance) {
                good_matches.push_back(match[0]);
            }
        }
        
        // Draw matches
        for (const auto& match : good_matches) {
            cv::Point2f pt = query_keypoints[match.queryIdx].pt;
            cv::circle(frame, pt, 5, cv::Scalar(0, 255, 255), 2);
            cv::line(frame, pt, pt + cv::Point2f(10, 10), 
                    cv::Scalar(0, 255, 255), 2);
        }
        
        // Display match count
        std::string match_text = "Matches: " + std::to_string(good_matches.size());
        cv::putText(frame, match_text, cv::Point(10, frame.rows - 20),
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
        
        // If enough matches, try to find homography
        if (good_matches.size() >= 4) {
            std::vector<cv::Point2f> src_pts, dst_pts;
            for (const auto& match : good_matches) {
                src_pts.push_back(query_keypoints[match.queryIdx].pt);
                dst_pts.push_back(reference_keypoints[match.trainIdx].pt);
            }
            
            cv::Mat H = cv::findHomography(src_pts, dst_pts, cv::RANSAC);
            
            if (!H.empty()) {
                // Draw bounding box
                std::vector<cv::Point2f> ref_corners = {
                    {0, 0},
                    {(float)reference_image.cols, 0},
                    {(float)reference_image.cols, (float)reference_image.rows},
                    {0, (float)reference_image.rows}
                };
                
                std::vector<cv::Point2f> scene_corners(4);
                cv::perspectiveTransform(ref_corners, scene_corners, H.inv());
                
                for (size_t i = 0; i < 4; i++) {
                    cv::line(frame, scene_corners[i], scene_corners[(i+1)%4],
                            cv::Scalar(0, 255, 0), 3);
                }
                
                cv::putText(frame, "Pattern Detected!", cv::Point(10, 90),
                           cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
            }
        }
    }
    
    void adjustParameters(char key) {
        switch (key) {
            // Harris adjustments
            case 'w': harris_threshold += 10; break;
            case 's': harris_threshold = std::max(10.0, harris_threshold - 10); break;
            
            // GFTT adjustments
            case 'e': gftt_quality += 0.01; break;
            case 'd': gftt_quality = std::max(0.001, gftt_quality - 0.01); break;
            case 'r': gftt_min_distance += 1; break;
            case 'f': gftt_min_distance = std::max(1.0, gftt_min_distance - 1); break;
            
            // FAST adjustments
            case 't': fast_threshold += 5; break;
            case 'g': fast_threshold = std::max(5, fast_threshold - 5); break;
            
            // ORB adjustments
            case 'y': orb_features += 100; break;
            case 'h': orb_features = std::max(100, orb_features - 100); break;
            
            // SIFT adjustments
            case 'u': sift_contrast += 0.01; break;
            case 'j': sift_contrast = std::max(0.01, sift_contrast - 0.01); break;
        }
    }
    
    void setDetector(DetectorType type) {
        current_detector = type;
        std::string names[] = {"Harris", "GFTT", "FAST", "ORB", "SIFT", "SURF"};
        std::cout << "\n[DETECTOR] Switched to: " << names[type] << std::endl;
    }
    
    DetectorType getCurrentDetector() const { return current_detector; }
    
    void printHelp() {
        std::cout << "\n" << std::string(70, '=') << std::endl;
        std::cout << "ROBUST FEATURE DETECTION" << std::endl;
        std::cout << std::string(70, '=') << std::endl;
        std::cout << "\nDetector Selection:" << std::endl;
        std::cout << "  '1' - Harris Corner Detector" << std::endl;
        std::cout << "  '2' - Good Features To Track (GFTT)" << std::endl;
        std::cout << "  '3' - FAST Features" << std::endl;
        std::cout << "  '4' - ORB Features" << std::endl;
        std::cout << "  '5' - SIFT Features" << std::endl;
        
        std::cout << "\nParameter Adjustments:" << std::endl;
        std::cout << "  Harris: 'w'/'s' - Increase/Decrease threshold" << std::endl;
        std::cout << "  GFTT: 'e'/'d' - Adjust quality level" << std::endl;
        std::cout << "        'r'/'f' - Adjust min distance" << std::endl;
        std::cout << "  FAST: 't'/'g' - Adjust threshold" << std::endl;
        std::cout << "  ORB: 'y'/'h' - Adjust max features" << std::endl;
        std::cout << "  SIFT: 'u'/'j' - Adjust contrast threshold" << std::endl;
        
        std::cout << "\nOther Controls:" << std::endl;
        std::cout << "  'p' - Save current frame as reference pattern" << std::endl;
        std::cout << "  'm' - Toggle matching mode (if reference saved)" << std::endl;
        std::cout << "  'c' - Capture screenshot" << std::endl;
        std::cout << "  'q' - Quit" << std::endl;
        std::cout << std::string(70, '=') << std::endl << std::endl;
    }
};

int main() {
    FeatureDetector detector;
    detector.printHelp();
    
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open camera" << std::endl;
        return -1;
    }
    
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    
    cv::Mat frame;
    bool matching_mode = false;
    int screenshot_count = 0;
    
    std::string detector_names[] = {"Harris", "GFTT", "FAST", "ORB", "SIFT", "SURF"};
    
    while (true) {
        cap >> frame;
        if (frame.empty()) break;
        
        cv::Mat display = frame.clone();
        
        // Detect features
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        detector.detectFeatures(display, keypoints, descriptors);
        
        // Draw features
        detector.drawFeatures(display, keypoints);
        
        // If in matching mode, match with reference
        if (matching_mode) {
            detector.matchAndDraw(display, keypoints, descriptors);
        }
        
        // Display info
        std::string detector_text = "Detector: " + 
                                   detector_names[detector.getCurrentDetector()];
        cv::putText(display, detector_text, cv::Point(10, 30),
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
        
        std::string feature_count = "Features: " + std::to_string(keypoints.size());
        cv::putText(display, feature_count, cv::Point(10, 60),
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
        
        if (matching_mode) {
            cv::putText(display, "MATCHING MODE", cv::Point(10, display.rows - 50),
                       cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
        }
        
        cv::imshow("Feature Detection", display);
        
        char key = cv::waitKey(500);
        if (key == 'q' || key == 27) {
            break;
        } else if (key == '1') {
            detector.setDetector(FeatureDetector::HARRIS);
        } else if (key == '2') {
            detector.setDetector(FeatureDetector::GFTT);
        } else if (key == '3') {
            detector.setDetector(FeatureDetector::FAST);
        } else if (key == '4') {
            detector.setDetector(FeatureDetector::ORB);
        } else if (key == '5') {
            detector.setDetector(FeatureDetector::SIFT);
        } else if (key == 'p') {
            detector.saveReference(frame, keypoints, descriptors);
        } else if (key == 'm') {
            matching_mode = !matching_mode;
            std::cout << "\n[MODE] Matching: " << (matching_mode ? "ON" : "OFF") << std::endl;
        } else if (key == 's') {
            std::string filename = "AR_features_screenshot_" + 
                                  std::to_string(screenshot_count++) + ".png";
            cv::imwrite(filename, display);
            std::cout << "\n[SAVED] " << filename << std::endl;
        } else {
            detector.adjustParameters(key);
        }
    }
    
    std::cout << std::endl;
    cap.release();
    cv::destroyAllWindows();
    
    return 0;
}