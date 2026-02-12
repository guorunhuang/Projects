#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <fstream>

class FullCalibrationSystem {
private:
    cv::Size boardSize;
    float squareSize;
    
    std::vector<std::vector<cv::Point2f>> corner_list;
    std::vector<std::vector<cv::Vec3f>> point_list;
    std::vector<cv::Mat> calibration_images;
    std::vector<cv::Mat> rvecs_saved;
    std::vector<cv::Mat> tvecs_saved;
    
    std::vector<cv::Point2f> current_corners;
    cv::Mat current_frame;
    bool corners_found;
    
    cv::Mat camera_matrix;
    cv::Mat dist_coeffs;
    bool is_calibrated;
    
    double last_reprojection_error;
    bool auto_calibrate;
    int min_calibration_images;
    
public:
    FullCalibrationSystem(int cols = 9, int rows = 6, float sq_size = 1.0f) 
        : boardSize(cols, rows), squareSize(sq_size), 
          corners_found(false), is_calibrated(false),
          last_reprojection_error(0.0), auto_calibrate(false),
          min_calibration_images(5) {
        
        // Initialize camera matrix with reasonable defaults
        camera_matrix = cv::Mat::eye(3, 3, CV_64FC1);
        
        // Will be set properly when first frame arrives
        camera_matrix.at<double>(0, 0) = 1.0; // fx
        camera_matrix.at<double>(1, 1) = 1.0; // fy
        camera_matrix.at<double>(0, 2) = 0.0; // cx (will update)
        camera_matrix.at<double>(1, 2) = 0.0; // cy (will update)
        
        // Initialize distortion coefficients
        // Length 0 = no distortion
        // Length 4 = k1, k2, p1, p2
        // Length 5 = k1, k2, p1, p2, k3
        // Length 8 = k1, k2, p1, p2, k3, k4, k5, k6
        dist_coeffs = cv::Mat::zeros(5, 1, CV_64FC1); // Start with 5 parameters
    }
    
    void initializeCameraMatrix(cv::Size image_size) {
        if (camera_matrix.at<double>(0, 2) == 0.0) {
            camera_matrix.at<double>(0, 0) = 1.0;
            camera_matrix.at<double>(1, 1) = 1.0;
            camera_matrix.at<double>(0, 2) = image_size.width / 2.0;
            camera_matrix.at<double>(1, 2) = image_size.height / 2.0;
            
            std::cout << "\n[INIT] Camera matrix initialized:" << std::endl;
            std::cout << camera_matrix << std::endl;
        }
    }
    
    bool detectCorners(cv::Mat& frame) {
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        
        current_corners.clear();
        corners_found = cv::findChessboardCorners(
            gray, 
            boardSize, 
            current_corners,
            cv::CALIB_CB_ADAPTIVE_THRESH | 
            cv::CALIB_CB_NORMALIZE_IMAGE |
            cv::CALIB_CB_FAST_CHECK
        );
        
        if (corners_found) {
            cv::cornerSubPix(
                gray, 
                current_corners,
                cv::Size(11, 11),
                cv::Size(-1, -1),
                cv::TermCriteria(
                    cv::TermCriteria::EPS + cv::TermCriteria::COUNT,
                    30, 
                    0.1
                )
            );
            
            current_frame = frame.clone();
            
            std::cout << "\r[DETECT] Corners: " << current_corners.size() 
                      << " | First: (" << std::fixed << std::setprecision(1)
                      << current_corners[0].x << ", " << current_corners[0].y << ")"
                      << std::flush;
        }
        
        return corners_found;
    }
    
    void drawCorners(cv::Mat& frame) {
        if (corners_found && !current_corners.empty()) {
            cv::drawChessboardCorners(frame, boardSize, current_corners, corners_found);
            
            // Highlight origin
            cv::circle(frame, current_corners[0], 10, cv::Scalar(0, 0, 255), 2);
            cv::putText(frame, "Origin", 
                       cv::Point(current_corners[0].x + 12, current_corners[0].y - 5),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
        }
    }
    
    std::vector<cv::Vec3f> generate3DPoints() {
        std::vector<cv::Vec3f> points;
        
        // World coordinates: Z=0 plane
        // X increases to the right: 0, 1, 2, ...
        // Y increases downward in image, but we use negative Y in world coords
        for (int i = 0; i < boardSize.height; i++) {
            for (int j = 0; j < boardSize.width; j++) {
                points.push_back(cv::Vec3f(
                    j * squareSize,      // X: 0, 1, 2, 3, ..., 8
                    -i * squareSize,     // Y: 0, -1, -2, -3, ..., -5
                    0.0f                 // Z: 0 (all points on same plane)
                ));
            }
        }
        
        return points;
    }
    
    bool saveCalibrationImage() {
        if (!corners_found || current_corners.empty()) {
            std::cout << "\n[ERROR] No valid corners to save!" << std::endl;
            return false;
        }
        
        corner_list.push_back(current_corners);
        point_list.push_back(generate3DPoints());
        calibration_images.push_back(current_frame.clone());
        
        std::cout << "\n[SAVED] Image #" << corner_list.size() << std::endl;
        
        // Auto-calibrate if enabled and we have enough images
        if (auto_calibrate && corner_list.size() >= min_calibration_images) {
            std::cout << "[AUTO] Running calibration..." << std::endl;
            calibrateCamera(current_frame.size());
        }
        
        return true;
    }
    
    double computeReprojectionError(
        const std::vector<std::vector<cv::Vec3f>>& object_points,
        const std::vector<std::vector<cv::Point2f>>& image_points,
        const std::vector<cv::Mat>& rvecs,
        const std::vector<cv::Mat>& tvecs) {
        
        std::vector<cv::Point2f> projected_points;
        double total_error = 0.0;
        int total_points = 0;
        
        for (size_t i = 0; i < object_points.size(); i++) {
            cv::projectPoints(object_points[i], rvecs[i], tvecs[i],
                            camera_matrix, dist_coeffs, projected_points);
            
            double error = cv::norm(image_points[i], projected_points, cv::NORM_L2);
            total_error += error * error;
            total_points += object_points[i].size();
        }
        
        return std::sqrt(total_error / total_points);
    }
    
    bool calibrateCamera(cv::Size image_size) {
        if (corner_list.size() < min_calibration_images) {
            std::cout << "\n[ERROR] Need at least " << min_calibration_images 
                      << " images. Have: " << corner_list.size() << std::endl;
            return false;
        }
        
        initializeCameraMatrix(image_size);
        
        std::cout << "\n" << std::string(70, '=') << std::endl;
        std::cout << "CAMERA CALIBRATION" << std::endl;
        std::cout << std::string(70, '=') << std::endl;
        std::cout << "Using " << corner_list.size() << " calibration images" << std::endl;
        std::cout << "Image size: " << image_size.width << "x" << image_size.height << std::endl;
        
        std::cout << "\nCamera Matrix BEFORE calibration:" << std::endl;
        std::cout << camera_matrix << std::endl;
        
        std::cout << "\nDistortion Coefficients BEFORE calibration:" << std::endl;
        std::cout << dist_coeffs.t() << std::endl;
        
        rvecs_saved.clear();
        tvecs_saved.clear();
        
        // Calibration flags
        int flags = cv::CALIB_FIX_ASPECT_RATIO;
        // Optional: Add cv::CALIB_ZERO_TANGENT_DIST if you want to skip p1, p2
        
        double rms_error = cv::calibrateCamera(
            point_list,
            corner_list,
            image_size,
            camera_matrix,
            dist_coeffs,
            rvecs_saved,
            tvecs_saved,
            flags
        );
        
        std::cout << "\n" << std::string(70, '-') << std::endl;
        std::cout << "CALIBRATION RESULTS" << std::endl;
        std::cout << std::string(70, '-') << std::endl;
        
        std::cout << "\nCamera Matrix AFTER calibration:" << std::endl;
        std::cout << camera_matrix << std::endl;
        
        std::cout << "\nCamera Parameters:" << std::endl;
        std::cout << "  fx (focal length X): " << camera_matrix.at<double>(0, 0) << " pixels" << std::endl;
        std::cout << "  fy (focal length Y): " << camera_matrix.at<double>(1, 1) << " pixels" << std::endl;
        std::cout << "  cx (principal point X): " << camera_matrix.at<double>(0, 2) << " pixels" << std::endl;
        std::cout << "  cy (principal point Y): " << camera_matrix.at<double>(1, 2) << " pixels" << std::endl;
        
        std::cout << "\nDistortion Coefficients AFTER calibration:" << std::endl;
        std::cout << dist_coeffs.t() << std::endl;
        std::cout << "  k1 (radial 1): " << dist_coeffs.at<double>(0) << std::endl;
        std::cout << "  k2 (radial 2): " << dist_coeffs.at<double>(1) << std::endl;
        std::cout << "  p1 (tangential 1): " << dist_coeffs.at<double>(2) << std::endl;
        std::cout << "  p2 (tangential 2): " << dist_coeffs.at<double>(3) << std::endl;
        std::cout << "  k3 (radial 3): " << dist_coeffs.at<double>(4) << std::endl;
        
        std::cout << "\nRMS Re-projection Error: " << std::fixed << std::setprecision(4) 
                  << rms_error << " pixels" << std::endl;
        
        // Evaluate quality
        std::string quality;
        if (rms_error < 0.5) quality = "EXCELLENT";
        else if (rms_error < 1.0) quality = "GOOD";
        else if (rms_error < 2.0) quality = "ACCEPTABLE";
        else quality = "POOR - Consider recalibrating";
        
        std::cout << "Quality Assessment: " << quality << std::endl;
        
        // Per-image errors
        std::cout << "\nPer-Image Re-projection Errors:" << std::endl;
        for (size_t i = 0; i < corner_list.size(); i++) {
            std::vector<cv::Point2f> projected;
            cv::projectPoints(point_list[i], rvecs_saved[i], tvecs_saved[i],
                            camera_matrix, dist_coeffs, projected);
            
            double error = 0.0;
            for (size_t j = 0; j < corner_list[i].size(); j++) {
                double dx = corner_list[i][j].x - projected[j].x;
                double dy = corner_list[i][j].y - projected[j].y;
                error += std::sqrt(dx*dx + dy*dy);
            }
            error /= corner_list[i].size();
            
            std::cout << "  Image " << std::setw(2) << (i+1) << ": " 
                      << std::fixed << std::setprecision(3) << error << " pixels";
            if (error > rms_error * 1.5) std::cout << " [HIGH]";
            std::cout << std::endl;
        }
        
        std::cout << std::string(70, '=') << std::endl << std::endl;
        
        last_reprojection_error = rms_error;
        is_calibrated = true;
        
        return true;
    }
    
    void saveCalibration(const std::string& filename) {
        cv::FileStorage fs(filename, cv::FileStorage::WRITE);
        
        time_t now = time(0);
        char* dt = ctime(&now);
        
        fs << "calibration_date" << dt;
        fs << "num_images" << (int)corner_list.size();
        fs << "image_width" << current_frame.cols;
        fs << "image_height" << current_frame.rows;
        fs << "board_width" << boardSize.width;
        fs << "board_height" << boardSize.height;
        fs << "square_size" << squareSize;
        fs << "rms_error" << last_reprojection_error;
        fs << "camera_matrix" << camera_matrix;
        fs << "distortion_coefficients" << dist_coeffs;
        
        // Save rotations and translations for each calibration image
        fs << "num_calibration_poses" << (int)rvecs_saved.size();
        
        fs << "rotation_vectors" << "[";
        for (size_t i = 0; i < rvecs_saved.size(); i++) {
            fs << rvecs_saved[i];
        }
        fs << "]";
        
        fs << "translation_vectors" << "[";
        for (size_t i = 0; i < tvecs_saved.size(); i++) {
            fs << tvecs_saved[i];
        }
        fs << "]";
        
        fs.release();
        
        std::cout << "[SAVED] Calibration data written to " << filename << std::endl;
        std::cout << "        Includes camera matrix, distortion coefficients," << std::endl;
        std::cout << "        and " << rvecs_saved.size() << " pose estimates" << std::endl;
    }
    
    bool loadCalibration(const std::string& filename) {
        cv::FileStorage fs(filename, cv::FileStorage::READ);
        if (!fs.isOpened()) {
            return false;
        }
        
        fs["camera_matrix"] >> camera_matrix;
        fs["distortion_coefficients"] >> dist_coeffs;
        
        std::string date;
        int num_images;
        double rms;
        
        fs["calibration_date"] >> date;
        fs["num_images"] >> num_images;
        fs["rms_error"] >> rms;
        
        last_reprojection_error = rms;
        is_calibrated = true;
        
        fs.release();
        
        std::cout << "\n[LOADED] Calibration from " << filename << std::endl;
        std::cout << "Date: " << date;
        std::cout << "Based on " << num_images << " images" << std::endl;
        std::cout << "RMS error: " << rms << " pixels" << std::endl;
        
        return true;
    }
    
    void saveImageWithCorners(int index, const std::string& filename) {
        if (index < 0 || index >= (int)calibration_images.size()) {
            return;
        }
        
        cv::Mat img = calibration_images[index].clone();
        cv::drawChessboardCorners(img, boardSize, corner_list[index], true);
        
        // Add text overlay
        std::string text = "Calibration Image " + std::to_string(index + 1);
        cv::putText(img, text, cv::Point(10, 30),
                   cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
        
        cv::imwrite(filename, img);
        std::cout << "[SAVED] " << filename << std::endl;
    }
    
    void setAutoCalibrate(bool enable) { auto_calibrate = enable; }
    void setMinImages(int min) { min_calibration_images = min; }
    int getCalibrationCount() const { return corner_list.size(); }
    bool isCalibrated() const { return is_calibrated; }
    bool hasCorners() const { return corners_found; }
    double getReprojectionError() const { return last_reprojection_error; }
    cv::Mat getCameraMatrix() const { return camera_matrix; }
    cv::Mat getDistCoeffs() const { return dist_coeffs; }
};

int main() {
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "[ERROR] Cannot open camera" << std::endl;
        return -1;
    }
    
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    
    FullCalibrationSystem calib(9, 6, 1.0f);
    calib.loadCalibration("calibration.xml");
    
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "CAMERA CALIBRATION SYSTEM" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    std::cout << "CONTROLS:" << std::endl;
    std::cout << "  's' - Save current frame for calibration" << std::endl;
    std::cout << "  'c' - Calibrate camera (manual)" << std::endl;
    std::cout << "  'a' - Toggle auto-calibration mode" << std::endl;
    std::cout << "  'w' - Write calibration to file" << std::endl;
    std::cout << "  'q' - Quit" << std::endl;
    std::cout << std::string(70, '=') << std::endl << std::endl;
    
    cv::Mat frame;
    bool auto_mode = false;
    
    while (true) {
        cap >> frame;
        if (frame.empty()) break;
        
        calib.detectCorners(frame);
        calib.drawCorners(frame);
        
        // Status overlay
        std::string status = "Images: " + std::to_string(calib.getCalibrationCount());
        cv::putText(frame, status, cv::Point(10, 30),
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        
        if (calib.isCalibrated()) {
            std::string calib_text = "CALIBRATED | Error: " + 
                                    std::to_string(calib.getReprojectionError()).substr(0, 5) + " px";
            cv::putText(frame, calib_text, cv::Point(10, 60),
                       cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        }
        
        if (auto_mode) {
            cv::putText(frame, "AUTO MODE", cv::Point(10, 90),
                       cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 100, 0), 2);
        }
        
        cv::imshow("Calibration", frame);
        
        char key = cv::waitKey(500);
        if (key == 'q' || key == 27) break;
        else if (key == 's') {
            if (calib.saveCalibrationImage()) {
                int idx = calib.getCalibrationCount() - 1;
                calib.saveImageWithCorners(idx, "calib_img_" + std::to_string(idx) + ".png");
            }
        } else if (key == 'c') {
            calib.calibrateCamera(frame.size());
        } else if (key == 'a') {
            auto_mode = !auto_mode;
            calib.setAutoCalibrate(auto_mode);
            std::cout << "[MODE] Auto-calibration: " << (auto_mode ? "ON" : "OFF") << std::endl;
        } else if (key == 'w') {
            if (calib.isCalibrated()) {
                calib.saveCalibration("calibration.xml");
            }
        }
    }
    
    cap.release();
    cv::destroyAllWindows();
    return 0;
}