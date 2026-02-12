#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>
#include <string>

class CalibrationSystem {
private:
    // Checkerboard dimensions (internal corners)
    cv::Size boardSize;
    float squareSize;
    
    // Storage for calibration data
    std::vector<std::vector<cv::Point2f>> corner_list;
    std::vector<std::vector<cv::Vec3f>> point_list;
    std::vector<cv::Mat> calibration_images;
    
    // Most recent detection
    std::vector<cv::Point2f> current_corners;
    cv::Mat current_frame;
    bool corners_found;
    
    // Camera parameters (after calibration)
    cv::Mat camera_matrix;
    cv::Mat dist_coeffs;
    bool is_calibrated;
    
public:
    CalibrationSystem(int cols = 9, int rows = 6, float sq_size = 1.0f) 
        : boardSize(cols, rows), squareSize(sq_size), 
          corners_found(false), is_calibrated(false) {
        camera_matrix = cv::Mat::eye(3, 3, CV_64F);
        dist_coeffs = cv::Mat::zeros(8, 1, CV_64F);
    }
    
    // Detect checkerboard corners in frame
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
            // Refine corner locations to sub-pixel accuracy
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
            
            // Store current frame for potential saving
            current_frame = frame.clone();
            
            // Print debug info
            std::cout << "Found " << current_corners.size() << " corners" << std::endl;
            if (!current_corners.empty()) {
                std::cout << "First corner at: (" 
                          << current_corners[0].x << ", " 
                          << current_corners[0].y << ")" << std::endl;
            }
        }
        
        return corners_found;
    }
    
    // Draw detected corners on frame
    void drawCorners(cv::Mat& frame) {
        if (corners_found && !current_corners.empty()) {
            cv::drawChessboardCorners(frame, boardSize, current_corners, corners_found);
        }
    }
    
    // Generate 3D world points for the checkerboard
    std::vector<cv::Vec3f> generate3DPoints() {
        std::vector<cv::Vec3f> points;
        
        // Create 3D points in world coordinates
        // Z=0 (flat plane), origin at top-left corner
        for (int i = 0; i < boardSize.height; i++) {
            for (int j = 0; j < boardSize.width; j++) {
                points.push_back(cv::Vec3f(
                    j * squareSize,      // X
                    -i * squareSize,     // Y (negative for top-down)
                    0.0f                 // Z
                ));
            }
        }
        
        return points;
    }
    
    // Save current detection for calibration
    bool saveCalibrationImage() {
        if (!corners_found || current_corners.empty()) {
            std::cout << "No valid corners to save!" << std::endl;
            return false;
        }
        
        // Add corners to list
        corner_list.push_back(current_corners);
        
        // Add corresponding 3D points
        point_list.push_back(generate3DPoints());
        
        // Save the image
        calibration_images.push_back(current_frame.clone());
        
        std::cout << "Saved calibration image #" << corner_list.size() << std::endl;
        
        return true;
    }
    
    // Perform camera calibration
    bool calibrateCamera(cv::Size image_size) {
        if (corner_list.size() < 5) {
            std::cout << "Need at least 5 calibration images. Currently have: " 
                      << corner_list.size() << std::endl;
            return false;
        }
        
        std::vector<cv::Mat> rvecs, tvecs;
        
        double rms_error = cv::calibrateCamera(
            point_list,
            corner_list,
            image_size,
            camera_matrix,
            dist_coeffs,
            rvecs,
            tvecs,
            cv::CALIB_FIX_ASPECT_RATIO
        );
        
        std::cout << "\n=== Calibration Complete ===" << std::endl;
        std::cout << "RMS reprojection error: " << rms_error << std::endl;
        std::cout << "Camera Matrix:\n" << camera_matrix << std::endl;
        std::cout << "Distortion Coefficients:\n" << dist_coeffs << std::endl;
        
        is_calibrated = true;
        return true;
    }
    
    // Save calibration data
    void saveCalibration(const std::string& filename) {
        cv::FileStorage fs(filename, cv::FileStorage::WRITE);
        fs << "camera_matrix" << camera_matrix;
        fs << "distortion_coefficients" << dist_coeffs;
        fs << "image_width" << current_frame.cols;
        fs << "image_height" << current_frame.rows;
        fs.release();
        std::cout << "Calibration saved to " << filename << std::endl;
    }
    
    // Load calibration data
    bool loadCalibration(const std::string& filename) {
        cv::FileStorage fs(filename, cv::FileStorage::READ);
        if (!fs.isOpened()) {
            return false;
        }
        fs["camera_matrix"] >> camera_matrix;
        fs["distortion_coefficients"] >> dist_coeffs;
        fs.release();
        is_calibrated = true;
        std::cout << "Calibration loaded from " << filename << std::endl;
        return true;
    }
    
    // Save calibration image with corners
    void saveImageWithCorners(int index, const std::string& filename) {
        if (index < 0 || index >= calibration_images.size()) {
            return;
        }
        cv::Mat img = calibration_images[index].clone();
        cv::drawChessboardCorners(img, boardSize, corner_list[index], true);
        cv::imwrite(filename, img);
        std::cout << "Saved calibration image to " << filename << std::endl;
    }
    
    int getCalibrationCount() const { return corner_list.size(); }
    bool isCalibrated() const { return is_calibrated; }
    cv::Mat getCameraMatrix() const { return camera_matrix; }
    cv::Mat getDistCoeffs() const { return dist_coeffs; }
};

int main(int argc, char** argv) {
    // Initialize video capture
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open camera" << std::endl;
        return -1;
    }
    
    // Create calibration system (9x6 checkerboard)
    CalibrationSystem calib(9, 6, 1.0f);
    
    // Try to load existing calibration
    calib.loadCalibration("calibration.xml");
    
    cv::Mat frame;
    std::cout << "\n=== Camera Calibration System ===" << std::endl;
    std::cout << "Controls:" << std::endl;
    std::cout << "  's' - Save current frame for calibration" << std::endl;
    std::cout << "  'c' - Calibrate camera using saved images" << std::endl;
    std::cout << "  'w' - Write calibration to file" << std::endl;
    std::cout << "  'q' - Quit" << std::endl;
    std::cout << "================================\n" << std::endl;
    
    while (true) {
        cap >> frame;
        if (frame.empty()) {
            break;
        }
        
        // Detect corners
        calib.detectCorners(frame);
        
        // Draw corners on frame
        calib.drawCorners(frame);
        
        // Display calibration count
        std::string status = "Calibration images: " + 
                            std::to_string(calib.getCalibrationCount());
        cv::putText(frame, status, cv::Point(10, 30), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        
        if (calib.isCalibrated()) {
            cv::putText(frame, "CALIBRATED", cv::Point(10, 60),
                       cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        }
        
        cv::imshow("Camera Calibration", frame);
        
        char key = cv::waitKey(100);
        if (key == 'q' || key == 27) { // q or ESC
            break;
        } else if (key == 's') {
            if (calib.saveCalibrationImage()) {
                // Save the image with corners for report
                int idx = calib.getCalibrationCount() - 1;
                std::string filename = "start_calib_img_" + std::to_string(idx) + ".png";
                calib.saveImageWithCorners(idx, filename);
            }
        } else if (key == 'c') {
            calib.calibrateCamera(frame.size());
        } else if (key == 'w') {
            calib.saveCalibration("calibration.xml");
        }
    }
    
    cap.release();
    cv::destroyAllWindows();
    
    return 0;
}