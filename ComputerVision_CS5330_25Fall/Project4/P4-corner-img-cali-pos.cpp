#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <cmath>

class PoseEstimation {
private:
    cv::Size boardSize;
    float squareSize;
    
    cv::Mat camera_matrix;
    cv::Mat dist_coeffs;
    
    cv::Mat rvec;  // Current rotation vector
    cv::Mat tvec;  // Current translation vector
    
    std::vector<cv::Point2f> current_corners;
    bool pose_valid;
    
public:
    PoseEstimation(int cols = 9, int rows = 6, float sq_size = 1.0f)
        : boardSize(cols, rows), squareSize(sq_size), pose_valid(false) {
        camera_matrix = cv::Mat::eye(3, 3, CV_64FC1);
        dist_coeffs = cv::Mat::zeros(5, 1, CV_64FC1);
    }
    
    bool loadCalibration(const std::string& filename) {
        cv::FileStorage fs(filename, cv::FileStorage::READ);
        if (!fs.isOpened()) {
            std::cerr << "[ERROR] Cannot open calibration file: " << filename << std::endl;
            return false;
        }
        
        fs["camera_matrix"] >> camera_matrix;
        fs["distortion_coefficients"] >> dist_coeffs;
        
        std::string date;
        double rms_error;
        fs["calibration_date"] >> date;
        fs["rms_error"] >> rms_error;
        
        fs.release();
        
        std::cout << "\n[LOADED] Calibration Parameters" << std::endl;
        std::cout << "Date: " << date;
        std::cout << "RMS Error: " << rms_error << " pixels" << std::endl;
        std::cout << "\nCamera Matrix:" << std::endl;
        std::cout << camera_matrix << std::endl;
        std::cout << "\nDistortion Coefficients:" << std::endl;
        std::cout << dist_coeffs.t() << std::endl << std::endl;
        
        return true;
    }
    
    std::vector<cv::Vec3f> generate3DPoints() {
        std::vector<cv::Vec3f> points;
        
        for (int i = 0; i < boardSize.height; i++) {
            for (int j = 0; j < boardSize.width; j++) {
                points.push_back(cv::Vec3f(
                    j * squareSize,
                    -i * squareSize,
                    0.0f
                ));
            }
        }
        
        return points;
    }
    
    bool detectAndEstimatePose(cv::Mat& frame) {
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        
        current_corners.clear();
        bool found = cv::findChessboardCorners(
            gray,
            boardSize,
            current_corners,
            cv::CALIB_CB_ADAPTIVE_THRESH |
            cv::CALIB_CB_NORMALIZE_IMAGE |
            cv::CALIB_CB_FAST_CHECK
        );
        
        if (!found) {
            pose_valid = false;
            return false;
        }
        
        // Refine corners
        cv::cornerSubPix(
            gray,
            current_corners,
            cv::Size(11, 11),
            cv::Size(-1, -1),
            cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.1)
        );
        
        // Generate 3D points
        std::vector<cv::Vec3f> object_points = generate3DPoints();
        
        // Solve PnP to get pose
        pose_valid = cv::solvePnP(
            object_points,
            current_corners,
            camera_matrix,
            dist_coeffs,
            rvec,
            tvec,
            false,
            cv::SOLVEPNP_ITERATIVE
        );
        
        return pose_valid;
    }
    
    void printPoseInformation() {
        if (!pose_valid) return;
        
        // Convert rotation vector to rotation matrix
        cv::Mat rotation_matrix;
        cv::Rodrigues(rvec, rotation_matrix);
        
        // Extract Euler angles (in degrees)
        double sy = std::sqrt(rotation_matrix.at<double>(0,0) * rotation_matrix.at<double>(0,0) +
                             rotation_matrix.at<double>(1,0) * rotation_matrix.at<double>(1,0));
        
        bool singular = sy < 1e-6;
        
        double x_angle, y_angle, z_angle;
        if (!singular) {
            x_angle = std::atan2(rotation_matrix.at<double>(2,1), rotation_matrix.at<double>(2,2));
            y_angle = std::atan2(-rotation_matrix.at<double>(2,0), sy);
            z_angle = std::atan2(rotation_matrix.at<double>(1,0), rotation_matrix.at<double>(0,0));
        } else {
            x_angle = std::atan2(-rotation_matrix.at<double>(1,2), rotation_matrix.at<double>(1,1));
            y_angle = std::atan2(-rotation_matrix.at<double>(2,0), sy);
            z_angle = 0;
        }
        
        // Convert to degrees
        x_angle = x_angle * 180.0 / CV_PI;
        y_angle = y_angle * 180.0 / CV_PI;
        z_angle = z_angle * 180.0 / CV_PI;
        
        // Get translation (camera position relative to target)
        double tx = tvec.at<double>(0);
        double ty = tvec.at<double>(1);
        double tz = tvec.at<double>(2);
        
        // Distance from camera to target
        double distance = std::sqrt(tx*tx + ty*ty + tz*tz);
        
        // Print pose information
        std::cout << "\r" << std::string(100, ' ') << "\r";  // Clear line
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Rotation (deg) | X: " << std::setw(7) << x_angle
                  << " Y: " << std::setw(7) << y_angle
                  << " Z: " << std::setw(7) << z_angle
                  << " || Translation | X: " << std::setw(6) << tx
                  << " Y: " << std::setw(6) << ty
                  << " Z: " << std::setw(6) << tz
                  << " | Dist: " << std::setw(6) << distance
                  << std::flush;
    }
    
    void drawPose(cv::Mat& frame) {
        if (!pose_valid) return;
        
        // Draw checkerboard corners
        cv::drawChessboardCorners(frame, boardSize, current_corners, true);
        
        // Draw coordinate axes
        std::vector<cv::Point3f> axis_points;
        float axis_length = 3.0f * squareSize;
        
        axis_points.push_back(cv::Point3f(0, 0, 0));           // Origin
        axis_points.push_back(cv::Point3f(axis_length, 0, 0)); // X axis
        axis_points.push_back(cv::Point3f(0, -axis_length, 0));// Y axis
        axis_points.push_back(cv::Point3f(0, 0, -axis_length));// Z axis
        
        std::vector<cv::Point2f> image_points;
        cv::projectPoints(axis_points, rvec, tvec, camera_matrix, dist_coeffs, image_points);
        
        // Draw axes
        cv::line(frame, image_points[0], image_points[1], cv::Scalar(0, 0, 255), 3); // X - Red
        cv::line(frame, image_points[0], image_points[2], cv::Scalar(0, 255, 0), 3); // Y - Green
        cv::line(frame, image_points[0], image_points[3], cv::Scalar(255, 0, 0), 3); // Z - Blue
        
        // Add labels
        cv::putText(frame, "X", image_points[1], cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
        cv::putText(frame, "Y", image_points[2], cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
        cv::putText(frame, "Z", image_points[3], cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 0, 0), 2);
        
        // Draw origin circle
        cv::circle(frame, image_points[0], 5, cv::Scalar(255, 255, 0), -1);
    }
    
    void drawPoseInfo(cv::Mat& frame) {
        if (!pose_valid) {
            cv::putText(frame, "No target detected", cv::Point(10, 30),
                       cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
            return;
        }
        
        // Convert rotation to Euler angles
        cv::Mat rotation_matrix;
        cv::Rodrigues(rvec, rotation_matrix);
        
        double sy = std::sqrt(rotation_matrix.at<double>(0,0) * rotation_matrix.at<double>(0,0) +
                             rotation_matrix.at<double>(1,0) * rotation_matrix.at<double>(1,0));
        
        double x_angle = std::atan2(rotation_matrix.at<double>(2,1), rotation_matrix.at<double>(2,2)) * 180.0 / CV_PI;
        double y_angle = std::atan2(-rotation_matrix.at<double>(2,0), sy) * 180.0 / CV_PI;
        double z_angle = std::atan2(rotation_matrix.at<double>(1,0), rotation_matrix.at<double>(0,0)) * 180.0 / CV_PI;
        
        double tx = tvec.at<double>(0);
        double ty = tvec.at<double>(1);
        double tz = tvec.at<double>(2);
        double distance = std::sqrt(tx*tx + ty*ty + tz*tz);
        
        // Draw info panel
        int y = 30;
        int spacing = 25;
        cv::Scalar color(0, 255, 0);
        
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(1);
        
        oss.str(""); oss << "Rotation X: " << x_angle << " deg";
        cv::putText(frame, oss.str(), cv::Point(10, y), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
        y += spacing;
        
        oss.str(""); oss << "Rotation Y: " << y_angle << " deg";
        cv::putText(frame, oss.str(), cv::Point(10, y), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
        y += spacing;
        
        oss.str(""); oss << "Rotation Z: " << z_angle << " deg";
        cv::putText(frame, oss.str(), cv::Point(10, y), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
        y += spacing;
        
        y += 10;
        oss.str(""); oss << "Translation X: " << tx;
        cv::putText(frame, oss.str(), cv::Point(10, y), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
        y += spacing;
        
        oss.str(""); oss << "Translation Y: " << ty;
        cv::putText(frame, oss.str(), cv::Point(10, y), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
        y += spacing;
        
        oss.str(""); oss << "Translation Z: " << tz;
        cv::putText(frame, oss.str(), cv::Point(10, y), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
        y += spacing;
        
        y += 10;
        oss.str(""); oss << "Distance: " << distance;
        cv::putText(frame, oss.str(), cv::Point(10, y), cv::FONT_HERSHEY_SIMPLEX, 0.6, 
                   cv::Scalar(0, 255, 255), 2);
    }
    
    bool isPoseValid() const { return pose_valid; }
    cv::Mat getRotationVector() const { return rvec; }
    cv::Mat getTranslationVector() const { return tvec; }
};

int main(int argc, char** argv) {
    std::string calib_file = "calibration.xml";
    
    if (argc > 1) {
        calib_file = argv[1];
    }
    
    PoseEstimation pose(9, 6, 1.0f);
    
    if (!pose.loadCalibration(calib_file)) {
        std::cerr << "[ERROR] Failed to load calibration. Please run calibration first." << std::endl;
        return -1;
    }
    
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "[ERROR] Cannot open camera" << std::endl;
        return -1;
    }
    
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "CAMERA POSE ESTIMATION" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    std::cout << "Position the checkerboard in front of the camera" << std::endl;
    std::cout << "Move the camera/target and observe the pose changes" << std::endl;
    std::cout << "\nCoordinate System:" << std::endl;
    std::cout << "  X-axis (RED): Points right along the checkerboard" << std::endl;
    std::cout << "  Y-axis (GREEN): Points down along the checkerboard" << std::endl;
    std::cout << "  Z-axis (BLUE): Points out from the checkerboard" << std::endl;
    std::cout << "\nTranslation values show camera position relative to target" << std::endl;
    std::cout << "Press 'q' to quit" << std::endl;
    std::cout << std::string(70, '=') << std::endl << std::endl;
    
    cv::Mat frame;
    
    while (true) {
        cap >> frame;
        if (frame.empty()) break;
        
        // Detect target and estimate pose
        pose.detectAndEstimatePose(frame);
        
        // Draw visualization
        pose.drawPose(frame);
        pose.drawPoseInfo(frame);
        
        // Print pose to console
        pose.printPoseInformation();
        
        cv::imshow("Pose Estimation", frame);
        
        char key = cv::waitKey(500);
        if (key == 'q' || key == 27) break;
    }
    
    std::cout << std::endl;  // New line after pose output
    
    cap.release();
    cv::destroyAllWindows();
    
    return 0;
}