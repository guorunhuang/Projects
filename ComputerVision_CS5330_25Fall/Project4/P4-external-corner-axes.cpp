#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <vector>

class CornerProjector {
private:
    cv::Mat camera_matrix;
    cv::Mat dist_coeffs;
    cv::Size board_size;
    float square_size;
    
    std::vector<cv::Point3f> object_points;
    std::vector<cv::Point3f> outer_corners_3d;
    std::vector<cv::Point3f> axis_points_3d;
    
public:
    CornerProjector(int cols = 9, int rows = 6, float sq_size = 1.0f)
        : board_size(cols, rows), square_size(sq_size) {
        
        // Generate internal corner points (used for pose estimation)
        for (int i = 0; i < board_size.height; i++) {
            for (int j = 0; j < board_size.width; j++) {
                object_points.push_back(cv::Point3f(
                    j * square_size,
                    -i * square_size,
                    0.0f
                ));
            }
        }
        
        // Define outer corners (4 corners of the entire checkerboard)
        // These are OUTSIDE the internal corners
        float width = board_size.width * square_size;
        float height = board_size.height * square_size;
        
        outer_corners_3d = {
            {-square_size, square_size, 0.0f},              // Top-left (outside)
            {width, square_size, 0.0f},                     // Top-right (outside)
            {width, -height, 0.0f},                         // Bottom-right (outside)
            {-square_size, -height, 0.0f}                   // Bottom-left (outside)
        };
        
        // Define 3D axes
        float axis_length = 3.0f * square_size;
        axis_points_3d = {
            {0, 0, 0},                      // Origin
            {axis_length, 0, 0},            // X-axis (Red)
            {0, -axis_length, 0},           // Y-axis (Green)
            {0, 0, -axis_length}            // Z-axis (Blue)
        };
        
        std::cout << "[INIT] Corner projector initialized" << std::endl;
        std::cout << "       Board size: " << board_size.width << "x" 
                  << board_size.height << std::endl;
        std::cout << "       Outer corners defined" << std::endl;
        std::cout << "       3D axes defined" << std::endl;
    }
    
    bool loadCalibration(const std::string& filename) {
        cv::FileStorage fs(filename, cv::FileStorage::READ);
        if (!fs.isOpened()) {
            std::cerr << "[ERROR] Cannot open calibration file" << std::endl;
            return false;
        }
        
        fs["camera_matrix"] >> camera_matrix;
        fs["distortion_coefficients"] >> dist_coeffs;
        fs.release();
        
        std::cout << "[LOADED] Calibration parameters" << std::endl;
        return true;
    }
    
    bool detectAndEstimatePose(cv::Mat& frame, cv::Mat& rvec, cv::Mat& tvec) {
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        
        std::vector<cv::Point2f> corners;
        bool found = cv::findChessboardCorners(
            gray, board_size, corners,
            cv::CALIB_CB_ADAPTIVE_THRESH | 
            cv::CALIB_CB_NORMALIZE_IMAGE |
            cv::CALIB_CB_FAST_CHECK
        );
        
        if (!found) return false;
        
        // Refine corners
        cv::cornerSubPix(
            gray, corners,
            cv::Size(11, 11),
            cv::Size(-1, -1),
            cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.1)
        );
        
        // Estimate pose
        bool success = cv::solvePnP(
            object_points, corners,
            camera_matrix, dist_coeffs,
            rvec, tvec
        );
        
        return success;
    }
    
    void drawOuterCorners(cv::Mat& frame, const cv::Mat& rvec, const cv::Mat& tvec) {
        // Project outer corners to image
        std::vector<cv::Point2f> outer_corners_2d;
        cv::projectPoints(
            outer_corners_3d,
            rvec, tvec,
            camera_matrix, dist_coeffs,
            outer_corners_2d
        );
        
        // Draw the outer corners as circles
        cv::Scalar colors[] = {
            cv::Scalar(255, 0, 0),    // Top-left: Blue
            cv::Scalar(0, 255, 0),    // Top-right: Green
            cv::Scalar(0, 0, 255),    // Bottom-right: Red
            cv::Scalar(255, 255, 0)   // Bottom-left: Cyan
        };
        
        std::string labels[] = {"TL", "TR", "BR", "BL"};
        
        for (size_t i = 0; i < outer_corners_2d.size(); i++) {
            // Draw circle
            cv::circle(frame, outer_corners_2d[i], 10, colors[i], -1);
            cv::circle(frame, outer_corners_2d[i], 12, cv::Scalar(255, 255, 255), 2);
            
            // Draw label
            cv::putText(frame, labels[i],
                       outer_corners_2d[i] + cv::Point2f(15, 5),
                       cv::FONT_HERSHEY_SIMPLEX, 0.6, colors[i], 2);
        }
        
        // Draw lines connecting outer corners
        for (size_t i = 0; i < outer_corners_2d.size(); i++) {
            cv::line(frame, outer_corners_2d[i], 
                    outer_corners_2d[(i+1) % outer_corners_2d.size()],
                    cv::Scalar(255, 255, 255), 2);
        }
    }
    
    void draw3DAxes(cv::Mat& frame, const cv::Mat& rvec, const cv::Mat& tvec) {
        // Project axes to image
        std::vector<cv::Point2f> axis_points_2d;
        cv::projectPoints(
            axis_points_3d,
            rvec, tvec,
            camera_matrix, dist_coeffs,
            axis_points_2d
        );
        
        // Draw axes with different colors and thickness
        // X-axis (Red)
        cv::arrowedLine(frame, axis_points_2d[0], axis_points_2d[1],
                       cv::Scalar(0, 0, 255), 4, cv::LINE_AA, 0, 0.15);
        cv::putText(frame, "X", axis_points_2d[1] + cv::Point2f(10, 10),
                   cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
        
        // Y-axis (Green)
        cv::arrowedLine(frame, axis_points_2d[0], axis_points_2d[2],
                       cv::Scalar(0, 255, 0), 4, cv::LINE_AA, 0, 0.15);
        cv::putText(frame, "Y", axis_points_2d[2] + cv::Point2f(10, 10),
                   cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
        
        // Z-axis (Blue)
        cv::arrowedLine(frame, axis_points_2d[0], axis_points_2d[3],
                       cv::Scalar(255, 0, 0), 4, cv::LINE_AA, 0, 0.15);
        cv::putText(frame, "Z", axis_points_2d[3] + cv::Point2f(10, 10),
                   cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 0, 0), 2);
        
        // Draw origin point
        cv::circle(frame, axis_points_2d[0], 8, cv::Scalar(255, 255, 0), -1);
        cv::circle(frame, axis_points_2d[0], 10, cv::Scalar(255, 255, 255), 2);
        cv::putText(frame, "Origin (0,0,0)", axis_points_2d[0] + cv::Point2f(15, -15),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }
    
    void drawAll(cv::Mat& frame, const cv::Mat& rvec, const cv::Mat& tvec,
                 bool show_corners = true, bool show_axes = true) {
        if (show_corners) {
            drawOuterCorners(frame, rvec, tvec);
        }
        if (show_axes) {
            draw3DAxes(frame, rvec, tvec);
        }
    }
    
    void printPoseInfo(const cv::Mat& rvec, const cv::Mat& tvec) {
        // Convert rotation vector to rotation matrix
        cv::Mat R;
        cv::Rodrigues(rvec, R);
        
        // Extract Euler angles
        double sy = std::sqrt(R.at<double>(0,0) * R.at<double>(0,0) +
                             R.at<double>(1,0) * R.at<double>(1,0));
        
        double x_angle = std::atan2(R.at<double>(2,1), R.at<double>(2,2)) * 180.0 / CV_PI;
        double y_angle = std::atan2(-R.at<double>(2,0), sy) * 180.0 / CV_PI;
        double z_angle = std::atan2(R.at<double>(1,0), R.at<double>(0,0)) * 180.0 / CV_PI;
        
        double tx = tvec.at<double>(0);
        double ty = tvec.at<double>(1);
        double tz = tvec.at<double>(2);
        
        std::cout << "\rRotation(deg): X=" << std::fixed << std::setprecision(1)
                  << x_angle << " Y=" << y_angle << " Z=" << z_angle
                  << " | Translation: X=" << tx << " Y=" << ty << " Z=" << tz
                  << "   " << std::flush;
    }
};

int main(int argc, char** argv) {
    std::string calib_file = "calibration.xml";
    if (argc > 1) {
        calib_file = argv[1];
    }
    
    CornerProjector projector(9, 6, 1.0f);
    
    if (!projector.loadCalibration(calib_file)) {
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
    std::cout << "PROJECT OUTSIDE CORNERS AND 3D AXES" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    std::cout << "\nVisualization:" << std::endl;
    std::cout << "  - OUTER CORNERS: 4 colored circles at board boundaries" << std::endl;
    std::cout << "    * Top-Left (TL): Blue" << std::endl;
    std::cout << "    * Top-Right (TR): Green" << std::endl;
    std::cout << "    * Bottom-Right (BR): Red" << std::endl;
    std::cout << "    * Bottom-Left (BL): Cyan" << std::endl;
    std::cout << "\n  - 3D AXES at origin (0,0,0):" << std::endl;
    std::cout << "    * X-axis: Red (points right)" << std::endl;
    std::cout << "    * Y-axis: Green (points down)" << std::endl;
    std::cout << "    * Z-axis: Blue (points toward camera)" << std::endl;
    std::cout << "\nControls:" << std::endl;
    std::cout << "  '1' - Show corners only" << std::endl;
    std::cout << "  '2' - Show axes only" << std::endl;
    std::cout << "  '3' - Show both (default)" << std::endl;
    std::cout << "  's' - Save screenshot" << std::endl;
    std::cout << "  'q' - Quit" << std::endl;
    std::cout << std::string(70, '=') << std::endl << std::endl;
    
    cv::Mat frame, rvec, tvec;
    bool show_corners = true;
    bool show_axes = true;
    int screenshot_count = 0;
    
    while (true) {
        cap >> frame;
        if (frame.empty()) break;
        
        // Detect target and estimate pose
        if (projector.detectAndEstimatePose(frame, rvec, tvec)) {
            // Draw projections
            projector.drawAll(frame, rvec, tvec, show_corners, show_axes);
            
            // Print pose information
            projector.printPoseInfo(rvec, tvec);
            
            // Display status
            std::string status = "Target detected";
            cv::putText(frame, status, cv::Point(10, 30),
                       cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        } else {
            cv::putText(frame, "No target detected", cv::Point(10, 30),
                       cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
        }
        
        // Display mode
        std::string mode;
        if (show_corners && show_axes) mode = "Mode: Corners + Axes";
        else if (show_corners) mode = "Mode: Corners only";
        else if (show_axes) mode = "Mode: Axes only";
        
        cv::putText(frame, mode, cv::Point(10, 60),
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
        
        cv::imshow("Project Corners and Axes", frame);
        
        char key = cv::waitKey(500);
        if (key == 'q' || key == 27) {
            break;
        } else if (key == '1') {
            show_corners = true;
            show_axes = false;
            std::cout << "\n[MODE] Showing corners only" << std::endl;
        } else if (key == '2') {
            show_corners = false;
            show_axes = true;
            std::cout << "\n[MODE] Showing axes only" << std::endl;
        } else if (key == '3') {
            show_corners = true;
            show_axes = true;
            std::cout << "\n[MODE] Showing both" << std::endl;
        } else if (key == 's') {
            std::string filename = "projection_screenshot_" + 
                                  std::to_string(screenshot_count++) + ".png";
            cv::imwrite(filename, frame);
            std::cout << "\n[SAVED] Screenshot: " << filename << std::endl;
        }
    }
    
    std::cout << std::endl;
    cap.release();
    cv::destroyAllWindows();
    
    return 0;
}