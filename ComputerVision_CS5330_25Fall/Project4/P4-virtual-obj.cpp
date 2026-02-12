#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <vector>
#include <cmath>

class VirtualObjectAR {
private:
    cv::Mat camera_matrix;
    cv::Mat dist_coeffs;
    cv::Size board_size;
    float square_size;
    std::vector<cv::Point3f> object_points;
    float animation_time;
    
public:
    VirtualObjectAR(int cols = 9, int rows = 6, float sq_size = 1.0f)
        : board_size(cols, rows), square_size(sq_size), animation_time(0.0f) {
        
        // Generate object points for pose estimation
        for (int i = 0; i < board_size.height; i++) {
            for (int j = 0; j < board_size.width; j++) {
                object_points.push_back(cv::Point3f(
                    j * square_size, -i * square_size, 0.0f));
            }
        }
    }
    
    bool loadCalibration(const std::string& filename) {
        cv::FileStorage fs(filename, cv::FileStorage::READ);
        if (!fs.isOpened()) return false;
        
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
        bool found = cv::findChessboardCorners(gray, board_size, corners,
            cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK);
        
        if (!found) return false;
        
        cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1),
            cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.1));
        
        return cv::solvePnP(object_points, corners, camera_matrix, dist_coeffs, rvec, tvec);
    }
    
    // Helper function to project and draw a line
    void drawLine3D(cv::Mat& frame, const cv::Point3f& p1, const cv::Point3f& p2,
                    const cv::Mat& rvec, const cv::Mat& tvec,
                    const cv::Scalar& color, int thickness = 2) {
        std::vector<cv::Point3f> points_3d = {p1, p2};
        std::vector<cv::Point2f> points_2d;
        
        cv::projectPoints(points_3d, rvec, tvec, camera_matrix, dist_coeffs, points_2d);
        cv::line(frame, points_2d[0], points_2d[1], color, thickness, cv::LINE_AA);
    }
    
    // 1. Floating House - Asymmetric object for debugging
    void drawFloatingHouse(cv::Mat& frame, const cv::Mat& rvec, const cv::Mat& tvec) {
        float cx = 4.5f;  // Center X
        float cy = -3.0f; // Center Y
        float base_z = -2.0f;
        float size = 2.0f;
        float height = 3.0f;
        float roof_height = 1.5f;
        
        // Base points (floor level)
        cv::Point3f bl(cx - size/2, cy - size/2, base_z);          // Back-left
        cv::Point3f br(cx + size/2, cy - size/2, base_z);          // Back-right
        cv::Point3f fl(cx - size/2, cy + size/2, base_z);          // Front-left
        cv::Point3f fr(cx + size/2, cy + size/2, base_z);          // Front-right
        
        // Top points (wall level)
        cv::Point3f tbl(cx - size/2, cy - size/2, base_z - height);
        cv::Point3f tbr(cx + size/2, cy - size/2, base_z - height);
        cv::Point3f tfl(cx - size/2, cy + size/2, base_z - height);
        cv::Point3f tfr(cx + size/2, cy + size/2, base_z - height);
        
        // Roof peak (asymmetric - not centered)
        cv::Point3f peak(cx + size/4, cy, base_z - height - roof_height);
        
        cv::Scalar wall_color(150, 100, 50);    // Brown walls
        cv::Scalar roof_color(180, 50, 50);     // Red roof
        cv::Scalar edge_color(255, 255, 255);   // White edges
        
        // Draw base (floor)
        drawLine3D(frame, bl, br, rvec, tvec, edge_color, 2);
        drawLine3D(frame, br, fr, rvec, tvec, edge_color, 2);
        drawLine3D(frame, fr, fl, rvec, tvec, edge_color, 2);
        drawLine3D(frame, fl, bl, rvec, tvec, edge_color, 2);
        
        // Draw walls
        drawLine3D(frame, bl, tbl, rvec, tvec, wall_color, 3);
        drawLine3D(frame, br, tbr, rvec, tvec, wall_color, 3);
        drawLine3D(frame, fl, tfl, rvec, tvec, wall_color, 3);
        drawLine3D(frame, fr, tfr, rvec, tvec, wall_color, 3);
        
        // Draw top of walls
        drawLine3D(frame, tbl, tbr, rvec, tvec, wall_color, 2);
        drawLine3D(frame, tbr, tfr, rvec, tvec, wall_color, 2);
        drawLine3D(frame, tfr, tfl, rvec, tvec, wall_color, 2);
        drawLine3D(frame, tfl, tbl, rvec, tvec, wall_color, 2);
        
        // Draw roof (asymmetric)
        drawLine3D(frame, tbl, peak, rvec, tvec, roof_color, 3);
        drawLine3D(frame, tbr, peak, rvec, tvec, roof_color, 3);
        drawLine3D(frame, tfl, peak, rvec, tvec, roof_color, 3);
        drawLine3D(frame, tfr, peak, rvec, tvec, roof_color, 3);
        
        // Draw door (front-left)
        float door_width = 0.6f;
        float door_height = 1.2f;
        cv::Point3f door_bl(cx - size/4, cy + size/2, base_z);
        cv::Point3f door_br(cx - size/4 + door_width, cy + size/2, base_z);
        cv::Point3f door_tl(cx - size/4, cy + size/2, base_z - door_height);
        cv::Point3f door_tr(cx - size/4 + door_width, cy + size/2, base_z - door_height);
        
        drawLine3D(frame, door_bl, door_br, rvec, tvec, cv::Scalar(100, 50, 0), 2);
        drawLine3D(frame, door_br, door_tr, rvec, tvec, cv::Scalar(100, 50, 0), 2);
        drawLine3D(frame, door_tr, door_tl, rvec, tvec, cv::Scalar(100, 50, 0), 2);
        drawLine3D(frame, door_tl, door_bl, rvec, tvec, cv::Scalar(100, 50, 0), 2);
        
        // Draw window (front-right)
        float win_size = 0.5f;
        cv::Point3f win_bl(cx + size/4 - win_size/2, cy + size/2, base_z - height/2 - win_size/2);
        cv::Point3f win_br(cx + size/4 + win_size/2, cy + size/2, base_z - height/2 - win_size/2);
        cv::Point3f win_tl(cx + size/4 - win_size/2, cy + size/2, base_z - height/2 + win_size/2);
        cv::Point3f win_tr(cx + size/4 + win_size/2, cy + size/2, base_z - height/2 + win_size/2);
        
        drawLine3D(frame, win_bl, win_br, rvec, tvec, cv::Scalar(200, 200, 255), 2);
        drawLine3D(frame, win_br, win_tr, rvec, tvec, cv::Scalar(200, 200, 255), 2);
        drawLine3D(frame, win_tr, win_tl, rvec, tvec, cv::Scalar(200, 200, 255), 2);
        drawLine3D(frame, win_tl, win_bl, rvec, tvec, cv::Scalar(200, 200, 255), 2);
        
        // Cross in window
        drawLine3D(frame, win_bl, win_tr, rvec, tvec, cv::Scalar(200, 200, 255), 1);
        drawLine3D(frame, win_br, win_tl, rvec, tvec, cv::Scalar(200, 200, 255), 1);
    }
    
    // 2. Rotating Windmill
    void drawWindmill(cv::Mat& frame, const cv::Mat& rvec, const cv::Mat& tvec) {
        float cx = 4.5f;
        float cy = -3.0f;
        float base_height = 4.0f;
        float blade_length = 1.5f;
        
        // Tower (tapered)
        cv::Point3f base1(cx - 0.3f, cy - 0.3f, 0.0f);
        cv::Point3f base2(cx + 0.3f, cy - 0.3f, 0.0f);
        cv::Point3f base3(cx + 0.3f, cy + 0.3f, 0.0f);
        cv::Point3f base4(cx - 0.3f, cy + 0.3f, 0.0f);
        
        cv::Point3f top1(cx - 0.15f, cy - 0.15f, -base_height);
        cv::Point3f top2(cx + 0.15f, cy - 0.15f, -base_height);
        cv::Point3f top3(cx + 0.15f, cy + 0.15f, -base_height);
        cv::Point3f top4(cx - 0.15f, cy + 0.15f, -base_height);
        
        cv::Scalar tower_color(100, 100, 150);
        
        // Draw tower edges
        drawLine3D(frame, base1, top1, rvec, tvec, tower_color, 3);
        drawLine3D(frame, base2, top2, rvec, tvec, tower_color, 3);
        drawLine3D(frame, base3, top3, rvec, tvec, tower_color, 3);
        drawLine3D(frame, base4, top4, rvec, tvec, tower_color, 3);
        
        // Draw base and top
        drawLine3D(frame, base1, base2, rvec, tvec, tower_color, 2);
        drawLine3D(frame, base2, base3, rvec, tvec, tower_color, 2);
        drawLine3D(frame, base3, base4, rvec, tvec, tower_color, 2);
        drawLine3D(frame, base4, base1, rvec, tvec, tower_color, 2);
        
        drawLine3D(frame, top1, top2, rvec, tvec, tower_color, 2);
        drawLine3D(frame, top2, top3, rvec, tvec, tower_color, 2);
        drawLine3D(frame, top3, top4, rvec, tvec, tower_color, 2);
        drawLine3D(frame, top4, top1, rvec, tvec, tower_color, 2);
        
        // Rotating blades
        cv::Point3f center(cx, cy, -base_height - 0.3f);
        float angle = animation_time * 2.0f;
        
        cv::Scalar blade_color(200, 200, 200);
        
        for (int i = 0; i < 4; i++) {
            float blade_angle = angle + i * CV_PI / 2.0f;
            
            // Blade end point
            float blade_x = cx + blade_length * std::cos(blade_angle);
            float blade_y = cy + blade_length * std::sin(blade_angle);
            cv::Point3f blade_end(blade_x, blade_y, center.z);
            
            // Draw blade
            drawLine3D(frame, center, blade_end, rvec, tvec, blade_color, 3);
            
            // Draw blade triangle (windmill sail)
            float perp_angle = blade_angle + CV_PI / 2.0f;
            float sail_width = 0.3f;
            cv::Point3f sail1(
                blade_x + sail_width * std::cos(perp_angle),
                blade_y + sail_width * std::sin(perp_angle),
                center.z
            );
            cv::Point3f sail2(
                blade_x - sail_width * std::cos(perp_angle),
                blade_y - sail_width * std::sin(perp_angle),
                center.z
            );
            
            drawLine3D(frame, blade_end, sail1, rvec, tvec, cv::Scalar(255, 200, 150), 2);
            drawLine3D(frame, blade_end, sail2, rvec, tvec, cv::Scalar(255, 200, 150), 2);
            drawLine3D(frame, sail1, center, rvec, tvec, cv::Scalar(255, 200, 150), 1);
            drawLine3D(frame, sail2, center, rvec, tvec, cv::Scalar(255, 200, 150), 1);
        }
        
        animation_time += 0.05f;
    }
    
    // 3. 3D Tree
    void drawTree(cv::Mat& frame, const cv::Mat& rvec, const cv::Mat& tvec) {
        float cx = 4.5f;
        float cy = -3.0f;
        float trunk_height = 2.0f;
        float trunk_width = 0.3f;
        
        cv::Scalar trunk_color(50, 100, 50);
        cv::Scalar leaves_color(50, 200, 50);
        
        // Trunk
        cv::Point3f trunk_bl(cx - trunk_width, cy - trunk_width, 0.0f);
        cv::Point3f trunk_br(cx + trunk_width, cy - trunk_width, 0.0f);
        cv::Point3f trunk_fl(cx - trunk_width, cy + trunk_width, 0.0f);
        cv::Point3f trunk_fr(cx + trunk_width, cy + trunk_width, 0.0f);
        
        cv::Point3f trunk_tbl(cx - trunk_width, cy - trunk_width, -trunk_height);
        cv::Point3f trunk_tbr(cx + trunk_width, cy - trunk_width, -trunk_height);
        cv::Point3f trunk_tfl(cx - trunk_width, cy + trunk_width, -trunk_height);
        cv::Point3f trunk_tfr(cx + trunk_width, cy + trunk_width, -trunk_height);
        
        // Draw trunk
        drawLine3D(frame, trunk_bl, trunk_tbl, rvec, tvec, trunk_color, 3);
        drawLine3D(frame, trunk_br, trunk_tbr, rvec, tvec, trunk_color, 3);
        drawLine3D(frame, trunk_fl, trunk_tfl, rvec, tvec, trunk_color, 3);
        drawLine3D(frame, trunk_fr, trunk_tfr, rvec, tvec, trunk_color, 3);
        
        // Leaves - pyramid layers
        float leaf_size = 1.5f;
        for (int layer = 0; layer < 3; layer++) {
            float z_offset = -trunk_height - layer * 0.8f;
            float current_size = leaf_size - layer * 0.3f;
            
            // Four corners at this layer
            cv::Point3f l1(cx - current_size, cy - current_size, z_offset);
            cv::Point3f l2(cx + current_size, cy - current_size, z_offset);
            cv::Point3f l3(cx + current_size, cy + current_size, z_offset);
            cv::Point3f l4(cx - current_size, cy + current_size, z_offset);
            
            cv::Point3f top(cx, cy, z_offset - 1.0f);
            
            // Draw pyramid edges
            drawLine3D(frame, l1, top, rvec, tvec, leaves_color, 2);
            drawLine3D(frame, l2, top, rvec, tvec, leaves_color, 2);
            drawLine3D(frame, l3, top, rvec, tvec, leaves_color, 2);
            drawLine3D(frame, l4, top, rvec, tvec, leaves_color, 2);
            
            // Draw base
            drawLine3D(frame, l1, l2, rvec, tvec, leaves_color, 1);
            drawLine3D(frame, l2, l3, rvec, tvec, leaves_color, 1);
            drawLine3D(frame, l3, l4, rvec, tvec, leaves_color, 1);
            drawLine3D(frame, l4, l1, rvec, tvec, leaves_color, 1);
        }
    }
    
    // 4. Rocket Ship
    void drawRocket(cv::Mat& frame, const cv::Mat& rvec, const cv::Mat& tvec) {
        float cx = 4.5f;
        float cy = -3.0f;
        float body_height = 3.0f;
        float body_radius = 0.5f;
        float nose_height = 1.0f;
        
        // Oscillating height (bouncing effect)
        float bounce = std::abs(std::sin(animation_time * 2.0f)) * 0.5f;
        float base_z = -bounce;
        
        cv::Scalar body_color(180, 180, 180);
        cv::Scalar window_color(100, 200, 255);
        cv::Scalar flame_color(255, 100, 0);
        
        // Body (octagonal)
        int segments = 8;
        std::vector<cv::Point3f> base_points, top_points;
        
        for (int i = 0; i < segments; i++) {
            float angle = i * 2 * CV_PI / segments;
            float x = cx + body_radius * std::cos(angle);
            float y = cy + body_radius * std::sin(angle);
            
            base_points.push_back(cv::Point3f(x, y, base_z));
            top_points.push_back(cv::Point3f(x, y, base_z - body_height));
        }
        
        // Draw body
        for (int i = 0; i < segments; i++) {
            drawLine3D(frame, base_points[i], top_points[i], rvec, tvec, body_color, 2);
            drawLine3D(frame, base_points[i], base_points[(i+1)%segments], 
                      rvec, tvec, body_color, 2);
            drawLine3D(frame, top_points[i], top_points[(i+1)%segments],
                      rvec, tvec, body_color, 2);
        }
        
        // Nose cone
        cv::Point3f nose_tip(cx, cy, base_z - body_height - nose_height);
        for (int i = 0; i < segments; i++) {
            drawLine3D(frame, top_points[i], nose_tip, rvec, tvec, cv::Scalar(200, 50, 50), 2);
        }
        
        // Windows (3 circular windows)
        for (int w = 0; w < 3; w++) {
            float win_z = base_z - 0.5f - w * 0.8f;
            int win_segments = 12;
            std::vector<cv::Point3f> window_points;
            
            for (int i = 0; i <= win_segments; i++) {
                float angle = i * 2 * CV_PI / win_segments;
                window_points.push_back(cv::Point3f(
                    cx + 0.2f * std::cos(angle),
                    cy + body_radius,
                    win_z + 0.2f * std::sin(angle)
                ));
            }
            
            for (size_t i = 0; i < window_points.size() - 1; i++) {
                drawLine3D(frame, window_points[i], window_points[i+1],
                          rvec, tvec, window_color, 2);
            }
        }
        
        // Fins (4 fins)
        float fin_height = 1.0f;
        float fin_width = 0.8f;
        for (int i = 0; i < 4; i++) {
            float angle = i * CV_PI / 2.0f;
            float x_dir = std::cos(angle);
            float y_dir = std::sin(angle);
            
            cv::Point3f fin_base(cx + body_radius * x_dir, cy + body_radius * y_dir, base_z);
            cv::Point3f fin_tip(cx + (body_radius + fin_width) * x_dir,
                               cy + (body_radius + fin_width) * y_dir,
                               base_z);
            cv::Point3f fin_top(cx + body_radius * x_dir,
                               cy + body_radius * y_dir,
                               base_z - fin_height);
            
            drawLine3D(frame, fin_base, fin_tip, rvec, tvec, cv::Scalar(255, 0, 0), 2);
            drawLine3D(frame, fin_tip, fin_top, rvec, tvec, cv::Scalar(255, 0, 0), 2);
            drawLine3D(frame, fin_top, fin_base, rvec, tvec, cv::Scalar(255, 0, 0), 2);
        }
        
        // Flame (animated)
        if (bounce < 0.3f) {
            float flame_height = 1.0f + std::sin(animation_time * 10.0f) * 0.3f;
            for (int i = 0; i < segments; i++) {
                cv::Point3f flame_end(
                    cx + body_radius * 0.3f * std::cos(i * 2 * CV_PI / segments),
                    cy + body_radius * 0.3f * std::sin(i * 2 * CV_PI / segments),
                    base_z + flame_height
                );
                drawLine3D(frame, base_points[i], flame_end, rvec, tvec, flame_color, 2);
            }
        }
        
        animation_time += 0.03f;
    }
};

int main() {
    VirtualObjectAR ar(9, 6, 1.0f);
    
    if (!ar.loadCalibration("calibration.xml")) {
        std::cerr << "Error: Cannot load calibration file" << std::endl;
        return -1;
    }
    
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open camera" << std::endl;
        return -1;
    }
    
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "VIRTUAL OBJECT AR SYSTEM" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    std::cout << "\nVirtual Objects:" << std::endl;
    std::cout << "  '1' - Floating House (asymmetric for debugging)" << std::endl;
    std::cout << "  '2' - Rotating Windmill" << std::endl;
    std::cout << "  '3' - 3D Tree" << std::endl;
    std::cout << "  '4' - Rocket Ship (animated)" << std::endl;
    std::cout << "  's' - Save screenshot" << std::endl;
    std::cout << "  'q' - Quit" << std::endl;
    std::cout << std::string(70, '=') << std::endl << std::endl;
    
    int mode = 1;
    int screenshot_count = 0;
    cv::Mat frame, rvec, tvec;
    
    while (true) {
        cap >> frame;
        if (frame.empty()) break;
        
        if (ar.detectAndEstimatePose(frame, rvec, tvec)) {
            // Draw selected virtual object
            switch (mode) {
                case 1: ar.drawFloatingHouse(frame, rvec, tvec); break;
                case 2: ar.drawWindmill(frame, rvec, tvec); break;
                case 3: ar.drawTree(frame, rvec, tvec); break;
                case 4: ar.drawRocket(frame, rvec, tvec); break;
            }
            
            cv::putText(frame, "Target detected", cv::Point(10, 30),
                       cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        } else {
            cv::putText(frame, "No target detected", cv::Point(10, 30),
                       cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
        }
        
        std::string mode_text[] = {"", "House", "Windmill", "Tree", "Rocket"};
        cv::putText(frame, "Object: " + mode_text[mode], cv::Point(10, 60),
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
        
        cv::imshow("Virtual Object AR", frame);
        
        char key = cv::waitKey(500);
        if (key == 'q' || key == 27) break;
        else if (key >= '1' && key <= '4') {
            mode = key - '0';
            std::cout << "[MODE] Switched to: " << mode_text[mode] << std::endl;
        } else if (key == 's') {
            std::string filename = "ar_screenshot_" + std::to_string(screenshot_count++) + ".png";
            cv::imwrite(filename, frame);
            std::cout << "[SAVED] " << filename << std::endl;
        }
    }
    
    cap.release();
    cv::destroyAllWindows();
    return 0;
}