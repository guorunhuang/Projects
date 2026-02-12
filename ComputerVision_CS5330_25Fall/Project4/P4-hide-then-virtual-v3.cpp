#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <vector>
#include <cmath>

class SimplifiedARSystem {
private:
    cv::Mat camera_matrix;
    cv::Mat dist_coeffs;
    cv::Size board_size;
    float square_size;
    std::vector<cv::Point3f> object_points;
    float animation_time;

public:
    // Textures (public so main can access them)
    cv::Mat wood_texture;
    cv::Mat grass_texture;
    
public:
    SimplifiedARSystem(int cols = 9, int rows = 6, float sq_size = 1.0f)
        : board_size(cols, rows), square_size(sq_size), animation_time(0.0f) {
        
        for (int i = 0; i < board_size.height; i++) {
            for (int j = 0; j < board_size.width; j++) {
                object_points.push_back(cv::Point3f(j * sq_size, -i * sq_size, 0.0f));
            }
        }
        
        generateTextures();
    }
    
    bool loadCalibration(const std::string& filename) {
        cv::FileStorage fs(filename, cv::FileStorage::READ);
        if (!fs.isOpened()) return false;
        
        fs["camera_matrix"] >> camera_matrix;
        fs["distortion_coefficients"] >> dist_coeffs;
        fs.release();
        
        std::cout << "[LOADED] Calibration" << std::endl;
        return true;
    }
    
    void generateTextures() {
        int tex_size = 512;
        
        // Wood texture
        wood_texture = cv::Mat(tex_size, tex_size, CV_8UC3);
        for (int i = 0; i < tex_size; i++) {
            for (int j = 0; j < tex_size; j++) {
                float noise = (std::sin(j * 0.1f) + 1.0f) * 0.5f;
                int brown = 60 + noise * 80;
                wood_texture.at<cv::Vec3b>(i, j) = cv::Vec3b(20, brown/2, brown);
            }
        }
        
        // Grass texture
        grass_texture = cv::Mat(tex_size, tex_size, CV_8UC3);
        cv::randu(grass_texture, cv::Scalar(20, 80, 20), cv::Scalar(40, 140, 40));
        cv::GaussianBlur(grass_texture, grass_texture, cv::Size(5, 5), 0);
        
        std::cout << "[GENERATED] Procedural textures" << std::endl;
    }
    
    bool detectAndEstimatePose(cv::Mat& frame, cv::Mat& rvec, cv::Mat& tvec,
                               std::vector<cv::Point2f>& corners) {
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        
        bool found = cv::findChessboardCorners(gray, board_size, corners,
            cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK);
        
        if (!found) return false;
        
        cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1),
            cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.1));
        
        return cv::solvePnP(object_points, corners, camera_matrix, dist_coeffs, rvec, tvec);
    }
    
    // Helper: Draw 3D line
    void drawLine3D(cv::Mat& frame, const cv::Point3f& p1, const cv::Point3f& p2,
                    const cv::Mat& rvec, const cv::Mat& tvec,
                    const cv::Scalar& color, int thickness = 2) {
        std::vector<cv::Point3f> points_3d = {p1, p2};
        std::vector<cv::Point2f> points_2d;
        cv::projectPoints(points_3d, rvec, tvec, camera_matrix, dist_coeffs, points_2d);
        cv::line(frame, points_2d[0], points_2d[1], color, thickness, cv::LINE_AA);
    }
    
    // Virtual Object: Floating Platform with Objects
    void drawFloatingPlatform(cv::Mat& frame, const cv::Mat& rvec, const cv::Mat& tvec) {
        float cx = 4.5f;
        float cy = -3.0f;
        float platform_z = -1.5f;
        float platform_size = 5.0f;
        float thickness = 0.3f;
        
        // Platform base
        std::vector<cv::Point3f> platform_base = {
            {cx - platform_size/2, cy - platform_size/2, platform_z},
            {cx + platform_size/2, cy - platform_size/2, platform_z},
            {cx + platform_size/2, cy + platform_size/2, platform_z},
            {cx - platform_size/2, cy + platform_size/2, platform_z}
        };
        
        // Platform top
        std::vector<cv::Point3f> platform_top;
        for (const auto& pt : platform_base) {
            platform_top.push_back(cv::Point3f(pt.x, pt.y, pt.z - thickness));
        }
        
        cv::Scalar platform_color(100, 100, 150);
        
        // Draw platform edges
        for (size_t i = 0; i < 4; i++) {
            drawLine3D(frame, platform_base[i], platform_base[(i+1)%4], rvec, tvec, platform_color, 3);
            drawLine3D(frame, platform_top[i], platform_top[(i+1)%4], rvec, tvec, platform_color, 3);
            drawLine3D(frame, platform_base[i], platform_top[i], rvec, tvec, platform_color, 3);
        }
        
        // Draw small cubes on platform
        float cube_size = 0.8f;
        std::vector<cv::Point3f> cube_positions = {
            {cx - 1.5f, cy - 1.5f, platform_z - thickness - cube_size},
            {cx + 1.5f, cy - 1.5f, platform_z - thickness - cube_size},
            {cx, cy + 1.5f, platform_z - thickness - cube_size}
        };
        
        for (const auto& pos : cube_positions) {
            drawCube(frame, pos, cube_size, rvec, tvec, cv::Scalar(180, 100, 100));
        }
    }
    
    void drawCube(cv::Mat& frame, const cv::Point3f& center, float size,
                  const cv::Mat& rvec, const cv::Mat& tvec, const cv::Scalar& color) {
        float hs = size / 2.0f;
        std::vector<cv::Point3f> vertices = {
            {center.x - hs, center.y - hs, center.z},
            {center.x + hs, center.y - hs, center.z},
            {center.x + hs, center.y + hs, center.z},
            {center.x - hs, center.y + hs, center.z},
            {center.x - hs, center.y - hs, center.z - size},
            {center.x + hs, center.y - hs, center.z - size},
            {center.x + hs, center.y + hs, center.z - size},
            {center.x - hs, center.y + hs, center.z - size}
        };
        
        // Draw edges
        for (int i = 0; i < 4; i++) {
            drawLine3D(frame, vertices[i], vertices[(i+1)%4], rvec, tvec, color, 2);
            drawLine3D(frame, vertices[i+4], vertices[(i+1)%4+4], rvec, tvec, color, 2);
            drawLine3D(frame, vertices[i], vertices[i+4], rvec, tvec, color, 2);
        }
    }
    
    // Effect 1: Background Blend
    void morphIntoBackground(cv::Mat& frame, const std::vector<cv::Point2f>& corners) {
        if (corners.size() < 4) return;
        
        cv::Rect rect = cv::boundingRect(corners);
        rect &= cv::Rect(0, 0, frame.cols, frame.rows);
        if (rect.width <= 0 || rect.height <= 0) return;
        
        // Sample background colors around the target
        std::vector<cv::Vec3b> bg_colors;
        int margin = 20;
        
        for (int i = std::max(0, rect.y - margin); i < std::min(frame.rows, rect.y + rect.height + margin); i += 10) {
            if (i < 0 || i >= frame.rows) continue;
            
            int left = std::max(0, rect.x - margin);
            int right = std::min(frame.cols - 1, rect.x + rect.width + margin);
            
            if (left >= 0 && left < frame.cols)
                bg_colors.push_back(frame.at<cv::Vec3b>(i, left));
            if (right >= 0 && right < frame.cols)
                bg_colors.push_back(frame.at<cv::Vec3b>(i, right));
        }
        
        if (bg_colors.empty()) return;
        
        // Calculate average background color
        cv::Vec3f avg_color(0, 0, 0);
        for (const auto& c : bg_colors) {
            avg_color += cv::Vec3f(c);
        }
        avg_color /= (float)bg_colors.size();
        
        // Create mask
        cv::Mat mask = cv::Mat::zeros(frame.size(), CV_8UC1);
        std::vector<cv::Point> contour;
        contour.push_back(corners[0]);
        contour.push_back(corners[board_size.width - 1]);
        contour.push_back(corners.back());
        contour.push_back(corners[corners.size() - board_size.width]);
        cv::fillConvexPoly(mask, contour, cv::Scalar(255));
        
        // Generate noise properly
        cv::RNG rng(12345);
        
        for (int i = 0; i < frame.rows; i++) {
            for (int j = 0; j < frame.cols; j++) {
                if (mask.at<uchar>(i, j) > 0) {
                    // Add random noise to average color
                    cv::Vec3b color;
                    for (int c = 0; c < 3; c++) {
                        int val = (int)avg_color[c] + rng.uniform(-20, 20);
                        color[c] = cv::saturate_cast<uchar>(val);
                    }
                    frame.at<cv::Vec3b>(i, j) = color;
                }
            }
        }
        
        // Blur for seamless blending
        cv::Mat blurred = frame.clone();
        cv::GaussianBlur(blurred, blurred, cv::Size(15, 15), 0);
        blurred.copyTo(frame, mask);
    }
    
    // Effect 2 & 3: Texture Replacement (Wood/Grass)
    void replaceWithTexture(cv::Mat& frame, const std::vector<cv::Point2f>& corners,
                           const cv::Mat& texture) {
        if (corners.size() < 4) return;
        
        // Source points (texture corners)
        std::vector<cv::Point2f> src_points = {
            cv::Point2f(0, 0),
            cv::Point2f(texture.cols - 1, 0),
            cv::Point2f(texture.cols - 1, texture.rows - 1),
            cv::Point2f(0, texture.rows - 1)
        };
        
        // Destination points (checkerboard corners)
        std::vector<cv::Point2f> dst_points = {
            corners[0],
            corners[board_size.width - 1],
            corners.back(),
            corners[corners.size() - board_size.width]
        };
        
        // Compute homography
        cv::Mat H = cv::getPerspectiveTransform(src_points, dst_points);
        
        // Warp texture
        cv::Mat warped = cv::Mat::zeros(frame.size(), frame.type());
        cv::warpPerspective(texture, warped, H, frame.size());
        
        // Create mask
        cv::Mat mask = cv::Mat::zeros(frame.size(), CV_8UC1);
        std::vector<cv::Point> contour(dst_points.begin(), dst_points.end());
        cv::fillConvexPoly(mask, contour, cv::Scalar(255));
        
        // Blend with feathering
        cv::Mat mask_blur;
        cv::GaussianBlur(mask, mask_blur, cv::Size(21, 21), 0);
        
        for (int i = 0; i < frame.rows; i++) {
            for (int j = 0; j < frame.cols; j++) {
                float alpha = mask_blur.at<uchar>(i, j) / 255.0f;
                if (alpha > 0.01f) {
                    cv::Vec3b orig = frame.at<cv::Vec3b>(i, j);
                    cv::Vec3b warp = warped.at<cv::Vec3b>(i, j);
                    frame.at<cv::Vec3b>(i, j) = alpha * warp + (1.0f - alpha) * orig;
                }
            }
        }
    }
};

int main() {
    SimplifiedARSystem ar(9, 6, 1.0f);
    
    if (!ar.loadCalibration("calibration.xml")) {
        std::cerr << "Error: Cannot load calibration" << std::endl;
        return -1;
    }
    
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open camera" << std::endl;
        return -1;
    }
    
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "SIMPLIFIED AR WITH HIDDEN TARGET" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    std::cout << "\nTarget Hiding Effects:" << std::endl;
    std::cout << "  '1' - Background Blend (target blends into surroundings)" << std::endl;
    std::cout << "  '2' - Wood Texture (wooden floor effect)" << std::endl;
    std::cout << "  '3' - Grass Texture (outdoor ground effect)" << std::endl;
    std::cout << "  '0' - Original (no hiding, AR only)" << std::endl;
    std::cout << "\nOther Controls:" << std::endl;
    std::cout << "  's' - Save screenshot" << std::endl;
    std::cout << "  'q' - Quit" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    std::cout << "\nAll modes include floating platform with cubes!" << std::endl;
    std::cout << std::string(70, '=') << std::endl << std::endl;
    
    int mode = 0;
    int screenshot_count = 0;
    
    std::string mode_names[] = {
        "Original (AR Only)",
        "Background Blend + AR",
        "Wood Texture + AR",
        "Grass Texture + AR"
    };
    
    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) break;
        
        cv::Mat display = frame.clone();
        cv::Mat rvec, tvec;
        std::vector<cv::Point2f> corners;
        
        if (ar.detectAndEstimatePose(display, rvec, tvec, corners)) {
            // First hide/replace the target
            switch (mode) {
                case 1: 
                    ar.morphIntoBackground(display, corners); 
                    break;
                case 2: 
                    ar.replaceWithTexture(display, corners, ar.wood_texture); 
                    break;
                case 3: 
                    ar.replaceWithTexture(display, corners, ar.grass_texture); 
                    break;
                default: 
                    break; // No hiding
            }
            
            // Then draw virtual object on top
            ar.drawFloatingPlatform(display, rvec, tvec);
            
            cv::putText(display, "Target detected - AR Active", cv::Point(10, 30),
                       cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        } else {
            cv::putText(display, "No target detected", cv::Point(10, 30),
                       cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
        }
        
        cv::putText(display, "Mode: " + mode_names[mode], cv::Point(10, 60),
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
        
        cv::imshow("Simplified AR System", display);
        
        char key = cv::waitKey(1);
        if (key == 'q' || key == 27) {
            break;
        } else if (key >= '0' && key <= '3') {
            mode = key - '0';
            std::cout << "[MODE] " << mode_names[mode] << std::endl;
        } else if (key == 's') {
            std::string filename = "ar_simplified_" + std::to_string(screenshot_count++) + ".png";
            cv::imwrite(filename, display);
            std::cout << "[SAVED] " << filename << std::endl;
        }
    }
    
    cap.release();
    cv::destroyAllWindows();
    return 0;
}