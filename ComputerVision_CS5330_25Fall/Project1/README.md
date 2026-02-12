<!-- 依赖：确保已安装 OpenCV（建议 4.x 版本），并配置好 VS Code 的编译环境（如 tasks.json 和 c_cpp_properties.json）。
编译命令示例（需替换 OpenCV 路径）： -->

# 编译任务1
g++ imgDisplay.cpp -o imgDisplay `pkg-config --cflags --libs opencv4`
# 编译任务2-6（需链接filter.cpp）
g++ vidDisplay.cpp filter.cpp -o vidDisplay `pkg-config --cflags --libs opencv4`