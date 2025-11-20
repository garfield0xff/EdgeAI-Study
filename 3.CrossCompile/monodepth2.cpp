#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <chrono>

namespace fs = std::filesystem;

class MonoDepth2 {
private:
    torch::jit::script::Module encoder;
    torch::jit::script::Module depth_decoder;
    int feed_width = 640;
    int feed_height = 192;
    torch::Device device;
    
public:
    MonoDepth2(const std::string& encoder_path, 
               const std::string& decoder_path,
               bool use_cuda = false) 
        : device(use_cuda && torch::cuda::cudnn_is_available() ? torch::kCUDA : torch::kCPU) {
        
        std::cout << "Using device: " << (device.is_cuda() ? "CUDA" : "CPU") << std::endl;
        
        // Load torch models  ( 저희는 torchscripts 사용 예정입니다 )
        try {
            std::cout << "Loading encoder from: " << encoder_path << std::endl;
            encoder = torch::jit::load(encoder_path);
            encoder.to(device);
            encoder.eval();
            
            std::cout << "Loading decoder from: " << decoder_path << std::endl;
            depth_decoder = torch::jit::load(decoder_path);
            depth_decoder.to(device);
            depth_decoder.eval();
            
            std::cout << "Models loaded successfully!" << std::endl;
        } catch (const c10::Error& e) {
            std::cerr << "Error loading models: " << e.what() << std::endl;
            throw;
        }
    }
    
    cv::Mat preprocessImage(const cv::Mat& image) {
        cv::Mat resized;
        cv::resize(image, resized, cv::Size(feed_width, feed_height));
        
        cv::Mat rgb;
        if (image.channels() == 4) {
            cv::cvtColor(resized, rgb, cv::COLOR_BGRA2RGB);
        } else if (image.channels() == 3) {
            cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
        } else {
            cv::cvtColor(resized, rgb, cv::COLOR_GRAY2RGB);
        }
        
        return rgb;
    }
    
    torch::Tensor matToTensor(const cv::Mat& mat) {
        cv::Mat float_mat;
        mat.convertTo(float_mat, CV_32F, 1.0/255.0);
        
        auto tensor = torch::from_blob(float_mat.data, 
                                       {1, mat.rows, mat.cols, mat.channels()}, 
                                       torch::kFloat32);
        
        // Convert from NHWC to NCHW 
        // 캐시 효율떄문에 변경하는겁니다. 배치 사이즈에서 채널로 loop 돌리는게 일반적이니까요. 민기님 한번 들여다 보시면 좋을듯
        // 현재는 안해도 상관은 없습니다.
        tensor = tensor.permute({0, 3, 1, 2});
        tensor = tensor.to(device);
        
        return tensor.clone();
    }
    
    cv::Mat tensorToMat(const torch::Tensor& tensor) {
        auto t = tensor.squeeze().detach().cpu();
        
        // Handle both single channel and multi-channel tensors
        cv::Mat mat;
        if (t.dim() == 2) {
            // Single channel depth map
            // grayscale 같은 흑백 이미지는 2차원 텐서를 사용하겠죠 
            mat = cv::Mat(t.size(0), t.size(1), CV_32F, t.data_ptr<float>());
        } else if (t.dim() == 3) {
            // Multi-channel (e.g., RGB)
            t = t.permute({1, 2, 0});
            mat = cv::Mat(t.size(0), t.size(1), CV_32FC3, t.data_ptr<float>());
        }
        
        return mat.clone();
    }
    
    cv::Mat applyColormap(const cv::Mat& depth_map) {
        // Normalize depth map
        double min_val, max_val;
        cv::minMaxLoc(depth_map, &min_val, &max_val);
        
        cv::Mat normalized;
        depth_map.convertTo(normalized, CV_8U, 255.0 / (max_val - min_val), -min_val * 255.0 / (max_val - min_val));
        
        cv::Mat colorized;
        cv::applyColorMap(normalized, colorized, cv::COLORMAP_MAGMA);
        
        return colorized;
    }
    
    std::pair<cv::Mat, cv::Mat> processImage(const cv::Mat& input_image) {
        auto start = std::chrono::high_resolution_clock::now();
        
        // Preprocess
        cv::Mat preprocessed = preprocessImage(input_image);
        torch::Tensor input_tensor = matToTensor(preprocessed);
        
        // Inference
        torch::NoGradGuard no_grad;
        
        // Encoder forward pass
        auto encoder_output = encoder.forward({input_tensor});
        
        // Decoder forward pass  
        torch::IValue decoder_output;
        
        // Check if encoder output is tuple (from wrapped model)
        if (encoder_output.isTuple()) {
            // Unpack tuple elements for decoder
            auto tuple_elements = encoder_output.toTuple()->elements();
            decoder_output = depth_decoder.forward(tuple_elements);
        } else if (encoder_output.isList()) {
            // Convert list to vector for decoder
            auto list_elements = encoder_output.toList();
            std::vector<torch::jit::IValue> vec_elements;
            for (const auto& elem : list_elements) {
                vec_elements.push_back(elem);
            }
            decoder_output = depth_decoder.forward(vec_elements);
        } else {
            // Single tensor output
            decoder_output = depth_decoder.forward({encoder_output});
        }
        
        // Extract disparity from output (wrapped decoder returns tensor directly)
        torch::Tensor disp;
        if (decoder_output.isTensor()) {
            disp = decoder_output.toTensor();
        } else if (decoder_output.isTuple()) {
            disp = decoder_output.toTuple()->elements()[0].toTensor();
        } else {
            throw std::runtime_error("Unexpected decoder output type");
        }
        
        // Resize to original dimensions
        auto resized_disp = torch::upsample_bilinear2d(
            disp,
            {input_image.rows, input_image.cols},
            false
        );
        
        // Convert to cv::Mat
        cv::Mat depth_map = tensorToMat(resized_disp);
        cv::Mat colorized = applyColormap(depth_map);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Processing time: " << duration.count() << "ms" << std::endl;
        
        return {depth_map, colorized};
    }
    
    void visualize(const cv::Mat& input_image, const cv::Mat& depth_map, const cv::Mat& colorized) {
        // Create display images
        cv::Mat display_input, display_depth, display_colorized;
        
        // Resize for display
        int display_width = 640;
        int display_height = static_cast<int>(display_width * input_image.rows / static_cast<float>(input_image.cols));
        
        cv::resize(input_image, display_input, cv::Size(display_width, display_height));
        cv::resize(colorized, display_colorized, cv::Size(display_width, display_height));
        
        // Normalize depth for display
        cv::Mat depth_normalized;
        cv::normalize(depth_map, depth_normalized, 0, 255, cv::NORM_MINMAX, CV_8U);
        cv::resize(depth_normalized, display_depth, cv::Size(display_width, display_height));
        cv::applyColorMap(display_depth, display_depth, cv::COLORMAP_MAGMA);
        
        // Stack horizontally
        cv::Mat combined;
        std::vector<cv::Mat> images = {display_input, display_depth, display_colorized};
        cv::hconcat(images, combined);
        
        // Add labels
        cv::putText(combined, "Input", cv::Point(10, 30), 
                   cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
        cv::putText(combined, "Depth", cv::Point(display_width + 10, 30), 
                   cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
        cv::putText(combined, "Colorized", cv::Point(2 * display_width + 10, 30), 
                   cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
        
        // Display
        cv::imshow("MonoDepth2 - Press ESC to exit, SPACE to pause", combined);
    }
    
    void processVideo(const std::string& video_path) {
        cv::VideoCapture cap(video_path);
        if (!cap.isOpened()) {
            std::cerr << "Error opening video: " << video_path << std::endl;
            return;
        }
        
        double fps = cap.get(cv::CAP_PROP_FPS);
        int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
        std::cout << "Video info: " << fps << " FPS, " << total_frames << " frames" << std::endl;
        
        cv::Mat frame;
        int frame_count = 0;
        bool paused = false;
        
        while (true) {
            if (!paused) {
                if (!cap.read(frame)) {
                    std::cout << "End of video" << std::endl;
                    break;
                }
                frame_count++;
                
                std::cout << "Frame " << frame_count << "/" << total_frames << std::endl;
                
                auto [depth_map, colorized] = processImage(frame);
                visualize(frame, depth_map, colorized);
            }
            
            int key = cv::waitKey(30);
            if (key == 27) { // ESC
                break;
            } else if (key == 32) { // SPACE
                paused = !paused;
                std::cout << (paused ? "Paused" : "Resumed") << std::endl;
            }
        }
        
        cap.release();
    }
    
    void processImageSequence(const std::string& directory_path) {
        std::vector<std::string> image_files;
        
        // Collect image files
        for (const auto& entry : fs::directory_iterator(directory_path)) {
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            
            if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp") {
                image_files.push_back(entry.path().string());
            }
        }
        
        if (image_files.empty()) {
            std::cerr << "No images found in: " << directory_path << std::endl;
            return;
        }
        
        std::sort(image_files.begin(), image_files.end());
        std::cout << "Found " << image_files.size() << " images" << std::endl;
        
        size_t current_idx = 0;
        bool playing = false;
        
        while (current_idx < image_files.size()) {
            std::cout << "Processing: " << image_files[current_idx] 
                     << " (" << current_idx + 1 << "/" << image_files.size() << ")" << std::endl;
            
            cv::Mat image = cv::imread(image_files[current_idx]);
            if (image.empty()) {
                std::cerr << "Failed to load: " << image_files[current_idx] << std::endl;
                current_idx++;
                continue;
            }
            
            auto [depth_map, colorized] = processImage(image);
            visualize(image, depth_map, colorized);
            
            int key = cv::waitKey(playing ? 30 : 0);
            
            if (key == 27) { // ESC
                break;
            } else if (key == 32) { // SPACE
                playing = !playing;
                std::cout << (playing ? "Playing" : "Paused") << std::endl;
            } else if (key == 83 || key == 115) { // 'S' or 's' - save
                std::string save_name = "depth_" + std::to_string(current_idx) + ".png";
                cv::imwrite(save_name, colorized);
                std::cout << "Saved: " << save_name << std::endl;
            } else if (!playing) {
                if (key == 81 || key == 113) { // Left arrow or 'q'
                    if (current_idx > 0) current_idx--;
                } else if (key == 83 || key == 115 || key >= 0) { // Right arrow or any other key
                    current_idx++;
                }
            } else if (playing) {
                current_idx++;
            }
        }
    }
};

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <encoder_model.pt> <decoder_model.pt> [image/video/directory]" << std::endl;
        std::cerr << "Controls:" << std::endl;
        std::cerr << "  ESC   - Exit" << std::endl;
        std::cerr << "  SPACE - Play/Pause" << std::endl;
        std::cerr << "  S     - Save current depth map" << std::endl;
        std::cerr << "  Arrow keys - Navigate (when paused)" << std::endl;
        return 1;
    }
    
    std::string encoder_path = argv[1];
    std::string decoder_path = argv[2];
    
    try {
        // Initialize MonoDepth2
        bool use_cuda = torch::cuda::cudnn_is_available();
        MonoDepth2 model(encoder_path, decoder_path, use_cuda);
        
        if (argc == 4) {
            std::string input_path = argv[3];
            
            // Check if input is file or directory
            if (fs::is_directory(input_path)) {
                std::cout << "Processing image sequence from: " << input_path << std::endl;
                model.processImageSequence(input_path);
            } else if (fs::is_regular_file(input_path)) {
                // Check if video or image
                std::string ext = fs::path(input_path).extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                
                if (ext == ".mp4" || ext == ".avi" || ext == ".mov" || ext == ".mkv") {
                    std::cout << "Processing video: " << input_path << std::endl;
                    model.processVideo(input_path);
                } else {
                    std::cout << "Processing single image: " << input_path << std::endl;
                    cv::Mat image = cv::imread(input_path);
                    if (image.empty()) {
                        std::cerr << "Failed to load image: " << input_path << std::endl;
                        return 1;
                    }
                    
                    auto [depth_map, colorized] = model.processImage(image);
                    model.visualize(image, depth_map, colorized);
                    
                    std::cout << "Press any key to exit..." << std::endl;
                    cv::waitKey(0);
                }
            } else {
                std::cerr << "Invalid input path: " << input_path << std::endl;
                return 1;
            }
        } else {
            // Test with dummy image
            std::cout << "No input specified, creating test image..." << std::endl;
            cv::Mat test_image(480, 640, CV_8UC3, cv::Scalar(100, 150, 200));
            
            auto [depth_map, colorized] = model.processImage(test_image);
            model.visualize(test_image, depth_map, colorized);
            
            std::cout << "Press any key to exit..." << std::endl;
            cv::waitKey(0);
        }
        
        cv::destroyAllWindows();
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}