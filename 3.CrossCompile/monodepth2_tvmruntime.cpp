// File: monodepth2_tvmruntime.cpp
// Description: MonoDepth2 inference using LibTorch for preprocessing/postprocessing
//              and TVM runtime for encoder/decoder inference on aarch64 target.
//              Updated for latest TVM: no ndarray.h, use DLTensor directly.

#include <tvm/runtime/c_runtime_api.h>

#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>

#include <torch/torch.h>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <chrono>

namespace fs = std::filesystem;
using Device = DLDevice;

class TVMMonoDepth2 {
private:
    tvm::runtime::Module encoder_mod;
    tvm::runtime::Module decoder_mod;
    tvm::runtime::PackedFunc encoder_set_input;
    tvm::runtime::PackedFunc encoder_run;
    tvm::runtime::PackedFunc encoder_get_output;
    tvm::runtime::PackedFunc decoder_set_input;
    tvm::runtime::PackedFunc decoder_run;
    tvm::runtime::PackedFunc decoder_get_output;
    torch::jit::script::Module encoder;
    torch::jit::script::Module depth_decoder;
    Device device{ kDLCPU, 0 };  // CPU only, CUDA example 가능
    int feed_width = 640;
    int feed_height = 192;

public:
    TVMMonoDepth2(const std::string& encoder_so, const std::string& decoder_so) {
        encoder_mod = tvm::runtime::Module::LoadFromFile(encoder_so);
        decoder_mod = tvm::runtime::Module::LoadFromFile(decoder_so);

        tvm::runtime::Module encoder_exec = encoder_mod.GetFunction("default")(device);
        tvm::runtime::Module decoder_exec = decoder_mod.GetFunction("default")(device);

        encoder_set_input = encoder_exec.GetFunction("set_input");
        encoder_run = encoder_exec.GetFunction("run");
        encoder_get_output = encoder_exec.GetFunction("get_output");

        decoder_set_input = decoder_exec.GetFunction("set_input");
        decoder_run = decoder_exec.GetFunction("run");
        decoder_get_output = decoder_exec.GetFunction("get_output");

        std::cout << "✅ TVM AOT modules loaded successfully!\n";
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

        torch::NoGradGuard no_grad;

        // ✅ TVM 경로로 인퍼런스
        torch::Tensor disp = runTVM(input_tensor);

        // Resize to original dimensions
        auto resized_disp = torch::upsample_bilinear2d(
            disp, {input_image.rows, input_image.cols}, false);

        cv::Mat depth_map = tensorToMat(resized_disp);
        cv::Mat colorized = applyColormap(depth_map);

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Processing time: " << duration.count() << "ms\n";

        return {depth_map, colorized};
    }

    // Run TVM module
    torch::Tensor runTVM(const torch::Tensor& input) {
        std::cout << "runTVM start!!" << "\n";
        // Create tvm NDArray
        tvm::runtime::NDArray input_arr = createNDArray(input);
        // set input_arr to encoder
        encoder_set_input("input_0", input_arr);
        // run encoder
        encoder_run();
        // allocate encoder output
        tvm::runtime::NDArray encoder_output = encoder_get_output(0);

        // Get encoder output shape
        std::vector<int64_t> enc_shape;
        for (int i = 0; i < encoder_output->ndim; ++i) {
            enc_shape.push_back(encoder_output->shape[i]);
        }

        // Copy encoder output to torch tensor
        auto enc_tensor = torch::empty(enc_shape, torch::kFloat32);
        encoder_output.CopyToBytes(enc_tensor.data_ptr(), enc_tensor.numel() * sizeof(float));

        // Decoder inference
        tvm::runtime::NDArray dec_input_arr = createNDArray(enc_tensor);
        decoder_set_input("input", dec_input_arr);
        decoder_run();
        tvm::runtime::NDArray decoder_output = decoder_get_output(0);

        // Get decoder output shape
        std::vector<int64_t> dec_shape;
        for (int i = 0; i < decoder_output->ndim; ++i) {
            dec_shape.push_back(decoder_output->shape[i]);
        }

        // Copy decoder output to torch tensor
        auto dec_tensor = torch::empty(dec_shape, torch::kFloat32);
        decoder_output.CopyToBytes(dec_tensor.data_ptr(), dec_tensor.numel() * sizeof(float));
        std::cout << "runTVM end!!" << "\n";
        return dec_tensor;
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

    // Create NDArray from torch::Tensor
    tvm::runtime::NDArray createNDArray(const torch::Tensor& t) {
        std::vector<int64_t> shape;
        for (int i = 0; i < t.dim(); ++i) {
            std::cout << "i: " << i << "size: " << t.size(i) << "\n";
            shape.push_back(t.size(i));
        }

        DLDataType dtype = getDLDataType(t);
        DLDevice dev = getDLDevice(t);

        // Create NDArray with TVM's memory management
        tvm::runtime::NDArray arr = tvm::runtime::NDArray::Empty(shape, dtype, dev);

        // Copy data from torch tensor to NDArray
        arr.CopyFromBytes(t.data_ptr(), t.numel() * t.element_size());

        return arr;
    }

    DLDataType getDLDataType(const torch::Tensor& t) {
        DLDataType dtype;
        dtype.lanes = 1;

        switch (t.scalar_type()) {
            case torch::kFloat32:
                dtype.code = kDLFloat;
                dtype.bits = 32;
                break;
            case torch::kFloat64:
                dtype.code = kDLFloat;
                dtype.bits = 64;
                break;
            case torch::kFloat16:
                dtype.code = kDLFloat;
                dtype.bits = 16;
                break;
            case torch::kInt32:
                dtype.code = kDLInt;
                dtype.bits = 32;
                break;
            case torch::kInt64:
                dtype.code = kDLInt;
                dtype.bits = 64;
                break;
            case torch::kInt16:
                dtype.code = kDLInt;
                dtype.bits = 16;
                break;
            case torch::kInt8:
                dtype.code = kDLInt;
                dtype.bits = 8;
                break;
            case torch::kUInt8:
                dtype.code = kDLUInt;
                dtype.bits = 8;
                break;
            default:
                throw std::runtime_error("Unsupported data type");
        }

        return dtype;
    }

    // DLTensor 메모리 해제 (shape, strides만 해제, 데이터는 torch가 관리)
    void freeDLTensor(DLTensor* dl_tensor) {
        if (dl_tensor) {
            delete[] dl_tensor->shape;
            delete[] dl_tensor->strides;
            delete dl_tensor;
        }
    }

    // torch::Tensor의 DLDevice 변환
    DLDevice getDLDevice(const torch::Tensor& t) {
        DLDevice device;

        if (t.is_cuda()) {
            device.device_type = kDLCUDA;
            device.device_id = t.get_device();
        } else if (t.is_cpu()) {
            device.device_type = kDLCPU;
            device.device_id = 0;
        } else {
            throw std::runtime_error("Unsupported device type");
        }

        return device;
    }

};

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <encoder_so> <decoder_so> [image/video/dir]\n";
        return 1;
    }
    std::cout << "========================================" << std::endl;
    std::cout << "Function Start!!" << std::endl;
    std::string encoder_so = argv[1];
    std::string decoder_so = argv[2];

    try {
        TVMMonoDepth2 app(encoder_so, decoder_so);

        if (argc == 4) {
            std::string input = argv[3];
            if (fs::is_directory(input)) app.processImageSequence(input);
            else if (fs::is_regular_file(input)) {
                cv::Mat image = cv::imread(input);
                if (image.empty()) { std::cerr << "Failed to load image\n"; return 1; }
                auto [depth, color] = app.processImage(image);
                app.visualize(image, depth, color);
                cv::waitKey(0);
            }
        } else {
            cv::Mat test_img(480, 640, CV_8UC3, cv::Scalar(100,150,200));
            auto [depth, color] = app.processImage(test_img);
            app.visualize(test_img, depth, color);
            cv::waitKey(0);
        }
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }

    cv::destroyAllWindows();
    return 0;
}
