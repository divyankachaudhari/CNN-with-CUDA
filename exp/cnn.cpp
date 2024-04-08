#include <iostream>
#include <fstream>
#include <vector>
#include <torch/torch.h>

// Function to read data from file and convert to tensor
torch::Tensor read_file_to_tensor(const std::string& filename) {
    std::vector<float> data;
    try {
        std::ifstream file(filename);
        float value;
        while (file >> value) {
            data.push_back(value);
        }
        file.close();
        return torch::tensor(data);
    } catch (const std::exception& e) {
        std::cerr << "Failed to open " << filename << ": " << e.what() << std::endl;
        return torch::Tensor();
    }
}

// Function to load weights and biases from file
std::pair<torch::Tensor, torch::Tensor> load_weights_biases(const std::string& filename, const std::vector<int64_t>& shape) {
    auto data = torch::from_file(filename);
    auto weights = data.narrow(0, 0, data.size(0) - shape[0]).view(shape);
    auto biases = data.narrow(0, data.size(0) - shape[0], shape[0]);
    return std::make_pair(weights, biases);
}

// LeNet model class
class LeNet : public torch::nn::Module {
public:
    LeNet() {
        auto conv1_params = load_weights_biases("trained_weights/conv1.txt", {20, 1, 5, 5});
        conv1_weights = register_parameter("conv1_weights", conv1_params.first);
        conv1_biases = register_parameter("conv1_biases", conv1_params.second);

        auto conv2_params = load_weights_biases("trained_weights/conv2.txt", {50, 20, 5, 5});
        conv2_weights = register_parameter("conv2_weights", conv2_params.first);
        conv2_biases = register_parameter("conv2_biases", conv2_params.second);

        auto fc1_params = load_weights_biases("trained_weights/fc1.txt", {500, 800});
        fc1_weights = register_parameter("fc1_weights", fc1_params.first);
        fc1_biases = register_parameter("fc1_biases", fc1_params.second);

        auto fc2_params = load_weights_biases("trained_weights/fc2.txt", {10, 500});
        fc2_weights = register_parameter("fc2_weights", fc2_params.first);
        fc2_biases = register_parameter("fc2_biases", fc2_params.second);
    }

    torch::Tensor forward(torch::Tensor x) {
        // Conv1
        x = torch::conv2d(x, conv1_weights, conv1_biases, {1, 1}, {0, 0});
        x = torch::max_pool2d(x, {2, 2}, {2, 2});

        // Conv2
        x = torch::conv2d(x, conv2_weights, conv2_biases, {1, 1}, {0, 0});
        x = torch::max_pool2d(x, {2, 2}, {2, 2});

        // Flatten
        x = x.view({x.size(0), -1});

        // FC1
        x = torch::linear(x, fc1_weights, fc1_biases);
        x = torch::relu(x);

        // FC2
        x = torch::linear(x, fc2_weights, fc2_biases);

        // Softmax
        x = torch::softmax(x, 1);

        return x;
    }

    std::pair<std::vector<float>, std::vector<int64_t>> predict(torch::Tensor x) {
        auto probabilities = forward(x);
        auto topk_output = probabilities.topk(5);
        auto top_probabilities = topk_output.values.squeeze(0).detach().cpu().numpy();
        auto top_indices = topk_output.indices.squeeze(0).detach().cpu().numpy();
        return std::make_pair(std::vector<float>(top_probabilities, top_probabilities + 5),
                              std::vector<int64_t>(top_indices, top_indices + 5));
    }

private:
    torch::Tensor conv1_weights, conv1_biases;
    torch::Tensor conv2_weights, conv2_biases;
    torch::Tensor fc1_weights, fc1_biases;
    torch::Tensor fc2_weights, fc2_biases;
};

int main() {
    torch::manual_seed(1);
    auto input_image = read_file_to_tensor("output.txt").view({1, 1, 28, 28});
    LeNet lenet;
    auto prediction = lenet.predict(input_image);

    // Print top 5 probabilities and their class labels
    auto top_probabilities = prediction.first;
    auto top_indices = prediction.second;
    for (size_t i = 0; i < 5; ++i) {
        std::cout << top_probabilities[i] << "% class " << top_indices[i] << std::endl;
    }

    return 0;
}
