# Neural Net using C++

This project is an implementation of a simple neural network with two hidden layers, made specifically to classify digits in the MNIST dataset.

For the complete technical overview see [link](https://github.com/Keith-Dao/Neural-Net-From-Scratch/blob/main/README.md).

## 1. Dependencies

The following libraries are required.

| Library  | Version | Link                                                           |
| -------- | ------- | -------------------------------------------------------------- |
| Eigen    | 3.4.0   | https://eigen.tuxfamily.org/index.php?title=Main_Page          |
| JSON     | 3.11.2  | https://github.com/nlohmann/json/releases/tag/v3.11.2          |
| yaml-cpp | 0.6.3   | https://github.com/jbeder/yaml-cpp/releases/tag/yaml-cpp-0.6.3 |

## 2. Setup

For each library above:

1. Download the required libraries above
2. For each library, run `cmake -B build`
3. Change directories to `build`
4. Run `make install`, this would require elevated permissions

For the project run:

1. Run `cmake -B build`
2. Change directories to `build`
3. Run `make`
