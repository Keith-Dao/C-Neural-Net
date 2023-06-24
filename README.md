# Neural Net using C++

This project is an implementation of a simple neural network with two hidden layers, made specifically to classify digits in the MNIST dataset.

For the complete technical overview see [link](https://github.com/Keith-Dao/Neural-Net-From-Scratch/blob/main/README.md).

## 1. Dependencies

The following libraries are required.

| Library    | Version | Link                                                                  |
| ---------- | ------- | --------------------------------------------------------------------- |
| Eigen      | 3.4.0   | https://gitlab.com/libeigen/eigen/-/releases/3.4.0                    |
| JSON       | 3.11.2  | https://github.com/nlohmann/json/releases/tag/v3.11.2                 |
| yaml-cpp   | 0.6.3   | https://github.com/jbeder/yaml-cpp/releases/tag/yaml-cpp-0.6.3        |
| matplot++  | 1.1.0   | https://github.com/alandefreitas/matplotplusplus/releases/tag/v1.1.0  |
| indicators | 2.3     | https://github.com/p-ranav/indicators/releases/tag/v2.3               |
| OpenCV     | 4.7.0   | https://github.com/opencv/opencv/releases/tag/4.7.0                   |
| tabulate   | 1.5     | https://github.com/p-ranav/tabulate/releases/tag/v1.5                 |
| Termcolor  | 2.1.0   | https://github.com/ikalnytskyi/termcolor/archive/refs/tags/v2.1.0.zip |

The following are also required.

| Dependencies | Version |
| ------------ | ------- |
| g++          | 12.2.1  |
| cmake        | 3.26.3  |
| gnuplot      | 5.4     |

## 2. Setup

For each library above:

1. Download the required libraries above
2. For each library, run `cmake -B build`
3. Change directories to `build`
4. Run `make install`, this would require elevated permissions

For matplot++:

1. Follow the instructions https://alandefreitas.github.io/matplotplusplus/integration/install/build-from-source/dependencies/

For the project run:

1. Run `cmake -B build`
2. Change directories to `build`
3. Run `make`
