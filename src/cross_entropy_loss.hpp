#pragma once
#include <Eigen/Dense>

namespace loss {
class CrossEntropyLoss {
#pragma region Reductions
  static std::unordered_map<std::string, std::function<double(Eigen::MatrixXd)>>
      reductions;
  std::string reduction;
#pragma endregion Reductions

public:
  CrossEntropyLoss(std::string reduction = "mean") {
    this->setReduction(reduction);
  }

#pragma region Properties
#pragma region Reduction
  /*
    Get the reduction method.
  */
  std::string getReduction() const;

  /*
    Set the reduction method.
  */
  void setReduction(std::string reduction);
#pragma endregion Reduction
#pragma endregion Properties
};
} // namespace loss