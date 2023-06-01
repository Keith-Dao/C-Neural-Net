#pragma once
#include <Eigen/Dense>
#include <memory>

namespace loss {
class CrossEntropyLoss {
#pragma region Reductions
  static std::unordered_map<std::string, std::function<double(Eigen::MatrixXd)>>
      reductions;
  std::string reduction;
#pragma endregion Reductions

  std::shared_ptr<Eigen::MatrixXi> targets = nullptr;
  std::shared_ptr<Eigen::MatrixXd> probabilities = nullptr;

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

#pragma region Load
// TODO
#pragma endregion Load

#pragma region Save
// TODO
#pragma endregion Save

#pragma region Forward
  /*
    Calculate the cross entropy loss given the logits and the one hot encoded
    labels.
  */
  double forward(const Eigen::MatrixXd &logits, const Eigen::MatrixXi &targets);

  /*
    Calculate the cross entropy loss given the logits and the target class
    labels.
  */
  double forward(const Eigen::MatrixXd &logits,
                 const std::vector<int> &targets);
#pragma endregion Forward

#pragma region Backward
// TODO
#pragma endregion Backward

#pragma region Builtins
  /*
    Calculate the cross entropy loss given the logits and the one hot encoded
    labels.
  */
  double operator()(const Eigen::MatrixXd &logits,
                    const Eigen::MatrixXi &targets) {
    return this->forward(logits, targets);
  }

  /*
   Calculate the cross entropy loss given the logits and the target class
   labels.
 */
  double operator()(const Eigen::MatrixXd &logits,
                    const std::vector<int> &targets) {
    return this->forward(logits, targets);
  }
#pragma endregion Builtins
};
} // namespace loss