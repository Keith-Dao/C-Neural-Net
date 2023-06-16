#pragma once
#include <Eigen/Dense>
#include <memory>
#include <nlohmann/json.hpp>
#include <nlohmann/json_fwd.hpp>

using json = nlohmann::json;

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
  CrossEntropyLoss(std::string reduction = "mean");

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
  /*
    Create a cross entropy loss instance from the JSON values.
  */
  static CrossEntropyLoss fromJson(const json &values);
#pragma endregion Load

#pragma region Save
  /*
      Get all relevant attributes in a serialisable format.

      Attributes includes:
          - reduction -- the reduction method used
    */
  json toJson();
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
  /*
    Perform the backward pass using the previous forward inputs.
  */
  Eigen::MatrixXd backward();
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

  bool operator==(const CrossEntropyLoss &other) const;
#pragma endregion Builtins
};
} // namespace loss