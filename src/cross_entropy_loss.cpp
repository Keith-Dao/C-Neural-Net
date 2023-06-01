#include "cross_entropy_loss.hpp"
#include "exceptions.hpp"

using namespace loss;

#pragma region Reductions
std::unordered_map<std::string, std::function<double(Eigen::MatrixXd)>>
    CrossEntropyLoss::reductions{
        {"mean", [](Eigen::MatrixXd matrix) { return matrix.mean(); }},
        {"sum", [](Eigen::MatrixXd matrix) { return matrix.sum(); }}};
#pragma endregion Reductions

#pragma region Properties
#pragma region Reduction
std::string CrossEntropyLoss::getReduction() const { return this->reduction; }

void CrossEntropyLoss::setReduction(std::string reduction) {
  if (!this->reductions.count(reduction)) {
    throw src_exceptions::InvalidReductionException();
  }
  this->reduction = reduction;
}
#pragma region Reduction
#pragma endregion Properties