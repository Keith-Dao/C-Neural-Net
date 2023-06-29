#pragma once
#include <exception>
#include <string>
#include <utility>
namespace Eigen {
template <typename Derived> class MatrixBase;
}

namespace exceptions::eigen {
class InvalidShapeException : public std::exception {
  std::pair<int, int> expected, got;
  virtual const char *what() const throw();

public:
  template <typename X, typename Y>
  InvalidShapeException(const Eigen::MatrixBase<X> &expected,
                        const Eigen::MatrixBase<Y> &got)
      : got(got.rows(), got.cols()),
        expected(expected.rows(), expected.cols()){};
};

class EmptyMatrixException : public std::exception {
  std::string variable;
  virtual const char *what() const throw();

public:
  EmptyMatrixException(std::string variable) : variable(variable){};
};
} // namespace exceptions::eigen