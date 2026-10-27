#include <matrix/matrix.hpp>
#include "jipks/jipks.hpp"
#include <iostream>

int main() {
  jipks::mlp::MLP mlp = jipks::mlp::MLP({2, 3, 3, 2, 1});

  std::vector<double> outputs = mlp.feed_forward({1, 0});
  for (size_t i = 0; i < outputs.size(); ++i)
    std::cout << outputs[i] << " ";
  std::cout << std::endl;

  return 0;
}
