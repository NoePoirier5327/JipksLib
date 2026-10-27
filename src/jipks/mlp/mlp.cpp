#include "mlp.hpp"
#include <stdexcept>

namespace jipks::mlp {
  MLP::MLP(const std::vector<size_t> &shape) {
    if (shape.size() < 2)
      throw std::invalid_argument("There should be at least an input layer and an output layer.");

    this->_layers.reserve(shape.size() - 1);
    for (size_t i = 1; i < shape.size(); ++i) {
      if (shape[i-1] == 0)
        throw std::invalid_argument("A layer should at least have 1 neuron.");
      this->_layers.emplace_back(shape[i - 1], shape[i]);
    }
  }

  std::vector<double> MLP::feed_forward(const std::vector<double> &inputs) const {
    if (this->_layers.empty() || inputs.size() != this->_layers.front().get_nb_inputs())
      throw std::invalid_argument("MLP input format incorrect.");

    // On construit la matrice des entrées.
    Matrix input_layer = Matrix(inputs.size(), 1);
    for (size_t i = 1; i <= inputs.size(); ++i) {
      if (inputs[i-1] > 1 || inputs[i-1] < 0)
        throw std::invalid_argument("MLP input format incorrect.");
      input_layer(i, 1) = inputs[i-1];
    }

    // On propage l'entrée pour obtenir la prédiction du réseau de neurones.
    Matrix output = input_layer;
    for (const Layer &layer: this->_layers) {
      output = layer.feed_forward(output);
    }

    // On construit le vecteur de sortie et le renvoie.
    std::vector<double> outputs;
    outputs.reserve(output.get_nb_line());
    for (size_t i = 1; i <= output.get_nb_line(); ++i)
      outputs.emplace_back(output(i, 1));

    return outputs;
  }
}
