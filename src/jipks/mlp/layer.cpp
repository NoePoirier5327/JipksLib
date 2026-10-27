#include "layer.hpp"
#include <random>
#include <cmath>
#include <stdexcept>

namespace jipks::mlp {
  double rng(double min, double max) {
    // Création d'un moteur aléatoire static (il n'est appelée qu'une seule fois durant la compilation)
    static std::random_device rd;
    static std::mt19937 gen(rd());

    // création d'une distribution uniforme réelle
    std::uniform_real_distribution<double> dis(min, max);

    // On renvoie le nombre généré aléatoirement
    return dis(gen);
  }

  // Fonction d'activation du réseau de neurones.
  double f(double x) {
    return 1 / (1 + exp(-1 * x));
  }

  // Fonction de propagation de l'activation d'un réseau de neurones.
  Matrix f_matrix(const Matrix &m) {
    Matrix result = Matrix(m.get_nb_line(), m.get_nb_col());

    for (size_t i = 1; i <= m.get_nb_line(); ++i)
      for (size_t j = 1; j <= m.get_nb_col(); ++j)
        result(i, j) = f(m(i, j));

    return result;
  }

  Layer::Layer(
    size_t nb_inputs,
    size_t nb_outputs
  ) : _weights(nb_outputs, nb_inputs), _biases(nb_outputs, 1) {
    // On initialise les poids et biais à des valeurs aléatoires entre 0 et 1.
    for (size_t i = 1; i <= this->_weights.get_nb_line(); ++i)
      for (size_t j = 1; j <= this->_weights.get_nb_col(); ++j)
        this->_weights(i, j) = rng(0, 1);
    
    for (size_t i = 1; i <= this->_biases.get_nb_line(); ++i)
      this->_biases(i, 1) = rng(0, 1);
  }

  Matrix Layer::feed_forward(const Matrix &inputs) const {
    if (this->_weights.get_nb_line() != this->_biases.get_nb_line())
      throw std::invalid_argument("Layer input format incorrect, weights.line != _biases.line.");

    if (this->_weights.get_nb_col() != inputs.get_nb_line())
      throw std::invalid_argument("Layer input format incorrect, weights.col != inputs.line.");

    if (inputs.get_nb_col() != 1)
      throw std::invalid_argument("Layer input format incorrect, inputs.col != 1.");

    return f_matrix(this->_weights * inputs + this->_biases);
  }

  Matrix Layer::get_weights() const {
    return this->_weights;
  }

  size_t Layer::get_nb_inputs() const {
    return this->_weights.get_nb_col();
  }

  size_t Layer::get_nb_outputs() const {
    return this->_weights.get_nb_line();
  }
}
