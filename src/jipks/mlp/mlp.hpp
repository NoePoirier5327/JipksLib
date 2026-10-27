#ifndef MLP_HEADER
#define MLP_HEADER

#include <matrix/matrix.hpp>
#include "layer.hpp"
#include <vector>

namespace jipks::mlp {
  /**
   * @class MLP
   * @brief Réseau de neurones artificiels multicouches.
   */
  class MLP {
    public:
      /**
       * @brief Constructeur du réseau multicouches.
       *
       * @param shape, contient en première position le nombre de neurones de la couche d'entrées, en dernière position le nombre de neurones de la couche de sortie et entre deux le nombre de neurones par couches de traitement.
       *
       * @throw std::invalid_argument shape.size() < 2
       */
      MLP(const std::vector<size_t> &shape);

      /**
       * @brief Méthode de prédiction du réseau de neurones.
       *
       * @param inputs, vecteur d'entrées du réseau de neurones.
       * @return vecteur des sorties du réseau de neurones.
       *
       * @throw std::invalid_argument si inputs.size() != nb_inputs.
       * @throw std::invalid_argument si inputs[i] > 1 ou inputs[i] < 0.
       */
      std::vector<double> feed_forward(const std::vector<double> &inputs) const;

    public:
      std::vector<Layer> _hidden_layers;
      Matrix _inputs_weight;
  };
}

#endif // !MLP_HEADER
