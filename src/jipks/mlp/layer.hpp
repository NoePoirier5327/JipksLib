#ifndef LAYER_HEADER
#define LAYER_HEADER

#include <matrix/matrix.hpp>
#include <cstdlib>

namespace jipks::mlp {
  /**
   * @brief Générateur de nombre aléatoires de la couche.
   *
   * @param min, borne inférieur incluse du nombre aléatoire à générer.
   * @param max, borne supérieur incluse du nombre aléatoire à générer.
   *
   * @return nombre aléatoire compris entre min et max.
   */
  double rng(double min, double max);

  /**
   * @class Layer
   * @brief Couche quelconque d'un mlp.
   */
  class Layer {
    public:
      /**
       * @brief Couche quelconque d'un mlp.
       *
       * Initialise les poits et biais à des valeurs aléatoires.
       *
       * @param nb_inputs, nombre d'entrées de la couche courante.
       * @param nb_outputs, nombre de sorties de la couche courante.
       *
       * @throw std::invalid_argument si nb_inputs < 1
       * @throw std;;invalid_argument si nb_outputs < 1
       */
      Layer(size_t nb_inputs, size_t nb_outputs);

      /**
       * @brief Méthode de prédiction de la couche courante.
       *
       * Applique la formule de prédiction d'une couche Y = f(W * X + B)
       * avec Y le vecteur de sorties, X celui des entrées, B celui des biais,
       * W la matrice des poids entre la couche d'entrées et celle courante et f
       * la fonction de normalisation des sorties.
       *
       * @param inputs, entrées X de la couche servant à la prédiction.
       * @param weights, poids entre la couche d'entrées et celle courante.
       *
       * @return prédiction de la couche courante.
       *
       * @throw std::invalid_argument si weights.get_nb_line() != _biases.get_nb_line()
       * @throw std::invalid_argument si weights.get_nb_col() != inputs.get_nb_line()
       * @throw std::invalid_argument si inputs.get_nb_col() != 1
       */
      Matrix feed_forward(const Matrix &inputs, const Matrix &weights) const;

      /**
       * @brief Accesseur des poids de la couche courante.
       */
      Matrix get_weights() const;

      /**
       * @brief Accesseur du nombre d'entrées de la couche.
       */
      size_t get_nb_inputs() const;

      /**
       * @brief Accesseur du nombre de sorties de la couche.
       */
      size_t get_nb_outputs() const;

    private:
      Matrix _weights;
      Matrix _biases;
  };
}

#endif // !LAYER_HEADER
