#ifndef MATRIX_HEADER
#define MATRIX_HEADER

#include <cstddef>

/**
 * @class Matrix
 * @brief Représente une matrice composée de nombres en double précision.
 */
class Matrix {
  public:
    /**
     * @brief Construit une nouvelle matrice de largeur width et hauteur height.
     *
     * Elle l'initialise avec des 0.
     *
     * @param nb_line, nombre de ligne dans la matrice à créer.
     * @param nb_col, nombre de colonnes dans la matrice à créer.
     */
    Matrix(size_t nb_line, size_t nb_col);

    /**
     * @brief Constructeur de copie
     *
     * Copie le contenu de la matrice en paramètre pour créer le contenu
     * de la nouvelle instance.
     *
     * @param m Copie cette matrice pour créer une nouvelle instance.
     */
    Matrix(const Matrix& m);

    /**
     * @brief Constructeur de déplacement.
     *
     * Transfère le contenu de la matrice en paramètre dans la nouvelle instance.
     *
     * @param m Matrice dont le contenu est à transférer.
     */
    Matrix(Matrix&& m) noexcept;

    /**
     * @brief Détruit la matrice courante.
     */
    ~Matrix();

    /**
     * @brief Accesseur de la largeur de la matrice courante.
     *
     * @return Copie de la largeur de la matrice courante.
     */
    size_t get_nb_col() const;

    /**
     * @brief Accesseur de la hauteur de la matrice courante.
     *
     * @return Copie de la hauteur de la matrice courante.
     */
    size_t get_nb_line() const;

    /**
     * @brief Accesseur/Mutateur de la matrice courante.
     * 
     * @param line, ligne de la matrice à récupérer.
     * @param col, colonne de la matrice à récupérer.
     *
     * @return référence vers la valeur qu'on a voulu récupérer.
     *
     * @throw std::out_of_range si line > _nb_line ou line < 1 ou col > _nb_col ou col < 1.
     */
    double& operator()(size_t line, size_t col);

    /**
     * @brief Accesseur de la matrice courante.
     *
     * @param line, ligne de la matrice à récupérer.
     * @param col, colonne de la matrice à récupérer.
     *
     * @return copie de la valeur à la position en paramètre.
     *
     * @throw std::out_of_range si line > _nb_line ou line < 1 ou col > _nb_col ou col < 1.
     */
    double operator()(size_t line, size_t col) const;

    /**
     * @brief Comparateur de deux matrices.
     *
     * @param m Seconde matrice à comparer.
     *
     * @return true si elle sont identique, false sinon
     *
     * @note si les deux matrices ont des dimensions différentes, alors, la comparaison renvoie faux.
     */
    bool operator==(const Matrix& m) const;

    /**
     * @brief Calcul le produit matriciel entre celle courante et celle en paramètre.
     *
     * Surcharge l'opérateur `*` pour pouvoir calculer `matrice = A * B`
     *
     * @param m Matrice avec laquelle réaliser le produit matricel.
     *
     * @return Matrice résultante du produit.
     *
     * @throw std::invalid_argument si les dimensions ne permettent pas le produit matriciel.
     */
    Matrix operator*(const Matrix& m) const;

    /**
     * @brief Surcharge *= pour le produit matriciel.
     *
     * Effectue le produit matriciel entre la matrice courante
     * et celle en paramètre.
     * Stocke le résultat du produit dans la matrice courante.
     *
     * @param m, matrice avec laquelle effectuer le produit matriciel.
     *
     * @throw std::invalid_argument si les dimensions ne permettent pas le produit matriciel.
     */
    void operator*=(const Matrix &m);

    /**
     * @brief Surcharge *= pour le produit entre une matrice et un scalaire.
     *
     * Effectue le produit entre la matrice courante et le 
     * scalaire en paramètre.
     * Stocke le résultat du produit dans la matrice courante.
     *
     * @param a, scalaire par lequel multiplier la matrice courante.
     */
    void operator*=(double a);

    /**
     * @brief Calcul la somme de deux matrices.
     *
     * Surcharge l'opérateur `+` pour pouvoir calculer `result = A + B`.
     * Le calcul se fait entre la matrice courante et celle en paramètre,
     * aucune n'est modifiée mais le résultat est renvoyée sous forme d'une nouvelle matrice.
     *
     * @param m Matrice avec laquelle réaliser la somme matriciel.
     *
     * @return Nouvelle matrice contenant le résultat de la somme.
     *
     * @throw std::invalid_argument si les dimensions ne permettent pas le calcul.
     */
    Matrix operator+(const Matrix& m) const;

    /**
     * @brief Surcharge l'opérateur +=.
     *
     * Fais la somme entre la matrice en paramètre et celle
     * courante et stocke le résultat dans la matrice courante.
     *
     * @param m, matrice à ajouter à celle courante.
     *
     * @throw std::invalid_argument si les dimensions ne permettent pas le calcul.
     */
    void operator+=(const Matrix &m);

    /**
     * @brief Copie la matrice en paramètre dans la matrice courante.
     *
     * Surcharge l'opérateur `=`, la méthode désalloue la matrice courante
     * et copie le contenu de la matrice en paramètre dans celle courante.
     *
     * @param m Matrice à copier.
     *
     * @return Référence vers l'instance de la matrice à attribuer.
     */
    Matrix& operator=(const Matrix& m);

    /**
     * @brief Transfère la matrice en paramètre dans la matrice courante.
     *
     * Surcharge l'opérateur `=`, la méthode transfère le contenu de la matrice
     * en paramètre dans la matrice en paramètre.
     *
     * @param m Matrice à transférer.
     *
     * @return Référence vers l'instance de la matrice à attribuer.
     */
    Matrix& operator=(Matrix&& m) noexcept;

    /**
     * @brief Affiche la matrice courante.
     */
    void display() const;

  private:
    /**
     * @brief Transforme la coordonnée matricielle en deux dimensions en une dimmension.
     *
     * @param line, index de la ligne à laquelle accéder.
     * @param col, index de la colonnne à laquelle accéder.
     *
     * @return index dans _content correspondant à la coordonnée en paramètres.
     *
     * @throw std::invalid_argument si line > _nb_line ou line < 1
     * @throw std::invalid_argument si col > _nb_col ou col < 1
     */
    size_t to_linear_coordinate(size_t line, size_t col) const;

    double *_content;
    size_t _nb_line;
    size_t _nb_col;
};

/**
 * @brief Créer une matrice identitée de taille n.
 *
 * @param n taille de la matrice à créer.
 *
 * @return Instance de la matrice identitée.
 */
Matrix new_identity_matrix(size_t n);

/**
 * @brief Multiplie une matrice par un scalaire.
 *
 * @param m, matrice à multiplier par le scalaire.
 * @param a, scalaire à multiplier avec la matrice.
 *
 * @return matrice résultante de la multiplication.
 */
Matrix operator*(const Matrix &m, double a);

/**
 * @brief Multiplie une matrice par un scalaire.
 *
 * @param a, scalaire à multiplier avec la matrice.
 * @param m, matrice à multiplier par le scalaire.
 *
 * @return matrice résultante de la multiplication.
 */
Matrix operator*(double a, const Matrix &m);

#endif // !MATRIX_HEADER
