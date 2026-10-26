#include <matrix/matrix.hpp>
#include <cmath>

Matrix f(const Matrix &m) {
  Matrix result = Matrix(m.get_nb_line(), m.get_nb_col());

  for (size_t i = 1; i <= m.get_nb_line(); ++i)
    for (size_t j = 1; j <= m.get_nb_col(); ++j)
      result(i, j) = 1 / (1 + exp(-1 * m(i, j)));

  return result;
}

int main() {
  Matrix W = Matrix(2, 2);
  Matrix X = Matrix(2, 1);
  Matrix B = Matrix(2, 1);

  W(1, 1) = 0.9;
  W(1, 2) = 0.79;
  W(2, 1) = 0.4;
  W(2, 2) = 0.26;

  X(1, 1) = 0;
  X(2, 1) = 1;

  B(1, 1) = 0.21;
  B(2, 1) = 0.4;

  f(W * X + B).display();

  return 0;
}
