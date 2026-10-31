use pokin::Matrix;

/// Fonction d'activation à appliquer à un mlp.
#[derive(Clone, Copy)]
pub enum ActivationFunction {
    Sigmoid,
    ReLU
}

/// Réseau de neurones artificiel multicouches (mlp).
pub struct MLP {
    layers: Vec<Layer>
}

impl MLP {
    /// Construit un nouveau réseau de neurones avec un certain nombre de couches, un certain nombre
    /// de neurones par couches et une fonction d'activation unique.
    ///
    /// # Arguments
    /// * `shape` contient le nombre de neurones par couches à instancier.
    /// * `activ_func` fonction d'activation à appliquer à tous le réseau qu'on créer.
    ///
    /// # Return
    /// Instance vers le nouveau réseau de neurones.
    pub fn new(shape: &[usize], activ_func: ActivationFunction) -> Self {
        assert!(shape.len() >= 2, "There should be at least an input layer and an output layer.");

        let mut layers = Vec::new();
        for i in 1..shape.len() {
            layers.push(Layer::new(shape[i-1], shape[i], activ_func));
        }

        Self {
            layers
        }
    }

    /// Fonction de prédiction du mlp courant.
    /// Propage l'entrée en paramètre dans tous le réseau de neurones et renvoie la sortie du
    /// réseau.
    ///
    /// # Argument
    /// * `input` entrée à propager dans le réseau.
    ///
    /// # Return
    /// Sortie du réseau correspondant à l'entrée en paramètre.
    pub fn predict(&mut self, input: &[f64]) -> Vec<f64> {
        assert!(!input.is_empty() && input.len() == self.layers[0].get_nb_input(), "MLP input format incorrect.");

        let mut input_layer = Matrix::new(input.len(), 1);
        for i in 1..=input.len() {
            let val = input[i-1];
            assert!((0f64..=1f64).contains(&val), "MLP input format incorrect.");
            input_layer.set(i, 1, val);
        }

        let mut output_layer = input_layer;
        for layer in &mut self.layers {
            output_layer = layer.feed_forward(&output_layer);
        }

        let mut output = Vec::new();
        for i in 1..=output_layer.get_nb_line() {
            output.push(output_layer.get(i, 1));
        }

        output
    }

    /// Corrige le réseau courant en fonction d'une entrée et de la sortie qu'on attends qu'il nous
    /// donne.
    ///
    /// # Arguments
    /// * `input` entrée soumise au réseau.
    /// * `target` sortie qu'on attends de lui.
    /// * `learning_rate` taux d'apprentissage à appliquer au réseau.
    ///
    /// # Return
    /// Erreur entre la prédiction et la sortie attendue.
    pub fn train(&mut self, input: &[f64], target: &[f64], learning_rate: f64) -> f64 {
        assert!((0f64..=1f64).contains(&learning_rate), "The learning rate should be within 0 and 1.");

        let prediction = self.predict(input);
        let loss = self.compute_mse(&prediction, target);

        let mut output_error = Matrix::new(prediction.len(), 1);
        for i in 0..prediction.len() {
            output_error.set(i+1, 1, prediction[i] - target[i]);
        }

        let mut error = output_error;
        for i in (0..self.layers.len()).rev() {
            error = self.layers[i].backward_propagate(&error, learning_rate);
        }

        loss
    }

    /// Algorithme des moindres carrées appliqué au calcul des erreurs entre les prédictions du
    /// réseau et les sorties cibles.
    fn compute_mse(&self, prediction: &[f64], target: &[f64]) -> f64 {
        assert!(prediction.len() == target.len(), "Incoherent sizes to calculate error.");

        let mut mse = 0f64;
        for i in 0..prediction.len() {
            let error = prediction[i] - target[i];
            mse += error * error;
        }

        mse / prediction.len() as f64
    }
}

/// Représente une couche d'un réseau de neurones multicouches (mlp).
struct Layer {
    weights: Matrix,
    biases: Matrix,
    last_input: Matrix,
    last_z: Matrix,
    delta_gradient: Matrix,
    weights_gradient: Matrix,
    f_prime_z: Matrix,
    activ_func: ActivationFunction
}

impl Layer {
    /// Construit une nouvelle couche de réseau de neurones.
    ///
    /// # Arguments
    /// * `nb_inputs` nombre d'entrées appliquées à la nouvelle couche.
    /// * `nb_outputs` nombre de sorties de la nouvelle couche.
    /// * `activ_func` fonction d'activation appliquée à la nouvelle couche.
    ///
    /// # Return
    /// Instance de la nouvelle couche.
    pub fn new(nb_inputs: usize, nb_outputs: usize, activ_func: ActivationFunction) -> Self {
        assert_ne!(nb_inputs, 0, "You can't create a non connected layer.");
        assert_ne!(nb_outputs, 0, "You can't create an empty layer.");

        // On initialise les poids à des valeurs aléatoires, initialisation Xavier.
        let limit = (6f64 / (nb_inputs + nb_outputs) as f64).sqrt();
        let mut weights = Matrix::new(nb_outputs, nb_inputs);
        for i in 1..=weights.get_nb_line() {
            for j in 1..=weights.get_nb_col() {
                weights.set(i, j, rand::random_range(-limit..limit));
            }
        }

        // Les biais sont initialisés à 0 par défaut.
        Self {
            weights,
            biases: Matrix::new(nb_outputs, 1),
            last_input: Matrix::new(nb_inputs, 1),
            last_z: Matrix::new(nb_outputs, 1),
            delta_gradient: Matrix::new_square(1),
            weights_gradient: Matrix::new_square(1),
            f_prime_z: Matrix::new_square(1),
            activ_func
        }
    }

    /// Calcul les prédictions de la couche courante à partir de celles de la couche précédente.
    /// Renvoie les prédictions de cette couche pour les appliquers à la couche suivante.
    ///
    /// # Argument
    /// * `input` entrées à appliquer à la couche courante.
    ///
    /// # Return
    /// Matrice contenant les prédictions de la couche courante.
    pub fn feed_forward(&mut self, input: &Matrix) -> Matrix {
        assert!(self.weights.get_nb_line() == self.biases.get_nb_line(), "Layer input format incorrect, weights.nb_line != biases.nb_line");
        assert!(self.weights.get_nb_col() == input.get_nb_line(), "Layer input format incorrect, weights.nb_col != input.nb_line");
        assert!(input.get_nb_col() == 1, "Layer input format incorrect, input.nb_col != 1");

        // On clone la dernière entrée.
        self.last_input = input.clone();

        // On calcul la sortie sans clone.
        self.last_z = Matrix::new(self.weights.get_nb_line(), 1);
        for i in 1..=self.last_z.get_nb_line() {
            let mut sum = 0_f64;
            for k in 1..=self.weights.get_nb_col() {
                sum += self.weights.get(i, k) * input.get(k, 1);
            }
            self.last_z.set(i, 1, sum + self.biases.get(i, 1));
        }

        // On active et renvoie.
        f_matrix(&self.last_z, self.activ_func)
    }

    /// Met à jour les poids et biais de la couche courante grâce à la descente de gradient
    /// en utilisant le taux d'apprentissage et la matrice des erreurs de la couche suivante.
    ///
    /// # Arguments
    /// * `output_error` matrice des erreurs de la couche suivante par rapport à cette couche.
    /// * `learning_rate` taux d'apprentissage du réseau de neurones courant.
    ///
    /// # Return
    /// Matrice des erreurs de cette couche à appliquer à la couche précédente.
    pub fn backward_propagate(&mut self, output_error: &Matrix, learning_rate: f64) -> Matrix {
        self.f_prime_z = f_prime_matrix(&self.last_z, self.activ_func);

        // On calcul la matrice de corrections des erreurs.
        self.delta_gradient = Matrix::elementwise_multiplication(output_error, &self.f_prime_z);

        // On calcul le gradient des poids.
        //let weights_gradient = delta.clone() * self.last_input.tranpose();
        self.weights_gradient = Matrix::multiplication_with_transposed(&self.delta_gradient, &self.last_input);

        // On met à jour les poids grâce au gradient
        for i in 1..=self.weights.get_nb_line() {
            for j in 1..=self.weights.get_nb_col() {
                self.weights.set(i, j, self.weights.get(i, j) - (learning_rate * self.weights_gradient.get(i, j)));
            }
        }

        // On met à jour les biais grâce au gradient.
        for i in 1..=self.biases.get_nb_line() {
            self.biases.set(i, 1, self.biases.get(i, 1) - (learning_rate * self.delta_gradient.get(i, 1)));
        }

        // On renvoie la matrice des erreurs de sortie.
        Matrix::transposed_multiplication(&self.weights, &self.delta_gradient)
    }

    /// Renvoie le nombre d'entrées de la couche courante.
    pub fn get_nb_input(&self) -> usize {
        self.weights.get_nb_col()
    }
}

/// Fonction d'activation du mlp.
/// Applique la fonction d'activation désignée par activ_func pour
/// l'appliquer à l'entrée x.
/// 
/// # Arguments
/// * `x` entrée de la fonction d'activation.
/// * `activ_func` fonction d'activation utilisée.
///
/// # Return
/// Entrée x activée par f.
fn f(x: f64, activ_func: ActivationFunction) -> f64 {
    match activ_func {
        ActivationFunction::Sigmoid => 1f64 / (1f64 + (-x).exp()),
        ActivationFunction::ReLU => if x > 0f64 { x } else { 0f64 }
    }
}

/// Dérivée de la fonction d'activation du mlp.
/// Applique la dérivée de la fonction d'activation désignée par
/// activ_func pour l'appliquer à x.
///
/// # Arguments
/// * `x` entrée de la dérivée de l'activation.
/// * `activ_func` fonction d'activation utilisée.
///
/// # Return
/// Résultat de f_prime(x).
fn f_prime(x: f64, activ_func: ActivationFunction) -> f64 {
    match activ_func {
        ActivationFunction::Sigmoid => {
            let fx = f(x, activ_func);
            fx * (1f64 - fx)
        },

        ActivationFunction::ReLU => if x > 0f64 { 1f64 } else { 0f64 }
    }
}

/// Applique la fonction d'activation f à une matrice de f64.
///
/// # Arguments
/// * `m` matrice des entrées à activer.
/// * `activ_func` fonction d'activation cible.
///
/// # Return
/// Matrice des entrées activée par f.
fn f_matrix(m: &Matrix, activ_func: ActivationFunction) -> Matrix {
    let mut result = Matrix::new(m.get_nb_line(), m.get_nb_col());

    for i in 1..=result.get_nb_line() {
        for j in 1..=result.get_nb_col() {
            result.set(i, j, f(m.get(i, j), activ_func));
        }
    }

    result
}

/// Applique f_prime à la matrice en paramètre.
///
/// # Arguments
/// * `m` matrice des entrées à passer dans f_prime.
/// * `activ_func` dérivée de la fonction d'activation cible.
///
/// # Return
/// Matrice des entrées passées dans f_prime.
fn f_prime_matrix(m: &Matrix, activ_func: ActivationFunction) -> Matrix {
    let mut result = Matrix::new(m.get_nb_line(), m.get_nb_col());

    for i in 1..=result.get_nb_line() {
        for j in 1..=result.get_nb_col() {
            result.set(i, j, f_prime(m.get(i, j), activ_func));
        }
    }

    result
}
