use jipks::mlp::{MLP, ActivationFunction};

fn main() {
    let mut network = MLP::new(&[2, 4, 1], ActivationFunction::Sigmoid);

    let training_inputs = [[0f64, 0f64], [0f64, 1f64], [1f64, 0f64], [1f64, 1f64]];
    let training_targets = [[0f64], [1f64], [1f64], [0f64]];

    let learning_rate = 0.5f64;
    let epochs = 1_000_000;

    println!("Training on XOR logic.");
    for _ in 0..epochs {
        for i in 0..training_inputs.len() {
            network.train(&training_inputs[i], &training_targets[i], learning_rate);
        }
    }

    let output = network.predict(&[1f64, 0f64]);
    for val in output {
        print!("{} ", val);
    }
    println!();
}
