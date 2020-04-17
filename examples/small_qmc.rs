extern crate monte_carlo;
use monte_carlo::graph::GraphState;
use monte_carlo::sse::qmc_graph::new_qmc;
use rayon::prelude::*;
use std::cmp::max;

fn run_transverse_quantum_monte_carlo(
    beta: f64,
    timesteps: usize,
    num_experiments: u64,
    edges: Vec<((usize, usize), f64)>,
    nvars: usize,
    transverse: f64,
) -> Vec<(Vec<bool>, f64)> {
    let biases = vec![0.0; nvars];
    (0..num_experiments)
        .map(|_| {
            let gs = GraphState::new(&edges, &biases);
            let cutoff = biases.len() * max(beta.round() as usize, 1);
            let mut qmc_graph = new_qmc(gs, transverse, cutoff, false, false);
            let e = qmc_graph.timesteps(timesteps as u64, beta);
            qmc_graph.debug_print();
            (qmc_graph.into_vec(), e)
        })
        .collect()
}

fn run_transverse_quantum_monte_carlo_and_measure_spins(
    beta: f64,
    timesteps: usize,
    num_experiments: u64,
    edges: Vec<((usize, usize), f64)>,
    nvars: usize,
    transverse: f64,
    spin_measurement: Option<(f64, f64)>,
    exponent: Option<i32>,
) -> Vec<(f64, f64)> {
    let biases = vec![0.0; nvars];
    let cutoff = biases.len();
    let exponent = exponent.unwrap_or(1);
    let (down_m, up_m) = spin_measurement.unwrap_or((-1.0, 1.0));
    (0..num_experiments)
        .into_par_iter()
        .map(|_| {
            let gs = GraphState::new(&edges, &biases);
            let mut qmc_graph = new_qmc(gs, transverse, cutoff, false, false);
            let ((measure, steps), average_energy) = qmc_graph.timesteps_measure(
                timesteps as u64,
                beta,
                (0.0, 0),
                |(acc, step), state, _| {
                    let acc = state
                        .iter()
                        .fold(0.0, |acc, b| if *b { acc + up_m } else { acc + down_m })
                        .powi(exponent)
                        + acc;
                    (acc, step + 1)
                },
                None,
            );
            (measure / steps as f64, average_energy)
        })
        .collect()
}

fn main() {
    let edges = vec![
        ((0, 1), -1.0),
        ((1, 2), 1.0),
        ((2, 3), 1.0),
        ((3, 0), 1.0),
        ((1, 7), 1.0),
        ((4, 5), -1.0),
        ((5, 6), 1.0),
        ((6, 7), 1.0),
        ((7, 4), 1.0),
    ];
    let biases = vec![0.0; 8];
    let transverse = 1.0;

    let n = 2048;

    let gs = GraphState::new(&edges, &biases);
    let mut qmc_graph = new_qmc(gs, transverse, biases.len(), false, false);
    let a = qmc_graph.calculate_bond_autocorrelation(n, 1.0, None, Some(false));
    let b = qmc_graph.calculate_bond_autocorrelation(n, 1.0, None, Some(true));

    a.into_iter().zip(b.into_iter()).for_each(|(a, b)| {
        println!("{}\t{}", a, b);
    })

    // let result = run_transverse_quantum_monte_carlo_and_measure_spins(
    //     40.0,
    //     100000,
    //     8,
    //     vec![
    //         ((0, 1), -1.0),
    //         ((1, 2), 1.0),
    //         ((2, 3), 1.0),
    //         ((3, 0), 1.0),
    //         ((1, 7), 1.0),
    //         ((4, 5), -1.0),
    //         ((5, 6), 1.0),
    //         ((6, 7), 1.0),
    //         ((7, 4), 1.0),
    //     ],
    //     8,
    //     1.0,
    //     None,
    //     Some(2),
    // );
    // println!("{:?}", result)
}
