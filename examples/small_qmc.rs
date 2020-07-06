extern crate ising_monte_carlo;
use ising_monte_carlo::graph::GraphState;
use ising_monte_carlo::sse::fast_ops::*;
use ising_monte_carlo::sse::qmc_graph::{new_qmc, QMCGraph};
use rand::prelude::ThreadRng;
use rayon::prelude::*;
use std::cmp::max;

fn main() {
    let edges = vec![
        ((0, 1), -1.0),
        ((1, 2), 1.0),
        ((2, 3), 1.0),
        ((3, 0), 1.0),
        // ((1, 7), 1.0),
        // ((4, 5), -1.0),
        // ((5, 6), 1.0),
        // ((6, 7), 1.0),
        // ((7, 4), 1.0),
    ];
    let transverse = 1.0;

    let rng = rand::thread_rng();
    let mut g = QMCGraph::<ThreadRng, FastOpNode, FastOps, FastOps>::new_with_rng(
        edges, transverse, 3, false, false, rng, None,
    );
    let a = g.timesteps(1000, 1.0);

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
