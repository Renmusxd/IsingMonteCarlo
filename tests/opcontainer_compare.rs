extern crate ising_monte_carlo;
extern crate rand;
use ising_monte_carlo::graph::Edge;
use ising_monte_carlo::sse::fast_ops::*;
use ising_monte_carlo::sse::qmc_graph::QMCGraph;
use ising_monte_carlo::sse::qmc_traits::*;
use ising_monte_carlo::sse::simple_ops::*;
use rand::prelude::*;

fn one_d_periodic(l: usize) -> Vec<(Edge, f64)> {
    (0..l).map(|i| ((i, (i + 1) % l), 1.0)).collect()
}

#[test]
fn single_cluster_test() {
    let l = 8;

    let rng: StdRng = SeedableRng::seed_from_u64(1234);
    let mut g = QMCGraph::<ThreadRng, SimpleOpNode, SimpleOpDiagonal, SimpleOpLooper>::new_with_rng(
        one_d_periodic(l),
        1.0,
        l,
        false,
        false,
        rng,
        Some(vec![true; l]),
    );
    let beta = 1.0;
    g.timesteps(1, beta);
    let state_a = g.into_vec();

    let rng: StdRng = SeedableRng::seed_from_u64(1234);
    let mut g = QMCGraph::<ThreadRng, FastOpNode, FastOps, FastOps>::new_with_rng(
        one_d_periodic(l),
        1.0,
        l,
        false,
        false,
        rng,
        Some(vec![true; l]),
    );
    let beta = 1.0;
    g.timesteps(1, beta);
    let state_b = g.into_vec();

    assert_eq!(state_a, state_b);
}
