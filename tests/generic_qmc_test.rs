extern crate ising_monte_carlo;
extern crate rand;
use ising_monte_carlo::graph::Edge;
use ising_monte_carlo::sse::qmc_traits::*;
use ising_monte_carlo::sse::*;
use rand::prelude::SmallRng;
use rand::SeedableRng;

fn one_d_periodic(l: usize) -> Vec<(Edge, f64)> {
    (0..l).map(|i| ((i, (i + 1) % l), 1.0)).collect()
}

fn make_simple_ising(l: usize, transverse: f64) -> DefaultQMCIsingGraph<SmallRng> {
    let rng = SmallRng::seed_from_u64(1234);
    DefaultQMCIsingGraph::<SmallRng>::new_with_rng(
        one_d_periodic(l),
        transverse,
        l,
        rng,
        Some(vec![true; l]),
    )
}

#[test]
fn convert_and_run() {
    let mut ising_graph = make_simple_ising(8, 1.0);
    let mut generic_graph: DefaultQMC<SmallRng> = ising_graph.clone().into();
    ising_graph.timestep(1.0);
    generic_graph.timestep(1.0);

    assert_eq!(ising_graph.into_vec(), generic_graph.into_vec());
}
