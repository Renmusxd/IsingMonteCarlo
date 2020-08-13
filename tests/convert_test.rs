use qmc::classical::graph::Edge;
use qmc::sse::*;
use rand::prelude::*;

fn one_d_periodic(l: usize) -> Vec<(Edge, f64)> {
    (0..l).map(|i| ((i, (i + 1) % l), 1.0)).collect()
}

#[test]
fn convert_and_run() -> Result<(), ()> {
    let edges = one_d_periodic(3);
    let mut ising = DefaultQMCIsingGraph::<SmallRng>::new_with_rng(
        edges,
        1.0,
        3,
        SmallRng::seed_from_u64(1234),
        Some(vec![true, true, true]),
    );
    let mut qmc = ising.clone().into_qmc();

    for _ in 0..10 {
        println!("Ising: {:?}", ising.get_manager_ref());
        ising.timestep(1.0);
        println!("QMC: {:?}", qmc.get_diagonal_manager_ref());
        qmc.timestep(1.0);
        println!("===================");
    }
    assert_eq!(ising.state_ref(), qmc.state_ref());
    Ok(())
}
