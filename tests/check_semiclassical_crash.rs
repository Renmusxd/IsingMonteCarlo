use qmc::classical::graph::Edge;
use qmc::sse::*;
use rand::prelude::*;

fn two_d_periodic(l: usize) -> Vec<(Edge, f64)> {
    let indices: Vec<(usize, usize)> = (0usize..l)
        .map(|i| (0usize..l).map(|j| (i, j)).collect::<Vec<(usize, usize)>>())
        .flatten()
        .collect();
    let f = |i, j| j * l + i;

    let right_connects = indices
        .iter()
        .cloned()
        .map(|(i, j)| ((f(i, j), f((i + 1) % l, j)), -1.0));
    let down_connects = indices.iter().cloned().map(|(i, j)| {
        (
            (f(i, j), f(i, (j + 1) % l)),
            if i % 2 == 0 { 1.0 } else { -1.0 },
        )
    });
    right_connects.chain(down_connects).collect()
}

#[test]
fn run_two() {
    for i in 0..16 {
        let l = 2;
        let edges = two_d_periodic(l);
        let rng = SmallRng::seed_from_u64(i);
        let mut ising = DefaultQMCIsingGraph::<SmallRng>::new_with_rng(
            edges,
            0.1,
            l * l,
            rng,
            Some(vec![false; l * l]),
        );
        ising.set_run_semiclassical(true);
        ising.timesteps(1000, 1.0);
    }
}

#[test]
fn run_three() {
    for i in 0..16 {
        let l = 3;
        let edges = two_d_periodic(l);
        let rng = SmallRng::seed_from_u64(i);
        let mut ising = DefaultQMCIsingGraph::<SmallRng>::new_with_rng(
            edges,
            0.1,
            l * l,
            rng,
            Some(vec![false; l * l]),
        );
        ising.set_run_semiclassical(true);
        ising.timesteps(1000, 1.0);
    }
}

#[test]
fn run_four() {
    for i in 0..16 {
        let l = 4;
        let edges = two_d_periodic(l);
        let rng = SmallRng::seed_from_u64(i);
        let mut ising = DefaultQMCIsingGraph::<SmallRng>::new_with_rng(
            edges,
            0.1,
            l * l,
            rng,
            Some(vec![false; l * l]),
        );
        ising.set_run_semiclassical(true);
        ising.timesteps(1000, 1.0);
        println!("Average cluster: {}", ising.average_cluster_size());
    }
}
