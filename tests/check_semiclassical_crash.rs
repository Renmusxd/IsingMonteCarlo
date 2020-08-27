use qmc::classical::graph::Edge;
use qmc::sse::*;
use rand::prelude::*;

fn two_d_periodic(l: usize) -> Vec<(Edge, f64)> {
    let indices: Vec<(usize, usize)> = (0usize..l)
        .map(|i| (0usize..l).map(|j| (i, j)).collect::<Vec<(usize, usize)>>())
        .flatten()
        .collect();
    let f = |i, j| (j % l) * l + (i % l);

    let right_connects = indices
        .iter()
        .cloned()
        .map(|(i, j)| ((f(i, j), f(i + 1, j)), -1.0));
    let down_connects = indices
        .iter()
        .cloned()
        .map(|(i, j)| ((f(i, j), f(i, j + 1)), if i % 2 == 0 { 1.0 } else { -1.0 }));
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

fn two_d_periodic_faces(l: usize, edges: &[(Edge, f64)]) -> Vec<Vec<usize>> {
    let indices: Vec<(usize, usize)> = (0usize..l)
        .map(|i| (0usize..l).map(|j| (i, j)).collect::<Vec<(usize, usize)>>())
        .flatten()
        .collect();
    let f = |i, j| -> usize { (j % l) * l + (i % l) };

    let index_of_edge = |a, b| -> usize {
        edges
            .iter()
            .enumerate()
            .find(|(_, (edge, _))| (a, b) == *edge || (b, a) == *edge)
            .map(|(indx, _)| indx)
            .unwrap()
    };

    indices
        .into_iter()
        .map(|(x, y)| vec![f(x, y), f(x + 1, y), f(x + 1, y + 1), f(x, y + 1)])
        .map(|vars| {
            (0..vars.len())
                .map(|indx| index_of_edge(vars[indx], vars[(indx + 1) % vars.len()]))
                .collect()
        })
        .collect()
}

#[test]
fn run_dual_four() -> Result<(), String> {
    for i in 0..16 {
        let l = 4;
        let edges = two_d_periodic(l);
        let faces = two_d_periodic_faces(4, &edges);
        let rng = SmallRng::seed_from_u64(i);
        let mut ising = DefaultQMCIsingGraph::<SmallRng>::new_with_rng(
            edges,
            0.01,
            l * l,
            rng,
            Some(vec![false; l * l]),
        );
        ising.set_run_semiclassical(true);

        ising.enable_semiclassical_loops(faces)?;
        ising.timesteps(1000, 1.0);
        println!("Average cluster: {}", ising.average_cluster_size());
    }
    Ok(())
}

#[test]
fn run_dual_four_more_q() -> Result<(), String> {
    for i in 0..16 {
        let l = 4;
        let edges = two_d_periodic(l);
        let faces = two_d_periodic_faces(4, &edges);
        let rng = SmallRng::seed_from_u64(i);
        let mut ising = DefaultQMCIsingGraph::<SmallRng>::new_with_rng(
            edges,
            0.1,
            l * l,
            rng,
            Some(vec![false; l * l]),
        );
        ising.set_run_semiclassical(true);

        ising.enable_semiclassical_loops(faces)?;
        ising.timesteps(1000, 1.0);
        println!("Average cluster: {}", ising.average_cluster_size());
    }
    Ok(())
}
