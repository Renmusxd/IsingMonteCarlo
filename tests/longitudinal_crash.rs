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

fn two_unit_cell() -> Vec<(Edge, f64)> {
    vec![
        ((0, 1), -1.0),
        ((1, 2), 1.0),
        ((2, 3), 1.0),
        ((3, 0), 1.0),
        ((1, 7), 1.0),
        ((4, 5), -1.0),
        ((5, 6), 1.0),
        ((6, 7), 1.0),
        ((7, 4), 1.0),
    ]
}

#[test]
fn run_simple() {
    for i in 0..16 {
        let edges = vec![((0, 1), 1.0)];
        let rng = SmallRng::seed_from_u64(i);
        let mut ising = DefaultQmcIsingGraph::<SmallRng>::new_with_rng(
            edges,
            1.0,
            1.0,
            2,
            rng,
            Some(vec![false; 2]),
        );
        ising.timesteps(1000, 1.0);
        assert!(ising.verify());
    }
}

#[test]
fn run_simple_rvb() {
    for i in 0..16 {
        let edges = vec![((0, 1), 1.0)];
        let rng = SmallRng::seed_from_u64(i);
        let mut ising = DefaultQmcIsingGraph::<SmallRng>::new_with_rng(
            edges,
            1.0,
            1.0,
            2,
            rng,
            Some(vec![false; 2]),
        );
        ising.set_run_rvb(true);
        ising.timesteps(1000, 1.0);
        assert!(ising.verify());
    }
}

#[test]
fn run_simple_heatbath() {
    for i in 0..16 {
        let edges = vec![((0, 1), 1.0)];
        let rng = SmallRng::seed_from_u64(i);
        let mut ising = DefaultQmcIsingGraph::<SmallRng>::new_with_rng(
            edges,
            1.0,
            1.0,
            2,
            rng,
            Some(vec![false; 2]),
        );
        ising.set_enable_heatbath(true);
        ising.timesteps(1000, 1.0);
        assert!(ising.verify());
    }
}

#[test]
fn run_three() {
    for i in 0..16 {
        let l = 3;
        let edges = two_d_periodic(l);
        let rng = SmallRng::seed_from_u64(i);
        let mut ising = DefaultQmcIsingGraph::<SmallRng>::new_with_rng(
            edges,
            0.1,
            0.1,
            l * l,
            rng,
            Some(vec![false; l * l]),
        );
        ising.set_run_rvb(true);
        ising.timesteps(1000, 1.0);
        assert!(ising.verify());
    }
}

#[test]
fn run_four() {
    for i in 0..16 {
        let l = 4;
        let edges = two_d_periodic(l);
        let rng = SmallRng::seed_from_u64(i);
        let mut ising = DefaultQmcIsingGraph::<SmallRng>::new_with_rng(
            edges,
            0.1,
            0.1,
            l * l,
            rng,
            Some(vec![false; l * l]),
        );
        ising.set_run_rvb(true);
        ising.timesteps(1000, 1.0);
        assert!(ising.verify());
    }
}

#[test]
fn run_two_unit_cell() {
    for i in 0..16 {
        let edges = two_unit_cell();
        let nvars = 8;
        let rng = SmallRng::seed_from_u64(i);
        let mut ising = DefaultQmcIsingGraph::<SmallRng>::new_with_rng(
            edges,
            1.0,
            0.1,
            nvars,
            rng,
            Some(vec![false; nvars]),
        );
        ising.timesteps(1000, 1.0);
        ising.set_run_rvb(true);
        ising.timesteps(1000, 1.0);
        assert!(ising.verify());
    }
}

#[test]
fn run_two_unit_cell_random_bonds() {
    for i in 0..16 {
        let mut rng = SmallRng::seed_from_u64(i);
        let edges = two_unit_cell()
            .into_iter()
            .map(|(vs, j)| (vs, j * rng.gen_range(0.5..2.0)))
            .collect();
        let nvars = 8;
        let mut ising = DefaultQmcIsingGraph::<SmallRng>::new_with_rng(
            edges,
            1.0,
            0.1,
            nvars,
            rng,
            Some(vec![false; nvars]),
        );
        ising.timesteps(1000, 1.0);
        ising.set_run_rvb(true);
        ising.timesteps(1000, 1.0);
        assert!(ising.verify());
    }
}
