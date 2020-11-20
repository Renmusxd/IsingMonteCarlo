use qmc::classical::graph::Edge;
use qmc::sse::fast_ops::*;
use qmc::sse::*;
use rand::prelude::*;
use smallvec::smallvec;

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

struct EN {
    bonds_for_var: Vec<Vec<usize>>,
    bonds: Vec<((usize, usize), bool)>,
}

impl EdgeNavigator for EN {
    fn n_bonds(&self) -> usize {
        self.bonds.len()
    }

    fn bonds_for_var(&self, var: usize) -> &[usize] {
        &self.bonds_for_var[var]
    }

    fn vars_for_bond(&self, bond: usize) -> (usize, usize) {
        self.bonds[bond].0
    }

    fn bond_prefers_aligned(&self, bond: usize) -> bool {
        self.bonds[bond].1
    }
}

#[test]
fn run_single_var() {
    let mut manager = FastOps::new_from_ops(
        1,
        vec![
            (
                0,
                FastOp::offdiagonal(smallvec![0], 0, smallvec![false], smallvec![false], true),
            ),
            (
                1,
                FastOp::offdiagonal(smallvec![0], 0, smallvec![false], smallvec![false], true),
            ),
        ]
        .into_iter(),
    );
    let edges = EN {
        bonds_for_var: vec![vec![]],
        bonds: vec![],
    };
    let mut state = vec![false];
    let mut rng = SmallRng::seed_from_u64(0);
    (0..100).for_each(|_| {
        manager.rvb_update(&edges, &mut state, 1, &mut rng);
    });
    println!("{:?}", state);
    assert!(manager.verify(&state));
}

#[test]
fn run_two_independent_vars() {
    let mut manager = FastOps::new_from_ops(
        2,
        vec![
            (
                0,
                FastOp::offdiagonal(smallvec![0], 0, smallvec![false], smallvec![false], true),
            ),
            (
                1,
                FastOp::offdiagonal(smallvec![1], 1, smallvec![false], smallvec![false], true),
            ),
            (
                2,
                FastOp::offdiagonal(smallvec![0], 0, smallvec![false], smallvec![false], true),
            ),
            (
                3,
                FastOp::offdiagonal(smallvec![1], 1, smallvec![false], smallvec![false], true),
            ),
        ]
        .into_iter(),
    );
    let edges = EN {
        bonds_for_var: vec![vec![], vec![]],
        bonds: vec![],
    };
    let mut state = vec![false, false];
    let mut rng = SmallRng::seed_from_u64(0);
    (0..100).for_each(|_| {
        manager.rvb_update(&edges, &mut state, 1, &mut rng);
    });
    println!("{:?}", state);
    assert!(manager.verify(&state));
}

#[test]
fn run_two_joined_vars() {
    let mut manager = FastOps::new_from_ops(
        2,
        vec![
            (
                0,
                FastOp::offdiagonal(smallvec![0], 2, smallvec![false], smallvec![false], true),
            ),
            (
                1,
                FastOp::offdiagonal(smallvec![1], 3, smallvec![false], smallvec![false], true),
            ),
            (
                2,
                FastOp::offdiagonal(smallvec![0], 2, smallvec![false], smallvec![false], true),
            ),
            (
                3,
                FastOp::offdiagonal(smallvec![1], 3, smallvec![false], smallvec![false], true),
            ),
            (
                4,
                FastOp::diagonal(smallvec![0, 1], 0, smallvec![false, false], false),
            ),
        ]
        .into_iter(),
    );
    let edges = EN {
        bonds_for_var: vec![vec![0, 1], vec![0, 1]],
        bonds: vec![((0, 1), true), ((0, 1), false)],
    };
    let mut state = vec![false, false];
    let mut rng = SmallRng::seed_from_u64(0);
    (0..100).for_each(|_| {
        manager.rvb_update(&edges, &mut state, 1, &mut rng);
    });
    println!("{:?}", state);
    assert!(manager.verify(&state));
}

#[test]
fn run_two_joined_vars_double() {
    let mut manager = FastOps::new_from_ops(
        2,
        vec![
            (
                0,
                FastOp::offdiagonal(smallvec![0], 2, smallvec![false], smallvec![false], true),
            ),
            (
                1,
                FastOp::offdiagonal(smallvec![1], 3, smallvec![false], smallvec![false], true),
            ),
            (
                2,
                FastOp::offdiagonal(smallvec![0], 2, smallvec![false], smallvec![false], true),
            ),
            (
                3,
                FastOp::offdiagonal(smallvec![1], 3, smallvec![false], smallvec![false], true),
            ),
            (
                4,
                FastOp::diagonal(smallvec![0, 1], 0, smallvec![false, false], false),
            ),
            (
                5,
                FastOp::offdiagonal(smallvec![0], 2, smallvec![false], smallvec![false], true),
            ),
            (
                6,
                FastOp::offdiagonal(smallvec![1], 3, smallvec![false], smallvec![false], true),
            ),
            (
                7,
                FastOp::offdiagonal(smallvec![0], 2, smallvec![false], smallvec![false], true),
            ),
            (
                8,
                FastOp::offdiagonal(smallvec![1], 3, smallvec![false], smallvec![false], true),
            ),
        ]
        .into_iter(),
    );
    let edges = EN {
        bonds_for_var: vec![vec![0, 1], vec![0, 1]],
        bonds: vec![((0, 1), true), ((0, 1), false)],
    };
    let mut state = vec![false, false];
    let mut rng = SmallRng::seed_from_u64(0);
    (0..100).for_each(|_| {
        manager.rvb_update(&edges, &mut state, 1, &mut rng);
    });
    println!("{:?}", state);
    assert!(manager.verify(&state));
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
        ising.set_run_rvb(true).unwrap();
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
        let mut ising = DefaultQMCIsingGraph::<SmallRng>::new_with_rng(
            edges,
            0.1,
            l * l,
            rng,
            Some(vec![false; l * l]),
        );
        ising.set_run_rvb(true).unwrap();
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
        let mut ising = DefaultQMCIsingGraph::<SmallRng>::new_with_rng(
            edges,
            1.0,
            nvars,
            rng,
            Some(vec![false; nvars]),
        );
        ising.timesteps(1000, 1.0);
        ising.set_run_rvb(true).unwrap();
        ising.timesteps(1000, 1.0);
        assert!(ising.verify());
    }
}
