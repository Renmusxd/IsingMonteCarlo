extern crate rand;
use qmc::sse::fast_ops::*;
use qmc::sse::qmc_traits::*;
use qmc::sse::semi_classical::*;
use rand::prelude::SmallRng;
use rand::SeedableRng;
use smallvec::smallvec;

struct EdgeNav {
    bonds: Vec<Vec<usize>>,
    edges: Vec<(usize, usize, bool)>,
}

impl EdgeNavigator for EdgeNav {
    fn bonds_for_var(&self, var: usize) -> &[usize] {
        &self.bonds[var]
    }

    fn vars_for_bond(&self, bond: usize) -> (usize, usize) {
        (self.edges[bond].0, self.edges[bond].1)
    }

    fn bond_prefers_aligned(&self, bond: usize) -> bool {
        self.edges[bond].2
    }
}

fn manager_and_edges(nvars: usize, edges: &[(usize, usize, bool)]) -> (FastOps, EdgeNav) {
    let mut edge_lookup = vec![vec![]; nvars];
    let ops = edges.iter().cloned().enumerate().map(|(p, (a, b, _))| {
        edge_lookup[a].push(p);
        edge_lookup[b].push(p);
        (
            p,
            FastOp::offdiagonal(
                smallvec![a, b],
                p,
                smallvec![false, false],
                smallvec![false, false],
                false,
            ),
        )
    });

    let manager = FastOps::new_from_ops(nvars, ops);
    let edges = EdgeNav {
        bonds: edge_lookup,
        edges: edges.to_vec(),
    };
    (manager, edges)
}

#[test]
fn single_classical_test() {
    for i in 0..1024 {
        let edges = [(0, 1, true), (1, 2, false), (2, 0, false)];
        let (mut manager, edges) = manager_and_edges(3, &edges);
        let mut rng = SmallRng::seed_from_u64(i);
        let mut state = vec![false; manager.get_nvars()];
        manager.run_classical_loop_update(&edges, &mut state, &mut rng);
        assert!(manager.verify(&state));
        let state_flipped = match state.as_slice() {
            &[true, false, false] => true,
            &[false, true, false] => true,
            &[true, false, true] => true,
            &[false, true, true] => true,
            _ => false,
        };
        if !state_flipped {
            println!("state: {:?}", state);
        }
        assert!(state_flipped);
    }
}

#[test]
fn single_classical_test_whole() {
    let edges = [(0, 1, true), (1, 2, true), (2, 3, true), (3, 0, true)];
    let (mut manager, edges) = manager_and_edges(4, &edges);
    let mut rng = SmallRng::seed_from_u64(0);
    let mut state = vec![false; manager.get_nvars()];
    manager.run_classical_loop_update(&edges, &mut state, &mut rng);
    assert!(manager.verify(&state));
    assert_eq!(state, vec![true; manager.get_nvars()]);
}

#[test]
fn single_classical_test_whole_false() {
    let edges = [(0, 1, false), (1, 2, false), (2, 3, false), (3, 0, false)];
    let (mut manager, edges) = manager_and_edges(4, &edges);
    let mut rng = SmallRng::seed_from_u64(0);
    let mut state = vec![false; manager.get_nvars()];
    manager.run_classical_loop_update(&edges, &mut state, &mut rng);
    assert!(manager.verify(&state));
    assert_eq!(state, vec![true; manager.get_nvars()]);
}

#[test]
fn single_classical_test_half() {
    for i in 0..1024 {
        let nvars = 10;
        let edges_vec = (0..nvars)
            .map(|i| (i, (i + 1) % nvars, i >= nvars >> 1))
            .collect::<Vec<_>>();
        let (mut manager, edges) = manager_and_edges(nvars, &edges_vec);
        let mut rng = SmallRng::seed_from_u64(i);
        let mut state = vec![false; manager.get_nvars()];
        manager.run_classical_loop_update(&edges, &mut state, &mut rng);
        assert!(manager.verify(&state));

        let edges_broken = edges_vec
            .into_iter()
            .filter_map(
                |(a, b, f)| {
                    if state[a] != state[b] {
                        Some(f)
                    } else {
                        None
                    }
                },
            )
            .collect::<Vec<_>>();
        let eb = match edges_broken.as_slice() {
            &[true, false] => true,
            &[false, true] => true,
            _ => false,
        };
        if !eb {
            println!("{:?}\t{:?}", state, edges_broken);
        }
        assert!(eb);
    }
}
