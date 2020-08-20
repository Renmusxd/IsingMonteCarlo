use crate::sse::qmc_traits::*;
use rand::Rng;
#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};

/// This does an SSE equivalent of the classical dimer loop update. Requires ising symmetry and
/// a set of bonds which are equivalent but either sat or broken.
pub trait ClassicalLoopUpdater: DiagonalUpdater {
    /// Check if variable is ever flipped by an offdiagonal op.
    fn var_ever_flips(&self, var: usize) -> bool;

    /// Perform an update, return size of cluster.
    fn run_classical_loop_update<R: Rng, EN: EdgeNavigator>(
        &mut self,
        edges: &EN,
        state: &mut [bool],
        mut rng: R,
    ) -> usize {
        let nvars = self.get_nvars();
        let mut in_cluster = self.get_var_alloc(nvars);
        let mut sat_set = self.get_sat_alloc();
        let mut broken_set = self.get_broken_alloc();

        let is_sat = |bond: usize| -> bool {
            let p_aligned = edges.bond_prefers_aligned(bond);
            let (a, b) = edges.vars_for_bond(bond);
            let aligned = state[a] == state[b];
            aligned == p_aligned
        };

        // Find the cluster.
        let mut boundary_alloc = self.get_boundary_alloc();
        let cluster_size = grow_classical_cluster(
            edges,
            is_sat,
            (&mut sat_set, &mut broken_set),
            &mut in_cluster,
            &mut boundary_alloc,
            |var| self.var_ever_flips(var),
            &mut rng,
        );
        self.return_boundary_alloc(boundary_alloc);

        // Flip the cluster.
        state.iter_mut().zip(in_cluster.iter()).for_each(|(s, c)| {
            if *c {
                *s = !*s
            }
        });

        // Move ops around the border.
        self.mutate_ps(
            self.get_cutoff(),
            (state, rng),
            |_, op, (state, mut rng)| {
                let new_op = if let Some(op) = op {
                    // Check input.
                    debug_assert!(op.get_vars().iter().zip(op.get_inputs().iter()).all(
                        |(v, b)| {
                            if in_cluster[*v] {
                                state[*v] != *b
                            } else {
                                state[*v] == *b
                            }
                        }
                    ));

                    // Now, if the op has no vars in the cluster ignore it. If all are in the
                    // cluster then just flip inputs and outputs, if mixed then see later.
                    let (all_in, all_out) =
                        op.get_vars().iter().cloned().map(|v| in_cluster[v]).fold(
                            (true, true),
                            |(all_true, all_false), b| {
                                let all_true = all_true & b;
                                let all_false = all_false & !b;
                                (all_true, all_false)
                            },
                        );
                    if all_out {
                        // Set the state. (could be offdiagonal)
                        op.get_vars()
                            .iter()
                            .zip(op.get_outputs().iter())
                            .for_each(|(v, b)| {
                                state[*v] = *b;
                            });
                        None
                    } else if all_in {
                        // Flip all inputs, otherwise the same.
                        let vars = Self::Op::make_vars(op.get_vars().iter().cloned());
                        let inputs = Self::Op::make_substate(op.get_inputs().iter().map(|b| !*b));
                        let outputs = Self::Op::make_substate(op.get_outputs().iter().map(|b| !*b));
                        let new_op = Self::Op::offdiagonal(
                            vars,
                            op.get_bond(),
                            inputs,
                            outputs,
                            op.is_constant(),
                        );
                        // Set the state. (could be offdiagonal)
                        new_op
                            .get_vars()
                            .iter()
                            .zip(new_op.get_outputs().iter())
                            .for_each(|(v, b)| {
                                state[*v] = *b;
                            });
                        Some(Some(new_op))
                    } else {
                        // We are only covering 2-variable edges.
                        debug_assert_eq!(op.get_vars().len(), 2);
                        // We only move diagonal ops.
                        debug_assert!(op.is_diagonal());
                        let bond = op.get_bond();
                        let new_bond = if sat_set.contains(&bond) {
                            broken_set.get_random(&mut rng).unwrap()
                        } else {
                            debug_assert!(
                                broken_set.contains(&bond),
                                "Bond failed to be broken or not broken: {}",
                                bond
                            );
                            sat_set.get_random(&mut rng).unwrap()
                        };
                        let (new_a, new_b) = edges.vars_for_bond(*new_bond);
                        let vars = Self::Op::make_vars([new_a, new_b].iter().cloned());
                        let state =
                            Self::Op::make_substate([new_a, new_b].iter().map(|v| state[*v]));
                        let new_op = Self::Op::diagonal(vars, *new_bond, state, op.is_constant());

                        Some(Some(new_op))
                    }
                } else {
                    None
                };
                (new_op, (state, rng))
            },
        );

        self.return_sat_alloc(sat_set);
        self.return_broken_alloc(broken_set);
        self.return_var_alloc(in_cluster);
        self.post_semiclassical_update_hook();
        cluster_size
    }

    /// Get the allocation.
    fn get_var_alloc(&mut self, nvars: usize) -> Vec<bool> {
        let mut alloc = Vec::default();
        alloc.resize(nvars, false);
        alloc
    }
    /// Return the allocation.
    fn return_var_alloc(&mut self, _alloc: Vec<bool>) {}

    /// Get the allocation.
    fn get_boundary_alloc(&mut self) -> Vec<(usize, bool)> {
        Vec::default()
    }

    /// Return the allocation.
    fn return_boundary_alloc(&mut self, _alloc: Vec<(usize, bool)>) {}

    /// Get the allocation
    fn get_sat_alloc(&mut self) -> BondContainer<usize> {
        BondContainer::default()
    }
    /// Get the allocation
    fn get_broken_alloc(&mut self) -> BondContainer<usize> {
        BondContainer::default()
    }

    /// Return the allocation.
    fn return_sat_alloc(&mut self, _alloc: BondContainer<usize>) {}
    /// Return the allocation.
    fn return_broken_alloc(&mut self, _alloc: BondContainer<usize>) {}

    /// Called after an update.
    fn post_semiclassical_update_hook(&mut self) {}
}

fn grow_classical_cluster<EN: EdgeNavigator, R: Rng, F, G>(
    edges: &EN,
    is_sat: F,
    sets: (&mut BondContainer<usize>, &mut BondContainer<usize>),
    in_cluster: &mut [bool],
    recursive_search_alloc: &mut Vec<(usize, bool)>,
    skip_var: G,
    mut rng: R,
) -> usize
where
    F: Fn(usize) -> bool + Copy,
    G: Fn(usize) -> bool + Copy,
{
    let nvars = in_cluster.len();
    let (sat_set, broken_set) = sets;
    let starting_var = rng.gen_range(0, nvars);

    let mut cluster_size = add_var(
        edges,
        is_sat,
        skip_var,
        (sat_set, broken_set),
        in_cluster,
        starting_var,
        recursive_search_alloc,
    );

    while sat_set.len() != broken_set.len() {
        let bond = if sat_set.len() > broken_set.len() {
            sat_set.pop_random(&mut rng).unwrap()
        } else {
            broken_set.pop_random(&mut rng).unwrap()
        };
        let (a, b) = edges.vars_for_bond(bond);

        cluster_size += match (in_cluster[a], in_cluster[b]) {
            (true, false) => add_var(
                edges,
                is_sat,
                skip_var,
                (sat_set, broken_set),
                in_cluster,
                b,
                recursive_search_alloc,
            ),
            (false, true) => add_var(
                edges,
                is_sat,
                skip_var,
                (sat_set, broken_set),
                in_cluster,
                a,
                recursive_search_alloc,
            ),
            _ => unreachable!(),
        }
    }
    cluster_size
}

fn add_var<EN, F, G>(
    edges: &EN,
    is_sat: F,
    skip_var: G,
    sets: (&mut BondContainer<usize>, &mut BondContainer<usize>),
    in_cluster: &mut [bool],
    var: usize,
    recursive_search_alloc: &mut Vec<(usize, bool)>,
) -> usize
where
    EN: EdgeNavigator,
    F: Fn(usize) -> bool,
    G: Fn(usize) -> bool,
{
    let skip_a = skip_var(var);
    recursive_search_alloc.push((var, skip_a));
    add_vars(
        edges,
        is_sat,
        skip_var,
        sets,
        in_cluster,
        recursive_search_alloc,
    )
}

fn add_vars<EN, F, G>(
    edges: &EN,
    is_sat: F,
    skip_var: G,
    sets: (&mut BondContainer<usize>, &mut BondContainer<usize>),
    in_cluster: &mut [bool],
    vars: &mut Vec<(usize, bool)>,
) -> usize
where
    EN: EdgeNavigator,
    F: Fn(usize) -> bool,
    G: Fn(usize) -> bool,
{
    let (sat_set, broken_set) = sets;
    let mut cluster_size = 0;
    while let Some((a, skip_a)) = vars.pop() {
        if !in_cluster[a] {
            cluster_size += 1;
            in_cluster[a] = true;
            let bonds = edges.bonds_for_var(a);
            for bond in bonds.iter().cloned() {
                let b = edges.other_var_for_bond(a, bond).unwrap();
                let skip_b = skip_var(b);
                if skip_a || skip_b {
                    vars.push((b, skip_b))
                } else if is_sat(bond) {
                    if sat_set.contains(&bond) {
                        sat_set.remove(&bond);
                    } else if !in_cluster[b] {
                        sat_set.insert(bond);
                    }
                // else { if !is_sat(bond) {...} }
                } else if broken_set.contains(&bond) {
                    broken_set.remove(&bond);
                } else if !in_cluster[b] {
                    broken_set.insert(bond);
                }
            }
        }
    }
    cluster_size
}

/// A struct which allows navigation around the variables in a model.
pub trait EdgeNavigator {
    /// Get the bonds attached to this variable.
    fn bonds_for_var(&self, var: usize) -> &[usize];
    /// Get the variables associated with this bond.
    fn vars_for_bond(&self, bond: usize) -> (usize, usize);
    /// Does the bond prefer aligned variables or antialigned.
    fn bond_prefers_aligned(&self, bond: usize) -> bool;
    /// Get the other variable attached by a bond
    fn other_var_for_bond(&self, var: usize, bond: usize) -> Option<usize> {
        let (a, b) = self.vars_for_bond(bond);
        if var == a {
            Some(b)
        } else if var == b {
            Some(a)
        } else {
            None
        }
    }
}

/// A HashSet with random sampling.
#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct BondContainer<T: Clone + Into<usize>> {
    map: Vec<Option<usize>>,
    keys: Vec<T>,
}

impl<T: Clone + Into<usize>> BondContainer<T> {
    /// Get a random entry from the HashSampler
    pub fn get_random<R: Rng>(&self, mut r: R) -> Option<&T> {
        if self.keys.is_empty() {
            None
        } else {
            // Choose a key to remove
            let index = r.gen_range(0, self.keys.len());
            Some(&self.keys[index])
        }
    }

    /// Pop a random element.
    pub fn pop_random<R: Rng>(&mut self, mut r: R) -> Option<T> {
        if self.keys.is_empty() {
            None
        } else {
            // Choose a key to remove
            let keys_index = r.gen_range(0, self.keys.len());
            Some(self.remove_index(keys_index))
        }
    }

    /// Remove a given value.
    pub fn remove(&mut self, value: &T) -> bool {
        let bond_number = value.clone().into();
        let address = self.map[bond_number];
        if let Some(address) = address {
            self.remove_index(address);
            true
        } else {
            false
        }
    }

    fn remove_index(&mut self, keys_index: usize) -> T {
        // Move key to last position.
        let last_indx = self.keys.len() - 1;
        self.keys.swap(keys_index, last_indx);
        // Update address
        let bond_number = self.keys[keys_index].clone().into();
        let old_indx = self.map[bond_number].as_mut().unwrap();
        *old_indx = keys_index;
        // Remove key
        let out = self.keys.pop().unwrap();
        self.map[out.clone().into()] = None;
        out
    }

    /// Check if a given element has been inserted.
    pub fn contains(&self, value: &T) -> bool {
        let t = value.clone().into();
        if t >= self.map.len() {
            false
        } else {
            self.map[t].is_some()
        }
    }

    /// Insert an element.
    pub fn insert(&mut self, value: T) -> bool {
        let entry_index = value.clone().into();
        if entry_index >= self.map.len() {
            self.map.resize(entry_index + 1, None);
        }
        match self.map[entry_index] {
            Some(_) => false,
            None => {
                self.map[entry_index] = Some(self.keys.len());
                self.keys.push(value);
                true
            }
        }
    }

    /// Clear the hashset
    pub fn clear(&mut self) {
        self.keys.clear();
        self.map.clear();
    }

    /// Get number of elements in set.
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }
}

#[cfg(test)]
mod sc_tests {
    use super::*;
    use rand::prelude::SmallRng;
    use rand::SeedableRng;

    #[test]
    fn test_bond_container() {
        let mut bc = BondContainer::<usize>::default();
        bc.insert(0);
        assert_eq!(bc.pop_random(SmallRng::seed_from_u64(0)), Some(0));
    }

    #[test]
    fn test_bond_container_more() {
        let mut bc = BondContainer::<usize>::default();
        bc.insert(0);
        bc.insert(1);
        bc.insert(2);
        assert_eq!(bc.len(), 3);
        let res = match bc.pop_random(SmallRng::seed_from_u64(0)) {
            Some(0) | Some(1) | Some(2) => true,
            _ => false,
        };
        assert!(res)
    }

    struct TestEdges {
        bonds: Vec<Vec<usize>>,
        edges: Vec<(usize, usize, bool)>,
    }

    impl TestEdges {
        fn new(nvars: usize, edges: &[(usize, usize, bool)]) -> Self {
            let mut edge_lookup = vec![vec![]; nvars];
            edges
                .iter()
                .cloned()
                .enumerate()
                .for_each(|(bond, (a, b, _))| {
                    edge_lookup[a].push(bond);
                    edge_lookup[b].push(bond);
                });
            Self {
                bonds: edge_lookup,
                edges: edges.to_vec(),
            }
        }
    }

    impl EdgeNavigator for TestEdges {
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

    fn is_sat<EN: EdgeNavigator>(edges: &EN, state: &[bool], bond: usize) -> bool {
        let (a, b) = edges.vars_for_bond(bond);
        let p_aligned = edges.bond_prefers_aligned(bond);
        let aligned = state[a] == state[b];
        aligned == p_aligned
    }

    #[test]
    fn singlevar_test() {
        let edges = TestEdges::new(1, &[]);
        let state = vec![false];
        let mut in_cluster = vec![false];
        let (mut a, mut b) = (BondContainer::default(), BondContainer::default());
        grow_classical_cluster(
            &edges,
            |bond| is_sat(&edges, &state, bond),
            (&mut a, &mut b),
            &mut in_cluster,
            &mut Vec::default(),
            |_| false,
            SmallRng::seed_from_u64(1234),
        );
        assert_eq!(in_cluster, vec![true])
    }

    #[test]
    fn twovar_test_whole() {
        let edges = TestEdges::new(2, &[(0, 1, true)]);
        let state = vec![false, false];
        let mut in_cluster = vec![false, false];
        let (mut a, mut b) = (BondContainer::default(), BondContainer::default());
        grow_classical_cluster(
            &edges,
            |bond| is_sat(&edges, &state, bond),
            (&mut a, &mut b),
            &mut in_cluster,
            &mut Vec::default(),
            |_| false,
            SmallRng::seed_from_u64(1234),
        );
        assert_eq!(in_cluster, vec![true, true])
    }

    #[test]
    fn threevar_test_halfa() {
        for i in 0..1024 {
            let edges = TestEdges::new(3, &[(0, 1, true), (0, 2, false), (1, 2, false)]);
            let state = vec![false, false, false];
            let mut in_cluster = vec![false, false, false];
            let (mut a, mut b) = (BondContainer::default(), BondContainer::default());
            grow_classical_cluster(
                &edges,
                |bond| is_sat(&edges, &state, bond),
                (&mut a, &mut b),
                &mut in_cluster,
                &mut Vec::default(),
                |_| false,
                SmallRng::seed_from_u64(i),
            );
            let a = match in_cluster.as_slice() {
                &[true, false, false] => true,
                &[false, true, false] => true,
                &[false, true, true] => true,
                &[true, false, true] => true,
                _ => false,
            };
            assert!(a)
        }
    }

    #[test]
    fn threevar_test_halfb() {
        for i in 0..1024 {
            let edges = TestEdges::new(3, &[(0, 1, false), (0, 2, true), (1, 2, false)]);
            let state = vec![false, false, false];
            let mut in_cluster = vec![false, false, false];
            let (mut a, mut b) = (BondContainer::default(), BondContainer::default());
            grow_classical_cluster(
                &edges,
                |bond| is_sat(&edges, &state, bond),
                (&mut a, &mut b),
                &mut in_cluster,
                &mut Vec::default(),
                |_| false,
                SmallRng::seed_from_u64(i),
            );
            let a = match in_cluster.as_slice() {
                &[true, false, false] => true,
                &[false, false, true] => true,
                &[false, true, true] => true,
                &[true, true, false] => true,
                _ => false,
            };
            assert!(a)
        }
    }

    #[test]
    fn threevar_test_all() {
        for i in 0..1024 {
            let edges = TestEdges::new(3, &[(0, 1, false), (0, 2, false), (1, 2, false)]);
            let state = vec![false, false, false];
            let mut in_cluster = vec![false, false, false];
            let (mut a, mut b) = (BondContainer::default(), BondContainer::default());
            grow_classical_cluster(
                &edges,
                |bond| is_sat(&edges, &state, bond),
                (&mut a, &mut b),
                &mut in_cluster,
                &mut Vec::default(),
                |_| false,
                SmallRng::seed_from_u64(i),
            );
            let a = match in_cluster.as_slice() {
                &[true, true, true] => true,
                _ => false,
            };
            assert!(a)
        }
    }
}
