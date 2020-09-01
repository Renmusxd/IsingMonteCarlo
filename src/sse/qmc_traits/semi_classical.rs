use crate::memory::allocator::{Factory, Reset, StackTuplizer};
use crate::sse::qmc_traits::*;
use rand::Rng;
#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};

/// This does an SSE equivalent of the classical dimer loop update. Requires ising symmetry and
/// a set of bonds which are equivalent but either sat or broken.
pub trait ClassicalLoopUpdater:
    DiagonalUpdater + Factory<Vec<bool>> + Factory<Vec<usize>> + Factory<BondContainer<usize>>
{
    /// Check if variable is ever flipped by an offdiagonal op.
    fn var_ever_flips(&self, var: usize) -> bool;

    /// Perform an edge update, return size of cluster and whether it was flipped.
    fn run_semiclassical_edge_update<R: Rng, EN: EdgeNavigator>(
        &mut self,
        edges: &EN,
        state: &mut [bool],
        rng: R,
    ) -> (usize, bool) {
        let bond_select = |s: &mut Self,
                           state: &[bool],
                           in_cluster: &mut [bool],
                           sat_set: &mut BondContainer<usize>,
                           broken_set: &mut BondContainer<usize>,
                           rng: &mut R| {
            let is_sat = |bond: usize| -> bool {
                let p_aligned = edges.bond_prefers_aligned(bond);
                let (a, b) = edges.vars_for_bond(bond);
                let aligned = state[a] == state[b];
                aligned == p_aligned
            };

            // Flip a single edge, flipping more is difficult while preserving detailed balance.
            let bond = rng.gen_range(0, edges.n_bonds());
            let (a, b) = edges.vars_for_bond(bond);
            let mut boundary_alloc = StackTuplizer::<usize, bool>::new(s);

            let skip_var = |var| s.var_ever_flips(var);
            boundary_alloc.push((a, skip_var(a)));
            boundary_alloc.push((b, skip_var(b)));
            let cluster_size = add_vars(
                edges,
                is_sat,
                skip_var,
                (sat_set, broken_set),
                in_cluster,
                &mut boundary_alloc,
            );
            boundary_alloc.dissolve(s);
            cluster_size
        };

        self.run_semiclassical_update(edges, bond_select, state, rng)
    }

    /// Use a function bond_select to select a region of variables and the bordering bonds, then
    /// perform a semiclassical update on those variables by flipping them and rearranging ops on
    /// the bonds. The function bond_select must take the current state, a mutable array giving
    /// whether variables are in the cluster to be flipped, and two mutable BondContainers which
    /// should be filled with bonds on the border of the cluster which are broken or unbroken and
    /// of equal weight, then must return the size of the cluster.
    /// See `run_semiclassical_edge_update` for an implementation.
    fn run_semiclassical_update<R: Rng, EN: EdgeNavigator, F>(
        &mut self,
        edges: &EN,
        bond_select: F,
        state: &mut [bool],
        mut rng: R,
    ) -> (usize, bool)
    where
        F: Fn(
            &mut Self,                 // self
            &[bool],                   // state
            &mut [bool],               // in_cluster
            &mut BondContainer<usize>, // sat_set
            &mut BondContainer<usize>, // broken_set
            &mut R,                    // rng
        ) -> usize,
    {
        let nvars = self.get_nvars();
        let mut in_cluster: Vec<bool> = self.get_instance();
        in_cluster.resize(nvars, false);
        let mut sat_set: BondContainer<usize> = self.get_instance();
        let mut broken_set: BondContainer<usize> = self.get_instance();

        let cluster_size = bond_select(
            self,
            state,
            &mut in_cluster,
            &mut sat_set,
            &mut broken_set,
            &mut rng,
        );

        debug_assert!(!sat_set
            .iter()
            .chain(broken_set.iter())
            .cloned()
            .any(|bond| {
                let (a, b) = edges.vars_for_bond(bond);
                self.var_ever_flips(a) || self.var_ever_flips(b)
            }));

        let should_edit = if sat_set.len() != broken_set.len() {
            let (sat_count, broken_count) = self.count_ops_on_border(&sat_set, &broken_set);
            let p = (sat_set.len() as f64 / broken_set.len() as f64)
                .powi(broken_count as i32 - sat_count as i32);
            if p > 1. {
                true
            } else {
                rng.gen_bool(p)
            }
        } else {
            true
        };

        // If this can be flipped, then set the right cluster size and do work, otherwise set to 0.
        if should_edit {
            // Flip the cluster.
            state.iter_mut().zip(in_cluster.iter()).for_each(|(s, c)| {
                if *c {
                    *s = !*s
                }
            });

            // Move ops around the border.
            self.mutate_ops(
                self.get_cutoff(),
                (state, rng),
                |_, op, _, (state, mut rng)| {
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
                    let (all_in, all_out) = check_borders(op, &in_cluster);
                    let new_op = if all_out {
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
                        let mut new_op = op.clone();
                        let (ins, outs) = new_op.get_mut_inputs_and_outputs();
                        ins.iter_mut().for_each(|b| *b = !*b);
                        outs.iter_mut().for_each(|b| *b = !*b);

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
                    };
                    (new_op, (state, rng))
                },
            );
        };
        self.return_instance(sat_set);
        self.return_instance(broken_set);
        self.return_instance(in_cluster);
        self.post_semiclassical_update_hook();
        (cluster_size, should_edit)
    }

    /// Count the number of bonds on border which belong to set_set and broken_set.
    fn count_ops_on_border(
        &self,
        sat_set: &BondContainer<usize>,
        broken_set: &BondContainer<usize>,
    ) -> (usize, usize) {
        count_using_iter_ops(self, sat_set, broken_set)
    }

    /// Called after an update.
    fn post_semiclassical_update_hook(&mut self) {}
}

/// Count ops using the iter_ops call.
pub fn count_using_iter_ops<C: ClassicalLoopUpdater + ?Sized>(
    c: &C,
    sat_set: &BondContainer<usize>,
    broken_set: &BondContainer<usize>,
) -> (usize, usize) {
    c.iterate_ops((0, 0), |_, op, _, acc| {
        let (sat, broken) = acc;
        if sat_set.contains(&op.get_bond()) {
            (sat + 1, broken)
        } else if broken_set.contains(&op.get_bond()) {
            (sat, broken + 1)
        } else {
            (sat, broken)
        }
    })
}

fn add_vars<EN, F, G>(
    edges: &EN,
    is_sat: F,
    skip_var: G,
    sets: (&mut BondContainer<usize>, &mut BondContainer<usize>),
    in_cluster: &mut [bool],
    vars: &mut StackTuplizer<usize, bool>,
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
    /// Number of bonds
    fn n_bonds(&self) -> usize;
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
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct BondContainer<T: Clone + Into<usize>> {
    map: Vec<Option<usize>>,
    keys: Option<Vec<T>>,
}

impl<T: Clone + Into<usize>> Reset for BondContainer<T> {
    fn reset(&mut self) {
        self.clear();
    }
}

impl<T: Clone + Into<usize>> Default for BondContainer<T> {
    fn default() -> Self {
        Self {
            map: Vec::default(),
            keys: Some(Vec::default()),
        }
    }
}

impl<T: Clone + Into<usize>> BondContainer<T> {
    /// Get a random entry from the HashSampler
    pub fn get_random<R: Rng>(&self, mut r: R) -> Option<&T> {
        let keys = self.keys.as_ref().unwrap();
        if keys.is_empty() {
            None
        } else {
            // Choose a key to remove
            let index = r.gen_range(0, keys.len());
            Some(&keys[index])
        }
    }

    /// Pop a random element.
    pub fn pop_random<R: Rng>(&mut self, mut r: R) -> Option<T> {
        let keys = self.keys.as_ref().unwrap();
        if keys.is_empty() {
            None
        } else {
            // Choose a key to remove
            let keys_index = r.gen_range(0, keys.len());
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
        let keys = self.keys.as_mut().unwrap();
        let last_indx = keys.len() - 1;
        keys.swap(keys_index, last_indx);
        // Update address
        let bond_number = keys[keys_index].clone().into();
        let old_indx = self.map[bond_number].as_mut().unwrap();
        *old_indx = keys_index;
        // Remove key
        let out = self.keys.as_mut().unwrap().pop().unwrap();
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
                let keys = self.keys.as_mut().unwrap();
                self.map[entry_index] = Some(keys.len());
                keys.push(value);
                true
            }
        }
    }

    /// Clear the set
    pub fn clear(&mut self) {
        let mut keys = self.keys.take().unwrap();
        keys.iter().for_each(|k| {
            let bond = k.clone().into();
            self.map[bond] = None
        });
        keys.clear();
        self.keys = Some(keys);
    }

    /// Get number of elements in set.
    pub fn len(&self) -> usize {
        self.keys.as_ref().unwrap().len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.keys.as_ref().unwrap().is_empty()
    }

    /// Iterate through items.
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.keys.as_ref().unwrap().iter()
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
}
