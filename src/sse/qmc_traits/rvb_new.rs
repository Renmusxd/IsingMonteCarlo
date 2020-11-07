use crate::memory::allocator::{Factory, Reset};
use crate::sse::{DiagonalUpdater, Op};
use rand::Rng;
#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};

const SENTINEL_FLIP_POSITION: usize = std::usize::MAX;

/// Resonating bond update.
pub trait RVBUpdater: DiagonalUpdater {
    /// Perform a resonating bond update.
    fn rvb_update<R: Rng, EN: EdgeNavigator>(
        &mut self,
        edges: &EN,
        state: &mut [bool],
        mut rng: R,
    ) -> (usize, bool) {
        // First get the cluster
        // This should be done by starting with a given variable, and a starting and adjacent ending
        // position for the cluster (two nearby constant ops).
        // Then use BFS to connect adjacent sections of starting and ending positions with some
        // probability
        // Output should be a datastructure which lists the p values for each variable entering and
        // exiting the cluster.
        // Example: vector of p positions and vector of variable indices + initial cluster at p=0.

        unimplemented!()
    }

    /// Fill `ps` with the p values of constant (Hij=k) ops for a given var.
    /// An implementation by the same name is provided for LoopUpdaters
    fn constant_ops_on_var(&self, var: usize, ps: &mut Vec<usize>);

    /// Fill `ps` with the p values of spin flips for a given var.
    /// An implementation by the same name is provided for LoopUpdaters
    fn spin_flips_on_var(&self, var: usize, ps: &mut Vec<usize>);
}

/// Get returns n with chance 1/2^(n+1)
/// Chance of failure is 1/2^(2^64) and should therefore be acceptable.
fn contiguous_bits<R: Rng>(mut r: R) -> usize {
    let mut acc = 0;
    let mut v = r.next_u64();
    loop {
        if v == std::u64::MAX {
            acc += 64;
            v = r.next_u64();
        } else {
            while v & 1 == 1 {
                acc += 1;
                v >>= 1;
            }
            break acc;
        }
    }
}

fn set_cluster<F>(vars: &[usize], cluster: &mut [bool], p: usize, bounds: F)
where
    F: Fn(usize) -> (usize, usize),
{
    vars.iter().cloned().enumerate().for_each(|(relv, v)| {
        let flips = bounds(relv);
        let in_cluster = match flips {
            (start, stop) if start == stop => true,
            (start, stop) if start < stop => start < p && p <= stop,
            (start, stop) if start > stop => !(stop < p && p <= start),
            _ => unreachable!(),
        };
        cluster[v] = in_cluster;
    });
}

fn fill_bonds<EN, F>(
    edges: &EN,
    is_sat: F,
    vars: &[usize],
    cluster: &[bool],
    state: &mut [bool],
    sat: &mut BondContainer<usize>,
    unsat: &mut BondContainer<usize>,
) where
    EN: EdgeNavigator,
    F: Fn(usize, &[bool]) -> bool,
{
    vars.iter().cloned().filter(|v| cluster[*v]).for_each(|v| {
        edges.bonds_for_var(v).iter().cloned().for_each(|bond| {
            let ov = edges.other_var_for_bond(v, bond).unwrap();
            if !cluster[ov] {
                if is_sat(bond, &state) {
                    sat.insert(bond);
                } else {
                    unsat.insert(bond);
                }
            }
        });
    })
}

fn toggle_state(vars: &[usize], cluster: &[bool], state: &mut [bool]) {
    vars.iter().cloned().filter(|v| cluster[*v]).for_each(|v| {
        state[v] = !state[v];
    })
}

/// Calculate the probability of performing an update on vars in `vars_list`, flips between p values
/// defined by `bounds`.
/// Could leave state modified.
fn calculate_graph_flip_prob<RVB, EN, F, G>(
    rvb: &RVB,
    (edges, is_sat): (&EN, F),
    (vars_list, bounds): (&[usize], G),
    relevant_spin_flips: &[usize],
    (sat_bonds, unsat_bonds): (&mut BondContainer<usize>, &mut BondContainer<usize>),
    (state, cluster): (&mut [bool], &mut [bool]),
) -> f64
where
    RVB: RVBUpdater + ?Sized,
    EN: EdgeNavigator,
    F: Fn(usize, &[bool]) -> bool + Copy,
    G: Fn(usize) -> (usize, usize) + Copy,
{
    set_cluster(vars_list, cluster, 0, bounds);
    sat_bonds.clear();
    unsat_bonds.clear();
    fill_bonds(
        edges,
        is_sat,
        vars_list,
        cluster,
        state,
        sat_bonds,
        unsat_bonds,
    );

    let t = (
        0usize,
        (sat_bonds, unsat_bonds),
        (0, 0),
        1.0f64,
        (state, cluster),
    );
    let res = rvb.try_iterate_ops(0, rvb.get_cutoff(), t, |_, op, p, acc| {
        let (mut next_flip, bonds, op_counts, mut mult, state_clust) = acc;
        let (sat_bonds, unsat_bonds) = bonds;
        let (mut nf, mut na) = op_counts;
        let (state, cluster) = state_clust;
        // Check consistency.
        debug_assert!(op
            .get_vars()
            .iter()
            .cloned()
            .zip(op.get_inputs().iter().cloned())
            .all(|(v, b)| { state[v] == b }));

        // Set state
        op.get_vars()
            .iter()
            .cloned()
            .zip(op.get_outputs().iter().cloned())
            .for_each(|(v, b)| {
                state[v] = b;
            });

        match relevant_spin_flips.get(next_flip) {
            Some(flip_p) if *flip_p == p => {
                // Get new bonds and mult.
                mult *= calculate_mult(sat_bonds, unsat_bonds, nf, na);
                if mult <= std::f64::EPSILON {
                    return Err(());
                }
                // Use p+1 since we measure from inputs not outputs.
                set_cluster(vars_list, cluster, p + 1, bounds);

                sat_bonds.clear();
                unsat_bonds.clear();
                na = 0;
                nf = 0;
                fill_bonds(
                    edges,
                    is_sat,
                    vars_list,
                    cluster,
                    state,
                    sat_bonds,
                    unsat_bonds,
                );

                // Technically we may need to increment more than once because there are some flip
                // positions that correspond to None ops (for vars with no constant ops). These are
                // places at SENTINEL_FLIP_POSITION which is larger than any reasonable world line,
                // we will run out of ops long before then.
                next_flip += 1;
            }
            _ => {
                // See whether this op is nf or na.
                if sat_bonds.contains(&op.get_bond()) {
                    // (X/0)^(0 - Y) = 0
                    if unsat_bonds.is_empty() {
                        return Err(());
                    }
                    nf += 1;
                } else if unsat_bonds.contains(&op.get_bond()) {
                    // (0/X)^(Y - 0) = 0
                    if sat_bonds.is_empty() {
                        return Err(());
                    }
                    na += 1;
                }
            }
        };

        Ok((
            next_flip,
            (sat_bonds, unsat_bonds),
            (nf, na),
            mult,
            (state, cluster),
        ))
    });
    if let Ok((_, bonds, op_counts, mult, _)) = res {
        if mult > std::f64::EPSILON {
            let (sat_bonds, unsat_bonds) = bonds;

            let (nf, na) = op_counts;
            mult * calculate_mult(sat_bonds, unsat_bonds, nf, na)
        } else {
            0.0
        }
    } else {
        0.0
    }
}

fn calculate_mult(
    sat: &BondContainer<usize>,
    unsat: &BondContainer<usize>,
    nf: i32,
    na: i32,
) -> f64 {
    if nf == na || sat.len() == unsat.len() {
        1.0
    } else {
        (sat.len() as f64 / unsat.len() as f64).powi(na - nf)
    }
}

pub(crate) fn check_borders<O: Op>(op: &O, in_cluster: &[bool]) -> (bool, bool) {
    op.get_vars().iter().cloned().map(|v| in_cluster[v]).fold(
        (true, true),
        |(all_true, all_false), b| {
            let all_true = all_true & b;
            let all_false = all_false & !b;
            (all_true, all_false)
        },
    )
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
