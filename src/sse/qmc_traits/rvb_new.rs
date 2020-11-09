use crate::memory::allocator::{Factory, Reset};
use crate::sse::{DiagonalUpdater, Op};
use rand::Rng;
#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};

const SENTINEL_FLIP_POSITION: usize = std::usize::MAX;

/// Resonating bond update.
trait RVBUpdater: DiagonalUpdater + Factory<Vec<usize>> + Factory<Vec<bool>> {
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

    /// Append `ps` with the p values of constant (Hij=k) ops for a given var.
    /// An implementation by the same name is provided for LoopUpdaters
    fn constant_ops_on_var(&self, var: usize, ps: &mut Vec<usize>);

    /// Fill `ps` with the p values of spin flips for a given var.
    /// An implementation by the same name is provided for LoopUpdaters
    fn spin_flips_on_var(&self, var: usize, ps: &mut Vec<usize>);
}

fn make_cluster_of_size<RVB: RVBUpdater, EN: EdgeNavigator, R: Rng>(
    rvb: &mut RVB,
    edges: &EN,
    mut size: usize,
    starting_cluster: &mut [bool],
    cluster_vars: &mut Vec<usize>,
    cluster_ends: &mut Vec<usize>,
    rng: &mut R,
) {
    if size == 0 {
        return;
    }
    starting_cluster.iter_mut().for_each(|b| *b = false);
    let mut helper = BFSHelper::new(rvb);

    let starting_var = rng.gen_range(0, rvb.get_nvars());
    helper.populate_ops_for_var(rvb, starting_var);
    let starting_flip = rng.gen_range(0, helper.get_constant_ops_for_var(starting_var).len());
    helper.push_flip(starting_var, starting_flip);

    let mut to_add = helper.pop_flip(rng);
    while let Some((var, flip_index)) = to_add {
        let flips = helper.get_constant_ops_for_var(var);
        let flip_start = flips[flip_index];
        let flip_end = flips[(flip_index + 1) % flips.len()];

        if flip_end <= flip_start {
            starting_cluster[var] = true;
        }
        cluster_vars.push(var);
        cluster_ends.push(flip_start);
        cluster_ends.push(flip_end);

        // So we added (s, e) for var, now add adjacent to list.
        let prev = (flip_index + flips.len() - 1) % flips.len();
        let next = (flip_index + 1) % flips.len();
        helper.push_flip(var, prev);
        helper.push_flip(var, next);
        // And adjacent vars too...
        edges
            .bonds_for_var(var)
            .iter()
            .filter_map(|b| edges.other_var_for_bond(var, *b))
            .for_each(|ov| {
                helper.populate_ops_for_var(rvb, ov);
                let ov_flips = helper.get_constant_ops_for_var(ov);
                if flip_start == flip_end {
                    // Handle edge case where start == end.
                    (0..ov_flips).for_each(|flip_start| helper.push_flip(ov, flip_start));
                } else {
                    // Find the position where flip_start would live.
                    let flip_pos = ov_flips.binary_search(&flip_start).unwrap_err();
                    // TODO add all adjacent blocks to (flip_start, next)
                    // This is definitely not done...
                    let mut starting_pos = (flip_pos + ov_flips.len() - 1) % ov_flips.len();
                    let mut ending_pos = flip_pos;
                    while in_cluster(flip_start, flip_end, ov_flips[ending_pos]) {
                        helper.push_flip(ov, starting_pos);
                        starting_pos = ending_pos;
                        ending_pos = (ending_pos + 1) % ov_flips.len();
                    }
                    // Now we are in a state where the ending pos is outside the cluster, but the
                    // starting pos is either before or inside, so add once more.
                    helper.push_flip(ov, starting_pos);
                }
            });

        size -= 1;
        if size == 0 {
            break;
        }
        to_add = helper.pop_flip(rng);
    }

    cluster_vars.sort_unstable();
    cluster_vars.dedup();
    cluster_ends.sort_unstable();
    drop_doubles(cluster_ends);

    helper.cleanup(rvb);
}

/// Returns true if p between start and stop with periodic bounds.
fn in_cluster(start: usize, stop: usize, p: usize) -> bool {
    match (start, stop) {
        (start, stop) if start == stop => true,
        (start, stop) if start < stop => start < p && p <= stop,
        (start, stop) if start > stop => !(stop < p && p <= start),
        _ => unreachable!(),
    }
}

fn drop_doubles(v: &mut Vec<usize>) {
    let mut ii = 0;
    let mut jj = 0;
    while jj < v.len() {
        let can_have_next = jj + 1 < v.len();
        if can_have_next && v[jj] == v[jj + 1] {
            // Two in a row, add neither
            jj += 2;
        } else {
            v[ii] = v[jj];
            ii += 1;
            jj += 1;
        }
    }
    v.resize(ii, 0);
}

struct BFSHelper {
    var_lookup: Vec<usize>,
    var_start_points: Vec<usize>,
    var_constant_pos: Vec<usize>,
    bfs_var: Vec<usize>,
    bfs_flip_pos: Vec<usize>,
    flip_start_seen: Vec<bool>,
}

impl BFSHelper {
    fn new<F: Factory<Vec<usize>> + Factory<Vec<bool>>>(f: &mut F) -> Self {
        Self {
            var_lookup: f.get_instance(),
            var_start_points: f.get_instance(),
            var_constant_pos: f.get_instance(),
            bfs_var: f.get_instance(),
            bfs_flip_pos: f.get_instance(),
            flip_start_seen: f.get_instance(),
        }
    }

    fn populate_ops_for_var<RVB: RVBUpdater>(&mut self, rvb: &mut RVB, var: usize) {
        if self.var_lookup.len() < var {
            self.var_lookup.resize(var + 1, std::usize::MAX);
        }
        if self.var_lookup[var] == std::usize::MAX {
            self.var_lookup[var] = self.var_start_points.len();
            self.var_start_points.push(self.var_constant_pos.len());
            rvb.constant_ops_on_var(var, &mut self.var_constant_pos);
        }
    }

    fn has_populated_ops_for_var(&self, var: usize) -> bool {
        self.var_lookup.len() > var && self.var_lookup[var] != std::usize::MAX
    }

    fn get_constant_ops_for_var(&self, var: usize) -> &[usize] {
        let var_index = self.var_lookup[var];
        debug_assert_ne!(
            var_index,
            std::usize::MAX,
            "Reading sentinel value in lookup!"
        );
        let start = self.var_start_points[var_index];
        if var_index + 1 == self.var_start_points.len() {
            &self.var_constant_pos[start..]
        } else {
            let end = self.var_start_points[var_index + 1];
            &self.var_constant_pos[start..end]
        }
    }

    fn push_flip(&mut self, var: usize, flip_start: usize) {
        let var_index = self.var_lookup[var];
        debug_assert_ne!(
            var_index,
            std::usize::MAX,
            "Reading sentinel value in lookup!"
        );
        let var_offset = self.var_start_points[var_index];
        let i = var_offset + flip_start;
        let start = self.var_constant_pos[i];

        if self.flip_start_seen.len() <= start {
            self.flip_start_seen.resize(start + 1, false);
        }
        if !self.flip_start_seen[start] {
            self.flip_start_seen[start] = true;
            self.bfs_var.push(var);
            self.bfs_flip_pos.push(flip_start);
        }
    }

    fn pop_flip<R: Rng>(&mut self, r: &mut R) -> Option<(usize, usize)> {
        if self.bfs_var.is_empty() {
            None
        } else {
            let pos = r.gen_range(0, self.bfs_var.len());
            let l = self.bfs_var.len() - 1;
            self.bfs_var.swap(pos, l);
            self.bfs_flip_pos.swap(pos, l);

            let var = self.bfs_var.pop().unwrap();
            let flip_pos = self.bfs_flip_pos.pop().unwrap();
            Some((var, flip_pos))
        }
    }

    fn cleanup<F: Factory<Vec<usize>> + Factory<Vec<bool>>>(self, f: &mut F) {
        f.return_instance(self.var_lookup);
        f.return_instance(self.var_start_points);
        f.return_instance(self.var_constant_pos);
        f.return_instance(self.bfs_var);
        f.return_instance(self.flip_start_seen);
    }
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
trait EdgeNavigator {
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
struct BondContainer<T: Clone + Into<usize>> {
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
    fn get_random<R: Rng>(&self, mut r: R) -> Option<&T> {
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
    fn pop_random<R: Rng>(&mut self, mut r: R) -> Option<T> {
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
    fn remove(&mut self, value: &T) -> bool {
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
    fn contains(&self, value: &T) -> bool {
        let t = value.clone().into();
        if t >= self.map.len() {
            false
        } else {
            self.map[t].is_some()
        }
    }

    /// Insert an element.
    fn insert(&mut self, value: T) -> bool {
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
    fn clear(&mut self) {
        let mut keys = self.keys.take().unwrap();
        keys.iter().for_each(|k| {
            let bond = k.clone().into();
            self.map[bond] = None
        });
        keys.clear();
        self.keys = Some(keys);
    }

    /// Get number of elements in set.
    fn len(&self) -> usize {
        self.keys.as_ref().unwrap().len()
    }

    /// Check if empty.
    fn is_empty(&self) -> bool {
        self.keys.as_ref().unwrap().is_empty()
    }

    /// Iterate through items.
    fn iter(&self) -> impl Iterator<Item = &T> {
        self.keys.as_ref().unwrap().iter()
    }
}

#[cfg(test)]
mod sc_tests {
    use super::*;
    use rand::prelude::SmallRng;
    use rand::SeedableRng;

    #[test]
    fn test_drop_doubles() {
        let mut v = vec![0, 1, 1, 2, 3, 1, 3, 4, 4, 6];
        drop_doubles(&mut v);
        println!("{:?}", v);
        assert_eq!(v, vec![0, 2, 3, 1, 3, 6])
    }

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
