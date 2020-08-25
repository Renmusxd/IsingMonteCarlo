use crate::sse::qmc_traits::*;
use rand::Rng;
#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};

/// This does an SSE equivalent of the classical dimer loop update. Requires ising symmetry and
/// a set of bonds which are equivalent but either sat or broken.
pub trait ClassicalLoopUpdater: DiagonalUpdater {
    /// Check if variable is ever flipped by an offdiagonal op.
    fn var_ever_flips(&self, var: usize) -> bool;

    /// Perform an edge update, return size of cluster and whether it was flipped.
    fn run_semiclassical_edge_update<R, EN>(
        &mut self,
        edges: &EN,
        state: &mut [bool],
        rng: R,
    ) -> (usize, bool)
    where
        EN: EdgeNavigator,
        R: Rng,
    {
        let bond_select = |s: &mut Self,
                           state: &[bool],
                           in_cluster: &mut [bool],
                           sat_set: &mut BondContainer<usize>,
                           broken_set: &mut BondContainer<usize>,
                           rng: &mut R|
         -> Option<usize> {
            let is_sat = |bond: usize| -> bool {
                let p_aligned = edges.bond_prefers_aligned(bond);
                let (a, b) = edges.vars_for_bond(bond);
                let aligned = state[a] == state[b];
                aligned == p_aligned
            };

            // Flip a single edge, flipping more is difficult while preserving detailed balance.
            let bond = rng.gen_range(0, edges.n_bonds());
            let (a, b) = edges.vars_for_bond(bond);
            let mut boundary_alloc = s.get_boundary_alloc();
            let skip_var = |var| s.var_ever_flips(var);
            boundary_alloc.extend([(a, skip_var(a)), (b, skip_var(b))].iter());
            let cluster_size = add_vars(
                edges,
                is_sat,
                skip_var,
                (sat_set, broken_set),
                in_cluster,
                &mut boundary_alloc,
            );
            s.return_boundary_alloc(boundary_alloc);
            Some(cluster_size)
        };

        self.run_semiclassical_update(edges, bond_select, state, rng)
    }

    /// Apply a classical dimer-loop update to a 2D system.
    fn run_two_d_semiclassical_loop_update<R, EN, D>(
        &mut self,
        edges: &EN,
        dual: &D,
        state: &mut [bool],
        rng: R,
    ) -> (usize, bool)
    where
        R: Rng,
        EN: EdgeNavigator,
        D: DualGraphNavigator,
    {
        let bond_select = |s: &mut Self,
                           state: &[bool],
                           in_cluster: &mut [bool],
                           sat_set: &mut BondContainer<usize>,
                           broken_set: &mut BondContainer<usize>,
                           rng: &mut R|
         -> Option<usize> {
            let mut rng = rng;
            let is_sat = |bond: usize| -> bool {
                let p_aligned = edges.bond_prefers_aligned(bond);
                let (a, b) = edges.vars_for_bond(bond);
                let aligned = state[a] == state[b];
                aligned == p_aligned
            };
            let other_face = |bond: usize, face: usize| -> usize {
                match dual.faces_sharing_bond(bond) {
                    (a, b) if a == face => b,
                    (a, b) if b == face => a,
                    _ => unreachable!(),
                }
            };

            let mut checked_bonds = s.get_checked_alloc();
            checked_bonds.resize(dual.n_faces(), false);
            // Most systems have more sat than broken, choose a broken starting bond
            let num_broken = (0..checked_bonds.len())
                .filter(|bond| is_sat(*bond))
                .count();
            let first_bond = (0..checked_bonds.len())
                .filter(|bond| is_sat(*bond))
                .nth(rng.gen_range(0, num_broken))
                .unwrap();
            let mut last_face = dual.faces_sharing_bond(first_bond).0; // Just pick the first one.
            let mut last_sat = is_sat(first_bond);

            loop {
                let bonds = dual.bonds_around_face(last_face);
                let next_bond = uniform_with_filter(
                    bonds,
                    |bond| {
                        let bond = **bond;
                        let next_bond_checked = checked_bonds[bond];
                        let sat_match = is_sat(bond) != last_sat;
                        let (a, b) = edges.vars_for_bond(bond);
                        let is_quantum = s.var_ever_flips(a) || s.var_ever_flips(b);
                        let not_visited = !next_bond_checked || bond == first_bond;
                        sat_match && is_quantum && not_visited
                    },
                    &mut rng,
                )
                .map(|choice| bonds[choice])?;

                if next_bond == first_bond {
                    // Success
                    break Some(());
                } else {
                    last_sat = !last_sat;
                    debug_assert_eq!(last_sat, is_sat(next_bond));
                    if last_sat {
                        sat_set.insert(next_bond);
                    } else {
                        broken_set.insert(next_bond);
                    }
                    last_face = other_face(next_bond, last_face);
                    checked_bonds[next_bond] = true;
                }
            }?;
            s.return_checked_alloc(checked_bonds);

            // Now we have a path from beginning to end, this defines the boundaries of our
            // clusters.
            let mut boundary = s.get_boundary_alloc();
            let mut checked_face = s.get_checked_alloc();

            checked_face.resize(dual.n_faces(), false);
            let mut cluster_size = 0;
            let interior_var = if rng.gen_bool(0.5) {
                edges.vars_for_bond(first_bond).0
            } else {
                edges.vars_for_bond(first_bond).1
            };

            // Add a variable to the cluster, add all faces to boundary to check.
            let mut add_var = |var: usize,
                               checked_face: &[bool],
                               in_cluster: &mut [bool],
                               boundary: &mut Vec<(usize, bool)>| {
                if !in_cluster[var] {
                    in_cluster[var] = true;
                    cluster_size += 1;
                    edges.bonds_for_var(var).iter().cloned().for_each(|bond| {
                        let (facea, faceb) = dual.faces_sharing_bond(bond);
                        [facea, faceb].iter().cloned().for_each(|face| {
                            if !checked_face[face] {
                                boundary.push((face, false))
                            }
                        })
                    })
                }
            };
            add_var(interior_var, &checked_face, in_cluster, &mut boundary);

            while let Some((face, _)) = boundary.pop() {
                if !checked_face[face] {
                    checked_face[face] = true;
                    let bonds = dual.bonds_around_face(face);
                    let nbonds = bonds.len();
                    let starting_relbond = (0..nbonds)
                        .find(|relbond| {
                            let (a, _) = edges.vars_for_bond(bonds[*relbond]);
                            in_cluster[a]
                        })
                        .unwrap();
                    let ok = (0..nbonds - 1)
                        .map(|relbond| (relbond + starting_relbond) % nbonds)
                        .try_for_each(|relbond| {
                            let bond = bonds[relbond];
                            let (a, b) = edges.vars_for_bond(bond);
                            let crosses_path =
                                sat_set.contains(&bond) || broken_set.contains(&bond);
                            let b_should_be = if crosses_path {
                                !in_cluster[a]
                            } else {
                                in_cluster[a]
                            };
                            if !in_cluster[b] {
                                // Add to the cluster if needed.
                                if b_should_be {
                                    add_var(b, &checked_face, in_cluster, &mut boundary)
                                }
                                Ok(())
                            } else {
                                // Need to check to make sure we didn't get a winding number issue.
                                // This can happen if the loop is closed but there's only a single
                                // side due to topology (thing straight line on torus).
                                if in_cluster[b] != b_should_be {
                                    Err(())
                                } else {
                                    Ok(())
                                }
                            }
                        });
                    // If encountered a winding number issue.
                    if ok.is_err() {
                        s.return_boundary_alloc(boundary);
                        s.return_checked_alloc(checked_face);
                        return None;
                    }
                }
            }
            s.return_boundary_alloc(boundary);
            s.return_checked_alloc(checked_face);
            Some(cluster_size)
        };

        // Check that each bond is connected to the next.
        debug_assert!((0..dual.n_faces()).all(|face| {
            let bonds = dual.bonds_around_face(face);
            (0..bonds.len()).all(|i| {
                let ba = bonds[i];
                let bb = bonds[(i + 1) % bonds.len()];
                match (edges.vars_for_bond(ba), edges.vars_for_bond(bb)) {
                    ((a, _), (c, _)) if a == c => true,
                    ((_, b), (c, _)) if b == c => true,
                    ((a, _), (_, d)) if a == d => true,
                    ((_, b), (_, d)) if b == d => true,
                    _ => false,
                }
            })
        }));
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
        ) -> Option<usize>,
    {
        let nvars = self.get_nvars();
        let mut in_cluster = self.get_var_alloc(nvars);
        let mut sat_set = self.get_sat_alloc();
        let mut broken_set = self.get_broken_alloc();

        let cluster_size = bond_select(
            self,
            state,
            &mut in_cluster,
            &mut sat_set,
            &mut broken_set,
            &mut rng,
        )
        .unwrap_or(0);

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
        if should_edit && cluster_size > 0 {
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
                        let new_op = make_replacement_op(
                            op,
                            edges,
                            &sat_set,
                            &broken_set,
                            &in_cluster,
                            state,
                            &mut rng,
                        );
                        Some(new_op)
                    } else {
                        None
                    };
                    (new_op, (state, rng))
                },
            );
        };

        self.return_sat_alloc(sat_set);
        self.return_broken_alloc(broken_set);
        self.return_var_alloc(in_cluster);
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

    /// Get the allocation
    fn get_checked_alloc(&mut self) -> Vec<bool> {
        Vec::default()
    }
    /// Return the allocation
    fn return_checked_alloc(&mut self, _bonds: Vec<bool>) {}

    /// Called after an update.
    fn post_semiclassical_update_hook(&mut self) {}
}

fn uniform_with_filter<T, F, R>(slice: &[T], f: F, mut rng: R) -> Option<usize>
where
    F: Fn(&&T) -> bool,
    R: Rng,
{
    let choices = slice.iter().filter(f).count();
    if choices == 0 {
        None
    } else {
        let choice = rng.gen_range(0, choices);
        Some(choice)
    }
}

fn check_borders<O: Op>(op: &O, in_cluster: &[bool]) -> (bool, bool) {
    op.get_vars().iter().cloned().map(|v| in_cluster[v]).fold(
        (true, true),
        |(all_true, all_false), b| {
            let all_true = all_true & b;
            let all_false = all_false & !b;
            (all_true, all_false)
        },
    )
}

/// Count ops using the iter_ops call.
pub fn count_using_iter_ops<C: ClassicalLoopUpdater + ?Sized>(
    c: &C,
    sat_set: &BondContainer<usize>,
    broken_set: &BondContainer<usize>,
) -> (usize, usize) {
    c.iterate_ops((0, 0), |_, op, acc| {
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

fn make_replacement_op<O: Op, R: Rng, EN: EdgeNavigator>(
    op: &O,
    edges: &EN,
    sat_set: &BondContainer<usize>,
    broken_set: &BondContainer<usize>,
    in_cluster: &[bool],
    state: &mut [bool],
    rng: &mut R,
) -> Option<O> {
    // Check input.
    debug_assert!(op
        .get_vars()
        .iter()
        .zip(op.get_inputs().iter())
        .all(|(v, b)| {
            if in_cluster[*v] {
                state[*v] != *b
            } else {
                state[*v] == *b
            }
        }));

    // Now, if the op has no vars in the cluster ignore it. If all are in the
    // cluster then just flip inputs and outputs, if mixed then see later.
    let (all_in, all_out) = check_borders(op, in_cluster);
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
        Some(new_op)
    } else {
        // We are only covering 2-variable edges.
        debug_assert_eq!(op.get_vars().len(), 2);
        // We only move diagonal ops.
        debug_assert!(op.is_diagonal());
        let bond = op.get_bond();
        let new_bond = if sat_set.contains(&bond) {
            broken_set.get_random(rng).unwrap()
        } else {
            debug_assert!(
                broken_set.contains(&bond),
                "Bond failed to be broken or not broken: {}",
                bond
            );
            sat_set.get_random(rng).unwrap()
        };
        let (new_a, new_b) = edges.vars_for_bond(*new_bond);
        let vars = O::make_vars([new_a, new_b].iter().cloned());
        let state = O::make_substate([new_a, new_b].iter().map(|v| state[*v]));
        let new_op = O::diagonal(vars, *new_bond, state, op.is_constant());

        Some(new_op)
    }
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

/// The dual graph is defined by the set of faces as vertices, and edges between them if they share
/// a bond. For 2D systems where each bond neighbors exactly 2 faces.
pub trait DualGraphNavigator {
    /// Number of faces in dual.
    fn n_faces(&self) -> usize;
    /// Get the two faces sharing a bond.
    fn faces_sharing_bond(&self, bond: usize) -> (usize, usize);
    /// List of bonds around a given face.
    fn bonds_around_face(&self, face: usize) -> &[usize];
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
