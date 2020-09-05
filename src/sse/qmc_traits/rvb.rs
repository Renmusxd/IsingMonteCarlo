use crate::memory::allocator::Factory;
use crate::sse::{BondContainer, DiagonalUpdater, EdgeNavigator, Op};
use rand::Rng;

const SENTINEL_FLIP_POSITION: usize = std::usize::MAX;

/// Resonating bond update.
pub trait RVBUpdater:
    DiagonalUpdater + Factory<Vec<usize>> + Factory<Vec<bool>> + Factory<BondContainer<usize>>
{
    /// Perform a resonating bond update.
    fn rvb_update<R: Rng, EN: EdgeNavigator>(
        &mut self,
        edges: &EN,
        state: &mut [bool],
        mut rng: R,
    ) -> (usize, bool) {
        let contiguous_vars = contiguous_bits(&mut rng);
        let mut vars_list: Vec<usize> = self.get_instance();
        let mut in_cluster: Vec<bool> = self.get_instance();
        in_cluster.resize(state.len(), false);
        // Fill in_cluster and vars_list.
        let starting_var = rng.gen_range(0, state.len());
        self.get_up_to_n_contiguous_vars(
            contiguous_vars,
            starting_var,
            edges,
            &mut in_cluster,
            &mut vars_list,
            &mut rng,
        );

        // Get all the Hij=const ops on the world-lines in the cluster.
        let mut var_starts: Vec<usize> = self.get_instance();
        let mut in_cluster_constant_ps: Vec<usize> = self.get_instance();
        vars_list.iter().cloned().for_each(|v| {
            var_starts.push(in_cluster_constant_ps.len());
            self.constant_ops_on_var(v, &mut in_cluster_constant_ps);
        });

        // Get the ps for contant ops for a given var in var_list (given by index).
        let constant_slice = |rel: usize| -> &[usize] {
            let start = var_starts[rel];
            if let Some(end) = var_starts.get(rel + 1) {
                &in_cluster_constant_ps[start..*end]
            } else {
                &in_cluster_constant_ps[start..]
            }
        };

        // Choose new flip positions for each variable in cluster.
        let mut flip_positions: Vec<usize> = self.get_instance();
        (0..vars_list.len()).for_each(|relv| {
            debug_assert_eq!(flip_positions.len() % 2, 0);
            let flips = constant_slice(relv);
            let (flip_a, flip_b) = if flips.is_empty() {
                (SENTINEL_FLIP_POSITION, SENTINEL_FLIP_POSITION)
            } else {
                let flip_a = rng.gen_range(0, flips.len());
                let flip_b = rng.gen_range(0, flips.len());
                (flips[flip_a], flips[flip_b])
            };
            flip_positions.push(flip_a);
            flip_positions.push(flip_b);
        });
        self.return_instance(in_cluster_constant_ps);

        let mut all_spin_flips: Vec<usize> = self.get_instance();
        all_spin_flips.extend_from_slice(&flip_positions);

        // Get all spin flips in cluster and in neighbors.
        // These, plus the flips from above, are all the stops for regions where we need to rebuild
        // boundaries and recalculate.
        let mut in_or_around_cluster: Vec<bool> = self.get_instance();
        in_or_around_cluster.resize(state.len(), false);
        vars_list.iter().cloned().for_each(|v| {
            in_or_around_cluster[v] = true;
            self.spin_flips_on_var(v, &mut all_spin_flips);
            edges.bonds_for_var(v).iter().cloned().for_each(|bond| {
                let ov = edges.other_var_for_bond(v, bond).unwrap();
                if !in_or_around_cluster[ov] {
                    self.spin_flips_on_var(ov, &mut all_spin_flips);
                    in_or_around_cluster[ov] = true;
                };
            });
        });
        self.return_instance(in_or_around_cluster);

        // Get all spin flips in sorted order.
        // Dups can be inserted from spin-flips being selected for additional flips.
        all_spin_flips.sort_unstable();
        all_spin_flips.dedup();

        // Now get the number of ferro and antiferro bonds on the border
        let is_sat = |bond: usize, state: &[bool]| -> bool {
            let p_aligned = edges.bond_prefers_aligned(bond);
            let (a, b) = edges.vars_for_bond(bond);
            let aligned = state[a] == state[b];
            aligned == p_aligned
        };
        let mut sat_bonds: BondContainer<usize> = self.get_instance();
        let mut unsat_bonds: BondContainer<usize> = self.get_instance();

        let bounds = |relv: usize| (flip_positions[2 * relv], flip_positions[2 * relv + 1]);

        // Using a copy is good for large (beta E)/(nvars)
        let mut state_copy: Vec<bool> = self.get_instance();
        state_copy.extend_from_slice(state);
        let p_to_succ = calculate_graph_flip_prob(
            self,
            (edges, is_sat),
            (&vars_list, bounds),
            &all_spin_flips,
            (&mut sat_bonds, &mut unsat_bonds),
            (&mut state_copy, &mut in_cluster),
        );
        self.return_instance(state_copy);

        let perform_update = if p_to_succ > 1.0 {
            true
        } else if p_to_succ <= std::f64::EPSILON {
            false
        } else {
            rng.gen_bool(p_to_succ)
        };

        if perform_update {
            // sat_bonds, unsat_bonds, and in_cluster should be set up already.
            perform_rvb_update(
                self,
                (edges, is_sat),
                (&vars_list, bounds),
                &all_spin_flips,
                (&mut sat_bonds, &mut unsat_bonds),
                (state, &mut in_cluster),
                &mut rng,
            )
        }

        let ret_val = (vars_list.len(), perform_update);

        // Return things.
        self.return_instance(sat_bonds);
        self.return_instance(unsat_bonds);
        self.return_instance(all_spin_flips);
        self.return_instance(flip_positions);
        self.return_instance(var_starts);
        self.return_instance(in_cluster);
        self.return_instance(vars_list);
        ret_val
    }

    /// Get up to n contiguous variables by following bonds on edges.
    fn get_up_to_n_contiguous_vars<R: Rng, EN: EdgeNavigator>(
        &self,
        mut n: usize,
        starting_var: usize,
        edges: &EN,
        selected_vars: &mut Vec<bool>,
        vars: &mut Vec<usize>,
        mut rng: R,
    ) {
        let mut v = starting_var;
        vars.push(starting_var);
        selected_vars[v] = true;
        while n > 0 {
            let bonds = edges.bonds_for_var(v);
            if bonds.is_empty() {
                break;
            }
            let bond_index = rng.gen_range(0, bonds.len());
            let other_var = edges.other_var_for_bond(v, bonds[bond_index]).unwrap();
            if selected_vars[other_var] {
                break;
            }
            vars.push(other_var);
            selected_vars[other_var] = true;
            v = other_var;
            n -= 1;
        }
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
    let res = rvb.try_iterate_ops(t, |_, op, p, acc| {
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

fn perform_rvb_update<RVB, EN, F, G, R>(
    rvb: &mut RVB,
    (edges, is_sat): (&EN, F),
    (vars_list, bounds): (&[usize], G),
    relevant_spin_flips: &[usize],
    (sat_bonds, unsat_bonds): (&mut BondContainer<usize>, &mut BondContainer<usize>),
    (state, cluster): (&mut [bool], &mut [bool]),
    rng: &mut R,
) where
    RVB: RVBUpdater + ?Sized,
    EN: EdgeNavigator,
    F: Fn(usize, &[bool]) -> bool + Copy,
    G: Fn(usize) -> (usize, usize) + Copy,
    R: Rng,
{
    // Flip state where appropriate.
    set_cluster(vars_list, cluster, 0, bounds);
    toggle_state(vars_list, cluster, state);
    sat_bonds.clear();
    unsat_bonds.clear();
    fill_bonds(
        edges,
        is_sat,
        vars_list,
        cluster,
        state,
        unsat_bonds,
        sat_bonds,
    );

    let cutoff = rvb.get_cutoff();
    let t = (0usize, (sat_bonds, unsat_bonds), (state, cluster), rng);
    rvb.mutate_ops(cutoff, t, |_, op, p, t| {
        let (mut next_flip, bonds, state_clust, rng) = t;
        let (sat_bonds, unsat_bonds) = bonds;
        let (state, cluster) = state_clust;
        let mut rng = rng;

        let op = match relevant_spin_flips.get(next_flip) {
            Some(flip_p) if *flip_p == p => {
                // Eat one flip.
                // Technically we may need to increment more than once because there are some flip
                // positions that correspond to None ops (for vars with no constant ops). These are
                // places at SENTINEL_FLIP_POSITION which is larger than any reasonable world line,
                // we will run out of ops long before then.
                next_flip += 1;

                // Whichever op we are on is an existing spin flip or a requested one.
                let new_flip = vars_list
                    .iter()
                    .cloned()
                    .enumerate()
                    .map(|(i, v)| (v, bounds(i)))
                    .find(|(_, (start, stop))| flip_p == start || flip_p == stop);
                let new_op = if let Some((v, (start, stop))) = new_flip {
                    // This is a new flip.
                    let mut new_op = op.clone();
                    let relv = new_op.index_of_var(v).unwrap();

                    if start == p {
                        let outs = new_op.get_outputs_mut();
                        outs[relv] = !outs[relv];
                        cluster[v] = !cluster[v];
                    }
                    if stop == p {
                        let ins = new_op.get_inputs_mut();
                        ins[relv] = !ins[relv];
                        cluster[v] = !cluster[v];
                    }
                    // Note that if start == stop == p then cluster remains the same.

                    Some(new_op)
                } else {
                    // Check if this is in the cluster.
                    let all_in = op.get_vars().iter().all(|v| cluster[*v]);
                    if all_in {
                        let mut new_op = op.clone();
                        new_op.get_inputs_mut().iter_mut().for_each(|b| *b = !*b);
                        new_op.get_outputs_mut().iter_mut().for_each(|b| *b = !*b);
                        Some(new_op)
                    } else {
                        // Don't currently handle this the mixed case.
                        let all_out = !op.get_vars().iter().any(|v| cluster[*v]);
                        debug_assert!(all_out);
                        // Since fully outside, nothing to be done.
                        None
                    }
                };

                // To update state.
                let (vars, outs) = new_op
                    .as_ref()
                    .map(|op| (op.get_vars(), op.get_outputs()))
                    .unwrap_or_else(|| (op.get_vars(), op.get_outputs()));

                debug_assert!(
                    {
                        // Check inputs.
                        let ins = new_op
                            .as_ref()
                            .map(|op| op.get_inputs())
                            .unwrap_or_else(|| op.get_inputs());

                        vars.iter()
                            .cloned()
                            .zip(ins.iter().cloned())
                            .all(|(v, inp)| state[v] == inp)
                    },
                    "state != input for bond: {}",
                    op.get_bond()
                );

                vars.iter()
                    .cloned()
                    .zip(outs.iter().cloned())
                    .for_each(|(v, b)| {
                        state[v] = b;
                    });

                // Now fill up the new sat and broken bonds.
                sat_bonds.clear();
                unsat_bonds.clear();
                fill_bonds(
                    edges,
                    is_sat,
                    vars_list,
                    cluster,
                    state,
                    unsat_bonds,
                    sat_bonds,
                );

                new_op.map(Some)
            }
            _ => {
                // No interesting spin flips here, normal treatment.

                // Now, if the op has no vars in the cluster ignore it. If all are in the
                // cluster then just flip inputs and outputs, if mixed then see later.
                let (all_in, all_out) = check_borders(op, cluster);
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

                    let new_bond = if sat_bonds.contains(&bond) {
                        unsat_bonds.get_random(&mut rng).unwrap()
                    } else {
                        debug_assert!(
                            unsat_bonds.contains(&bond),
                            "Bond failed to be broken or not broken: {}",
                            bond
                        );
                        sat_bonds.get_random(&mut rng).unwrap()
                    };
                    let (new_a, new_b) = edges.vars_for_bond(*new_bond);
                    let vars = RVB::Op::make_vars([new_a, new_b].iter().cloned());
                    let state = RVB::Op::make_substate([new_a, new_b].iter().map(|v| state[*v]));
                    let new_op = RVB::Op::diagonal(vars, *new_bond, state, op.is_constant());

                    Some(Some(new_op))
                };

                new_op
            }
        };

        let t = (next_flip, (sat_bonds, unsat_bonds), (state, cluster), rng);
        (op, t)
    });
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
