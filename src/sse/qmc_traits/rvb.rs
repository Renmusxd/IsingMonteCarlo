use crate::sse::*;
use crate::util::allocator::Factory;
use crate::util::bondcontainer::BondContainer;
use rand::Rng;

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
    /// Get the magnitude of the largest entry for the bond matrix
    fn bond_mag(&self, b: usize) -> f64;
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

/// Resonating bond update.
pub trait RvbUpdater:
    DiagonalSubsection
    + Factory<Vec<usize>>
    + Factory<Vec<bool>>
    + Factory<BondContainer<usize>>
    + Factory<BondContainer<VarPos>>
    + Factory<Vec<Option<usize>>>
{
    /// Fill `ps` with the p values of constant (Hij=k) ops for a given var.
    /// An implementation by the same name is provided for LoopUpdaters
    fn constant_ops_on_var(&self, var: usize, ps: &mut Vec<usize>);

    /// Fill `ps` with the p values of spin flips for a given var.
    /// An implementation by the same name is provided for LoopUpdaters
    fn spin_flips_on_var(&self, var: usize, ps: &mut Vec<usize>);

    // TODO add some check for offdiagonal 2-site ops on border.

    /// Perform a resonating bond update.
    /// `edges` lists all the 2-site bonds which define the lattice.
    /// `state` is the propagated state at time 0.
    /// `updates` is the number of updated to perform.
    /// `diagonal_edge_hamiltonian` gives the weight of a diagonal edge given a spin state.
    /// `rng` prng instance to use.
    fn rvb_update<R: Rng, EN: EdgeNavigator, H>(
        &mut self,
        edges: &EN,
        state: &mut [bool],
        updates: usize,
        diagonal_edge_hamiltonian: H,
        rng: &mut R,
    ) -> usize
    where
        H: Fn(usize, bool, bool) -> f64,
    {
        self.rvb_update_with_ising_weight(
            edges,
            state,
            updates,
            diagonal_edge_hamiltonian,
            |_| 1.0,
            rng,
        )
    }

    /// Perform a resonating bond update.
    /// `edges` lists all the 2-site bonds which define the lattice.
    /// `state` is the propagated state at time 0.
    /// `updates` is the number of updated to perform.
    /// `diagonal_edge_hamiltonian` gives the weight of a diagonal edge given a spin state.
    /// `ising_ratio` provides the weight ratio of a node after a global ising flip to current.
    /// `rng` prng instance to use.
    fn rvb_update_with_ising_weight<R: Rng, EN: EdgeNavigator, H, F>(
        &mut self,
        edges: &EN,
        state: &mut [bool],
        updates: usize,
        diagonal_edge_hamiltonian: H,
        ising_ratio: F,
        rng: &mut R,
    ) -> usize
    where
        H: Fn(usize, bool, bool) -> f64,
        F: Fn(&Self::Op) -> f64,
    {
        let mut var_starts: Vec<usize> = self.get_instance();
        let mut var_lengths: Vec<usize> = self.get_instance();
        let mut constant_ps: Vec<usize> = self.get_instance();
        // This helps us sample evenly.
        let mut vars_with_zero_ops: Vec<usize> = self.get_instance();

        // O(beta * n)
        find_constants(
            self,
            &mut var_starts,
            &mut var_lengths,
            &mut constant_ps,
            &mut vars_with_zero_ops,
        );
        let mut num_succ = 0;
        for _ in 0..updates {
            // Pick starting flip.
            let choice = rng.gen_range(0..(constant_ps.len() + vars_with_zero_ops.len()));
            let (v, flip) = if choice < constant_ps.len() {
                let res = var_starts.binary_search(&choice);
                let v = match res {
                    Err(i) => i - 1,
                    Ok(mut i) => {
                        // Get the last i with this value.
                        while i + 1 < var_starts.len() && var_starts[i + 1] == var_starts[i] {
                            i = i + 1
                        }
                        i
                    }
                };
                (v, Some(choice))
            } else {
                let choice = choice - constant_ps.len();
                (vars_with_zero_ops[choice], None)
            };

            let mut cluster_vars: Vec<usize> = self.get_instance();
            let mut cluster_flips: Vec<Option<usize>> = self.get_instance();

            let cluster_size = contiguous_bits(rng) + 1;

            let mut cbm = WeightedBoundaryManager::new_from_factory(self);
            build_cluster(
                cluster_size,
                (v, flip),
                self.get_cutoff(),
                (&mut cluster_vars, &mut cluster_flips),
                (&var_starts, &var_lengths),
                &constant_ps,
                edges,
                &mut cbm,
                rng,
            );

            let mut boundary_vars: Vec<usize> = self.get_instance();
            let mut boundary_flips_pos: Vec<Option<usize>> = self.get_instance();
            cbm.dissolve_into(self, &mut boundary_vars, &mut boundary_flips_pos);

            let mut cluster_starting_state: Vec<bool> = self.get_instance();
            let mut cluster_toggle_ps: Vec<usize> = self.get_instance();
            let mut subvars: Vec<usize> = self.get_instance();
            let mut var_to_subvar: Vec<Option<usize>> = self.get_instance();
            var_to_subvar.resize(self.get_nvars(), None);

            subvars.extend(cluster_vars.iter().chain(boundary_vars.iter()).cloned());
            subvars.sort_unstable();
            subvars.dedup();
            subvars
                .iter()
                .cloned()
                .enumerate()
                .for_each(|(subvar, var)| var_to_subvar[var] = Some(subvar));

            cluster_starting_state.resize(subvars.len(), false);
            cluster_vars
                .iter()
                .cloned()
                .zip(cluster_flips.iter().cloned())
                .for_each(|(v, fi)| {
                    let subvar = var_to_subvar[v].unwrap();
                    if let Some(fi) = fi {
                        let vstart = var_starts[v];
                        let fi_rel = fi - vstart;

                        if fi_rel + 1 >= var_lengths[v] {
                            cluster_starting_state[subvar] = true;
                            cluster_toggle_ps.push(constant_ps[fi]);
                            cluster_toggle_ps.push(constant_ps[vstart]);
                        } else {
                            cluster_toggle_ps.push(constant_ps[fi]);
                            cluster_toggle_ps.push(constant_ps[fi + 1]);
                        }
                    } else {
                        cluster_starting_state[subvar] = true;
                    }
                });
            self.return_instance(cluster_flips);
            self.return_instance(cluster_vars);

            // Should be able to infer substate based solely on boundary + cluster.
            // First get all vars in boundary or in cluster.
            let mut substate: Vec<bool> = self.get_instance();
            substate.extend(subvars.iter().cloned().map(|v| state[v]));

            // Lets find the tops of each boundary.
            let mut subvar_boundary_tops: Vec<Option<usize>> = self.get_instance();
            subvar_boundary_tops.resize(subvars.len(), None);
            boundary_vars
                .iter()
                .zip(boundary_flips_pos.iter())
                .for_each(|(bv, bfp)| {
                    let subvar =
                        var_to_subvar[*bv].expect("Boundary must be in var_to_subvar array.");
                    match (bfp, subvar_boundary_tops[subvar]) {
                        (Some(bfp), Some(bt)) => {
                            if *bfp < bt {
                                subvar_boundary_tops[subvar] = Some(constant_ps[*bfp])
                            }
                        }
                        (Some(_), None) | (None, None) => {
                            subvar_boundary_tops[subvar] = bfp.map(|bfp| constant_ps[bfp])
                        }
                        (None, Some(_)) => unreachable!(),
                    };
                });
            self.return_instance(boundary_vars);
            self.return_instance(boundary_flips_pos);

            // Now lets get the cluster boundaries.
            cluster_toggle_ps.sort_unstable();
            remove_doubles(&mut cluster_toggle_ps);

            let p_to_flip = calculate_flip_prob(
                self,
                (state, &mut substate),
                (&mut cluster_starting_state, &cluster_toggle_ps),
                &subvar_boundary_tops,
                (&subvars, |v| var_to_subvar[v]),
                (&diagonal_edge_hamiltonian, &ising_ratio, edges),
            );
            let should_mutate = if p_to_flip >= 1.0 {
                true
            } else {
                if p_to_flip < 0. {
                    println!("P = {}", p_to_flip);
                    debug_print_diagonal(self, state);
                }
                rng.gen_bool(p_to_flip)
            };

            if should_mutate {
                // Great, mutate the graph.
                mutate_graph(
                    self,
                    (state, &mut substate),
                    (&mut cluster_starting_state, &cluster_toggle_ps),
                    &subvar_boundary_tops,
                    (&subvars, |v| var_to_subvar[v]),
                    (&diagonal_edge_hamiltonian, edges),
                    rng,
                );
                let starting_cluster = cluster_starting_state
                    .iter()
                    .cloned()
                    .filter(|b| *b)
                    .count()
                    > 0;

                if starting_cluster {
                    subvars
                        .iter()
                        .cloned()
                        .zip(cluster_starting_state.iter().cloned())
                        .for_each(|(v, c)| {
                            state[v] = state[v] != c;
                        });
                }
                num_succ += 1;
            }
            self.return_instance(var_to_subvar);
            self.return_instance(subvar_boundary_tops);
            self.return_instance(substate);
            self.return_instance(subvars);
            self.return_instance(cluster_toggle_ps);
            self.return_instance(cluster_starting_state);
        }

        self.return_instance(vars_with_zero_ops);
        self.return_instance(constant_ps);
        self.return_instance(var_lengths);
        self.return_instance(var_starts);
        num_succ
    }
}

/// Returns true if substate is changed.
fn mutate_graph<RVB: RvbUpdater + ?Sized, VS, EN: EdgeNavigator + ?Sized, R: Rng, H>(
    rvb: &mut RVB,
    (state, substate): (&[bool], &mut [bool]),
    (cluster_state, cluster_flips): (&mut [bool], &[usize]),
    boundary_tops: &[Option<usize>], // top for each of vars.
    (vars, var_to_subvar): (&[usize], VS),
    (diagonal_hamiltonian, edges): (H, &EN),
    rng: &mut R,
) where
    VS: Fn(usize) -> Option<usize>,
    H: Fn(usize, bool, bool) -> f64,
{
    // Find all spots where cluster is empty.
    let mut jump_to: Vec<usize> = rvb.get_instance();
    let mut continue_until: Vec<usize> = rvb.get_instance();

    let mut count = cluster_state.iter().filter(|b| **b).count();
    // If cluster hits t=0 then we need to mutate right away.
    let has_starting_cluster = if count != 0 {
        jump_to.push(0);
        // We need to adjust substate since there's a starting cluster.
        substate
            .iter_mut()
            .zip(cluster_state.iter().cloned())
            .for_each(|(b, c)| *b = *b != c);

        true
    } else {
        false
    };
    cluster_flips.iter().cloned().for_each(|p| {
        // If count is currently 0, this p will change that. We will need to
        // jump here for mutations.
        if count == 0 {
            jump_to.push(p)
        }

        let op = rvb.get_node_ref(p).unwrap().get_op_ref();
        count = op
            .get_vars()
            .iter()
            .cloned()
            .filter_map(|v| var_to_subvar(v))
            .fold(count, |count, subvar| {
                cluster_state[subvar] = !cluster_state[subvar];
                if cluster_state[subvar] {
                    count + 1
                } else {
                    count - 1
                }
            });

        // If this op set the count to zero
        if count == 0 {
            continue_until.push(p)
        }
    });
    // If we end with a count that means we will start with one too, go all the way to the end.
    if count != 0 {
        debug_assert!(has_starting_cluster);
        continue_until.push(rvb.get_cutoff());
    }
    debug_assert_eq!(
        jump_to.len(),
        continue_until.len(),
        "Should have the same number of starts and ends."
    );

    // Now we have a series of pairs.
    let mut bonds: BondContainer<usize> = rvb.get_instance();
    let mut next_cluster_index = 0;

    vars.iter()
        .cloned()
        .filter(|v| cluster_state[var_to_subvar(*v).unwrap()])
        .for_each(|v| {
            edges.bonds_for_var(v).iter().cloned().for_each(|b| {
                let ov = edges.other_var_for_bond(v, b).unwrap();
                if !cluster_state[var_to_subvar(ov).unwrap()] {
                    let (va, vb) = edges.vars_for_bond(b);
                    let subva = var_to_subvar(va).unwrap();
                    let subvb = var_to_subvar(vb).unwrap();
                    let w = diagonal_hamiltonian(b, substate[subva], substate[subvb]);
                    bonds.insert(b, w);
                }
            })
        });

    jump_to
        .iter()
        .cloned()
        .zip(continue_until.iter().cloned())
        .fold(
            (substate, cluster_state, rng),
            |(substate, cluster_state, rng), (from, until)| {
                rvb.get_propagated_substate_with_hint(
                    from,
                    substate,
                    state,
                    vars,
                    boundary_tops.iter().cloned(),
                );
                substate
                    .iter_mut()
                    .zip(cluster_state.iter().cloned())
                    .for_each(|(b, c)| *b = *b != c);

                let mut args = rvb.get_empty_args(SubvarAccess::Varlist(&vars));
                rvb.fill_args_at_p_with_hint(from, &mut args, vars, boundary_tops.iter().cloned());

                let acc = (next_cluster_index, &mut bonds, substate, cluster_state, rng);
                let ret = rvb.mutate_subsection_ops(
                    from,
                    until,
                    acc,
                    |_, op, p, acc| {
                        let (mut next_cluster_index, bonds, substate, cluster_state, mut rng) = acc;
                        let in_bonds = bonds.contains(&op.get_bond());
                        let at_next_cluster_flip = next_cluster_index < cluster_flips.len()
                            && p == cluster_flips[next_cluster_index];
                        let newop = if in_bonds {
                            // Rotatable ops must be diagonal.
                            debug_assert!(op.is_diagonal());

                            // Need to rotate
                            let new_bond = bonds.get_random(&mut rng).unwrap().0;

                            let (new_a, new_b) = edges.vars_for_bond(new_bond);
                            let vars = RVB::Op::make_vars([new_a, new_b].iter().cloned());
                            let state = RVB::Op::make_substate(
                                [new_a, new_b]
                                    .iter()
                                    .cloned()
                                    .map(|v| var_to_subvar(v).unwrap())
                                    .map(|subvar| substate[subvar]),
                            );
                            let new_op = RVB::Op::diagonal(vars, new_bond, state, op.is_constant());

                            Some(Some(new_op))
                        } else {
                            let new_op = if at_next_cluster_flip {
                                // We are at a cluster boundary, flips cluster state and bonds.
                                debug_assert!(op.is_constant());
                                // Cluster flips must be entirely in subvar region.
                                debug_assert!(op
                                    .get_vars()
                                    .iter()
                                    .cloned()
                                    .all(|v| var_to_subvar(v).is_some()));

                                let new_op = op.clone_and_edit_in_out(|ins, outs| {
                                    op.get_vars()
                                        .iter()
                                        .cloned()
                                        .map(|v| var_to_subvar(v).unwrap())
                                        .zip(ins.iter_mut().zip(outs.iter_mut()))
                                        .for_each(|(subvar, (bin, bout))| {
                                            // Flip if cluster_state is true.
                                            *bin = *bin != cluster_state[subvar];
                                            // Flip if cluster_state _will be_ true.
                                            *bout = *bout != !cluster_state[subvar];
                                        });
                                });
                                new_op
                                    .get_vars()
                                    .iter()
                                    .cloned()
                                    .map(|v| var_to_subvar(v).unwrap())
                                    .zip(new_op.get_outputs().iter().cloned())
                                    .for_each(|(subvar, bout)| {
                                        cluster_state[subvar] = !cluster_state[subvar];
                                        substate[subvar] = bout
                                    });

                                next_cluster_index += 1;
                                Some(Some(new_op))
                            } else {
                                // Flip appropriate inputs/outputs.
                                // If any are out, then all are out - otherwise would be in sat or unsat
                                debug_assert!({
                                    let all_in = op.get_vars().iter().cloned().all(|v| {
                                        var_to_subvar(v)
                                            .map(|subvar| cluster_state[subvar])
                                            .unwrap_or(false)
                                    });
                                    let all_out = op.get_vars().iter().cloned().all(|v| {
                                        var_to_subvar(v)
                                            .map(|subvar| !cluster_state[subvar])
                                            .unwrap_or(true)
                                    });
                                    let succ = (all_in != all_out) && (all_in || all_out);
                                    if !succ {
                                        println!("subvars: {:?}", vars);
                                        println!(
                                            "op: {:?}\t{:?} -> {:?}",
                                            op.get_vars(),
                                            op.get_inputs(),
                                            op.get_outputs()
                                        );
                                        println!("all_in: {}\tall_out: {}", all_in, all_out);
                                    }
                                    succ
                                });

                                let any_subvars = op
                                    .get_vars()
                                    .iter()
                                    .cloned()
                                    .any(|v| var_to_subvar(v).is_some());
                                let any_in_cluster = op
                                    .get_vars()
                                    .iter()
                                    .cloned()
                                    .filter_map(&var_to_subvar)
                                    .any(|subvar| cluster_state[subvar]);
                                // If out of known region or if diagonal and not flipped by cluster.
                                if !any_subvars || (!any_in_cluster && op.is_diagonal()) {
                                    None
                                } else {
                                    let new_op = if any_in_cluster {
                                        debug_assert!(op
                                            .get_vars()
                                            .iter()
                                            .cloned()
                                            .all(|v| var_to_subvar(v).is_some()));

                                        let new_op = op.clone_and_edit_in_out_symmetric(|state| {
                                            state.iter_mut().for_each(|b| *b = !*b);
                                        });

                                        if !new_op.is_diagonal() {
                                            // Update state
                                            new_op
                                                .get_vars()
                                                .iter()
                                                .cloned()
                                                .map(|v| var_to_subvar(v).unwrap())
                                                .zip(new_op.get_outputs().iter().cloned())
                                                .for_each(|(subvar, bout)| {
                                                    substate[subvar] = bout;
                                                });
                                        }

                                        Some(Some(new_op))
                                    } else {
                                        if !op.is_diagonal() {
                                            // Update state
                                            op.get_vars()
                                                .iter()
                                                .cloned()
                                                .filter_map(|v| var_to_subvar(v))
                                                .zip(op.get_outputs().iter().cloned())
                                                .for_each(|(subvar, bout)| {
                                                    substate[subvar] = bout;
                                                });
                                        }
                                        None
                                    };

                                    new_op
                                }
                            };

                            // Now update bonds
                            op.get_vars()
                                .iter()
                                .cloned()
                                .filter_map(|v| var_to_subvar(v).map(|subvar| (v, subvar)))
                                .for_each(|(v, subvar)| {
                                    edges
                                        .bonds_for_var(v)
                                        .iter()
                                        .cloned()
                                        .filter_map(|b| {
                                            let ov = edges.other_var_for_bond(v, b).unwrap();
                                            var_to_subvar(ov).map(|o_subvar| (b, subvar, o_subvar))
                                        })
                                        .for_each(|(b, subvar, o_subvar)| {
                                            if cluster_state[subvar] == cluster_state[o_subvar] {
                                                // Remove bond from borders.
                                                if bonds.contains(&b) {
                                                    bonds.remove(&b);
                                                }
                                            } else {
                                                let (va, vb) = edges.vars_for_bond(b);
                                                let subva = var_to_subvar(va).unwrap();
                                                let subvb = var_to_subvar(vb).unwrap();
                                                let w = diagonal_hamiltonian(
                                                    b,
                                                    substate[subva],
                                                    substate[subvb],
                                                );
                                                bonds.insert(b, w);
                                            }
                                        })
                                });

                            new_op
                        };

                        (
                            newop,
                            (next_cluster_index, bonds, substate, cluster_state, rng),
                        )
                    },
                    Some(args),
                );
                next_cluster_index = ret.0;
                let substate = ret.2;
                let cluster_state = ret.3;
                let rng = ret.4;
                (substate, cluster_state, rng)
            },
        );
    rvb.return_instance(bonds);

    rvb.return_instance(jump_to);
    rvb.return_instance(continue_until);
}

fn calculate_flip_prob<RVB: RvbUpdater + ?Sized, VS, EN: EdgeNavigator + ?Sized, H, F>(
    rvb: &mut RVB,
    (state, substate): (&[bool], &mut [bool]),
    (cluster_state, cluster_flips): (&mut [bool], &[usize]),
    boundary_tops: &[Option<usize>], // top for each of vars.
    (vars, var_to_subvar): (&[usize], VS),
    (diagonal_hamiltonian, ising_ratio, edges): (H, F, &EN),
) -> f64
where
    VS: Fn(usize) -> Option<usize>,
    H: Fn(usize, bool, bool) -> f64,
    F: Fn(&RVB::Op) -> f64,
{
    let mut cluster_size = cluster_state.iter().cloned().filter(|x| *x).count();
    let mut psel = rvb.get_first_p();
    let mut next_cluster_index = 0;
    let mut mult = 1.0;

    let ws_for_flip = |b: usize, subvar_to_flip: usize, substate: &[bool]| {
        let (va, vb) = edges.vars_for_bond(b);
        let suba = var_to_subvar(va).unwrap();
        let subb = var_to_subvar(vb).unwrap();
        debug_assert!(subvar_to_flip == suba || subvar_to_flip == subb);

        let ba = substate[suba];
        let bb = substate[subb];
        let w_before = diagonal_hamiltonian(b, ba, bb);

        let (ba, bb) = if subvar_to_flip == suba {
            (!ba, bb)
        } else {
            (ba, !bb)
        };

        let w_after = diagonal_hamiltonian(b, ba, bb);
        (w_before, w_after)
    };

    let mut bonds_before: BondContainer<usize> = rvb.get_instance();
    let mut bonds_after: BondContainer<usize> = rvb.get_instance();
    let mut n_bonds = 0;
    if cluster_size != 0 {
        vars.iter()
            .cloned()
            .map(|v| (v, var_to_subvar(v).unwrap()))
            .filter(|(_, subvar)| cluster_state[*subvar])
            .for_each(|(v, subvar)| {
                edges.bonds_for_var(v).iter().cloned().for_each(|b| {
                    let ov = edges.other_var_for_bond(v, b).unwrap();
                    let o_subvar = var_to_subvar(ov).unwrap();
                    if !cluster_state[o_subvar] {
                        let (wbef, waft) = ws_for_flip(b, subvar, substate);
                        bonds_before.insert(b, wbef);
                        bonds_after.insert(b, waft);
                    }
                })
            });
    }

    // Jump to cluster start, requires propagating state. Since not inside a cluster it's
    // safe to use just the boundary.
    while let Some(mut p) = psel {
        // Skip ahead.
        if cluster_size == 0 {
            debug_assert_eq!(bonds_before.len(), 0);
            debug_assert_eq!(bonds_after.len(), 0);
            debug_assert_eq!(n_bonds, 0);
            // Jump to next nonzero spot.
            if next_cluster_index < cluster_flips.len() {
                // psel can be set but won't be read.
                // psel = Some(p);
                p = cluster_flips[next_cluster_index];

                rvb.get_propagated_substate_with_hint(
                    p,
                    substate,
                    state,
                    vars,
                    boundary_tops.iter().cloned(),
                );
            } else {
                // Done with clusters, jump to the end.
                break;
            }
        }

        let node = rvb.get_node_ref(p).unwrap();
        let op = node.get_op_ref();

        // Check if op is anywhere near the cluster.
        let near_cluster = op
            .get_vars()
            .iter()
            .cloned()
            .any(|v| var_to_subvar(v).is_some());
        if near_cluster {
            let is_cluster_bound =
                next_cluster_index < cluster_flips.len() && p == cluster_flips[next_cluster_index];
            let will_flip_spins = !op.is_diagonal();
            let will_change_bonds = will_flip_spins || is_cluster_bound;
            let completely_in_cluster = op.get_vars().iter().cloned().all(|v| {
                var_to_subvar(v)
                    .map(|subvar| cluster_state[subvar])
                    .unwrap_or(false)
            });

            // Count which bond it belongs to.
            let b = op.get_bond();
            if bonds_before.contains(&b) {
                debug_assert!(!is_cluster_bound);
                debug_assert!(!will_flip_spins);
                debug_assert!(bonds_after.contains(&b));
                // TODO for offdiagonal edges just look at weight difference, no rotation.
                n_bonds += 1;
            } else {
                // We are at a cluster boundary, flips cluster state and bonds.
                if is_cluster_bound {
                    debug_assert!(op.is_constant());
                    debug_assert_eq!(op.get_vars().len(), 1);

                    let v = op.get_vars()[0];
                    let subvar = var_to_subvar(v).unwrap();

                    cluster_state[subvar] = !cluster_state[subvar];
                    if cluster_state[subvar] {
                        cluster_size += 1
                    } else {
                        cluster_size -= 1
                    };
                    next_cluster_index += 1;
                }

                if will_flip_spins {
                    op.get_vars()
                        .iter()
                        .cloned()
                        .zip(op.get_outputs().iter().cloned())
                        .filter_map(|(v, bout)| {
                            var_to_subvar(v)
                                .zip(Some(bout))
                                .map(|(subvar, bout)| (subvar, bout))
                        })
                        .for_each(|(subvar, bout)| {
                            substate[subvar] = bout;
                        });
                }

                if completely_in_cluster {
                    let ising_flip_weight = ising_ratio(op);
                    mult *= ising_flip_weight;
                    if mult < std::f64::EPSILON {
                        break;
                    }
                }

                if will_change_bonds {
                    // Commit the counts so far.
                    mult *= calculate_mult(&bonds_before, &bonds_after, n_bonds);
                    n_bonds = 0;
                    // Break early if we reach 0 probability.
                    if mult < std::f64::EPSILON {
                        break;
                    }

                    // Now update bonds
                    op.get_vars()
                        .iter()
                        .cloned()
                        .filter_map(|v| var_to_subvar(v).map(|subvar| (v, subvar)))
                        .for_each(|(v, subvar)| {
                            edges
                                .bonds_for_var(v)
                                .iter()
                                .cloned()
                                .filter_map(|b| {
                                    let ov = edges.other_var_for_bond(v, b).unwrap();
                                    var_to_subvar(ov).map(|o_subvar| (b, subvar, o_subvar))
                                })
                                .for_each(|(b, subvar, o_subvar)| {
                                    if cluster_state[subvar] == cluster_state[o_subvar] {
                                        // Remove bond from borders.
                                        if bonds_before.contains(&b) {
                                            bonds_before.remove(&b);
                                            bonds_after.remove(&b);
                                        }
                                    } else {
                                        let subvar = if cluster_state[subvar] {
                                            subvar
                                        } else {
                                            o_subvar
                                        };
                                        let (wbef, waft) = ws_for_flip(b, subvar, substate);
                                        // Insert or update weight.
                                        bonds_before.insert(b, wbef);
                                        bonds_after.insert(b, waft);
                                    }
                                })
                        });
                }
            }
        }

        // Move on
        psel = rvb.get_next_p(node);
    }
    // Commit remaining stuff.
    mult *= calculate_mult(&bonds_before, &bonds_after, n_bonds);

    rvb.return_instance(bonds_before);
    rvb.return_instance(bonds_after);

    mult
}

fn remove_doubles<T: Eq + Copy>(v: &mut Vec<T>) {
    let mut ii = 0;
    let mut jj = 0;
    while jj + 1 < v.len() {
        if v[jj] == v[jj + 1] {
            jj += 2;
        } else {
            v[ii] = v[jj];
            ii += 1;
            jj += 1;
        }
    }
    if jj < v.len() {
        v[ii] = v[jj];
        ii += 1;
        jj += 1;
    }
    while jj > ii {
        v.pop();
        jj -= 1;
    }
}

trait ClusterBoundaryManager {
    fn pop_index<R: Rng>(&mut self, rng: &mut R) -> (usize, Option<usize>, f64);
    fn push_adjacent(&mut self, var: usize, pos: Option<usize>, weight: Option<f64>);
    fn is_empty(&self) -> bool;
}

/// Variable / P positions pairs.
#[derive(Default, Debug, Clone, Copy)]
pub struct VarPos {
    v: usize,
    p: Option<usize>,
}

impl From<VarPos> for usize {
    fn from(vp: VarPos) -> usize {
        vp.p.unwrap_or(vp.v)
    }
}

struct WeightedBoundaryManager {
    boundary_flips: BondContainer<VarPos>,
    boundary_noflips: BondContainer<VarPos>,
    var_pos_popped: Vec<bool>,
    var_nopos_popped: Vec<bool>,
}

impl WeightedBoundaryManager {
    fn new_from_factory<F>(f: &mut F) -> Self
    where
        F: Factory<BondContainer<VarPos>> + Factory<Vec<bool>> + ?Sized,
    {
        Self {
            boundary_flips: f.get_instance(),
            boundary_noflips: f.get_instance(),
            var_pos_popped: f.get_instance(),
            var_nopos_popped: f.get_instance(),
        }
    }

    fn dissolve_into<F>(
        self,
        f: &mut F,
        boundary_vars: &mut Vec<usize>,
        boundary_flips: &mut Vec<Option<usize>>,
    ) where
        F: Factory<BondContainer<VarPos>> + Factory<Vec<bool>> + ?Sized,
    {
        self.boundary_flips
            .iter()
            .chain(self.boundary_noflips.iter())
            .for_each(|(varpos, _)| {
                boundary_vars.push(varpos.v);
                boundary_flips.push(varpos.p);
            });
        f.return_instance(self.boundary_flips);
        f.return_instance(self.boundary_noflips);
        f.return_instance(self.var_pos_popped);
        f.return_instance(self.var_nopos_popped);
    }
}

impl ClusterBoundaryManager for WeightedBoundaryManager {
    fn pop_index<R: Rng>(&mut self, rng: &mut R) -> (usize, Option<usize>, f64) {
        let total_weight =
            self.boundary_flips.get_total_weight() + self.boundary_noflips.get_total_weight();
        let f_ratio = self.boundary_flips.get_total_weight() / total_weight;
        let pick_flips = rng.gen_bool(f_ratio);
        let (boundary, poss) = if pick_flips {
            (&mut self.boundary_flips, &mut self.var_pos_popped)
        } else {
            (&mut self.boundary_noflips, &mut self.var_nopos_popped)
        };
        let (v, w) = *boundary.get_random(rng).unwrap();
        let indx: usize = v.clone().into();
        poss[indx] = true;
        boundary.remove(&v);

        (v.v, v.p, w)
    }

    fn push_adjacent(&mut self, var: usize, pos: Option<usize>, weight: Option<f64>) {
        let weight = weight.unwrap_or(1.0);
        let (boundary, poss) = if pos.is_some() {
            (&mut self.boundary_flips, &mut self.var_pos_popped)
        } else {
            (&mut self.boundary_noflips, &mut self.var_nopos_popped)
        };

        let varpos = VarPos { v: var, p: pos };

        // If this hasn't already been popped.
        let indx: usize = varpos.clone().into();
        if indx >= poss.len() {
            poss.resize(indx + 1, false);
        }
        if !poss[indx] {
            let weight = boundary.get_weight(&varpos).unwrap_or(0.) + weight;
            boundary.insert(varpos, weight);
        }
    }

    fn is_empty(&self) -> bool {
        self.boundary_flips.is_empty() && self.boundary_noflips.is_empty()
    }
}

fn build_cluster<EN, CBM, R>(
    mut cluster_size: usize,
    (init_var, init_flip): (usize, Option<usize>),
    cutoff: usize,
    (cluster_vars, cluster_flips): (&mut Vec<usize>, &mut Vec<Option<usize>>),
    (var_starts, var_lengths): (&[usize], &[usize]),
    constant_ps: &[usize],
    edges: &EN,
    cbm: &mut CBM,
    rng: &mut R,
) where
    EN: EdgeNavigator,
    CBM: ClusterBoundaryManager,
    R: Rng,
{
    cbm.push_adjacent(init_var, init_flip, None);

    while cluster_size > 0 && !cbm.is_empty() {
        let (v, flip, _) = cbm.pop_index(rng);

        // Check that popped values make sense.
        debug_assert!(flip.map(|f| f >= var_starts[v]).unwrap_or(true));
        debug_assert!(flip
            .map(|f| f < var_starts[v] + var_lengths[v])
            .unwrap_or(var_lengths[v] == 0));

        cluster_vars.push(v);
        cluster_flips.push(flip);

        // Add above and below.
        if let Some(flip) = flip {
            let relflip = flip - var_starts[v];
            let flip_dec = (relflip + var_lengths[v] - 1) % var_lengths[v] + var_starts[v];
            let flip_inc = (relflip + 1) % var_lengths[v] + var_starts[v];

            cbm.push_adjacent(v, Some(flip_dec), None);
            cbm.push_adjacent(v, Some(flip_inc), None);
        }

        // Add neighbors to what we just added.
        edges.bonds_for_var(v).iter().for_each(|b| {
            let weight = edges.bond_mag(*b);
            let ov = edges.other_var_for_bond(v, *b).unwrap();
            if var_lengths[ov] == 0 {
                cbm.push_adjacent(ov, None, Some(weight));
            } else {
                let pis = var_starts[ov]..var_starts[ov] + var_lengths[ov];
                if let Some(flip) = flip {
                    let relflip = flip - var_starts[v];
                    let flip_inc = (relflip + 1) % var_lengths[v] + var_starts[v];

                    let pstart = constant_ps[flip];
                    let pend = constant_ps[flip_inc];
                    find_overlapping_starts(pstart, pend, cutoff, &constant_ps[pis])
                        .map(|i| i + var_starts[ov])
                        .for_each(|flip_pos| {
                            cbm.push_adjacent(ov, Some(flip_pos), Some(weight));
                        })
                } else {
                    // Add them all.
                    pis.for_each(|pi| {
                        cbm.push_adjacent(ov, Some(pi), Some(weight));
                    })
                }
            }
        });

        cluster_size -= 1;
    }
}

fn find_overlapping_starts(
    p_start: usize,
    p_end: usize,
    cutoff: usize,
    flip_positions: &[usize],
) -> impl Iterator<Item = usize> + '_ {
    debug_assert_ne!(flip_positions.len(), 0);
    let bin_found = flip_positions.binary_search(&p_start).unwrap_err();
    let prev_bin_found = (bin_found + flip_positions.len() - 1) % flip_positions.len();
    let lowest_ps = flip_positions[prev_bin_found];
    let offset_p_start = (p_start + cutoff - lowest_ps) % cutoff;
    let offset_p_end = (p_end + cutoff - lowest_ps) % cutoff;
    flip_positions[prev_bin_found..]
        .iter()
        .cloned()
        .zip(prev_bin_found..)
        .chain(
            flip_positions[..prev_bin_found]
                .iter()
                .cloned()
                .zip(0..prev_bin_found),
        )
        .take_while(move |(p, ip)| {
            let check_start = (*p + cutoff - lowest_ps) % cutoff;
            let next_p = flip_positions[(*ip + 1) % flip_positions.len()];
            let check_end = (next_p + cutoff - lowest_ps) % cutoff;
            let has_overlap_start = check_start < offset_p_start && offset_p_start < check_end;
            let has_start_within = offset_p_start < check_start && check_start < offset_p_end;
            let eq = (p_start == p_end) || (check_start == check_end);
            let overlap = has_overlap_start || has_start_within;
            eq || overlap
        })
        .map(|(_, ip)| ip)
}

fn find_constants<RVB>(
    rvb: &RVB,
    var_starts: &mut Vec<usize>,
    var_lengths: &mut Vec<usize>,
    constant_ps: &mut Vec<usize>,
    vars_with_zero_ops: &mut Vec<usize>,
) where
    RVB: RvbUpdater + ?Sized,
{
    // TODO: can parallelize this!
    // O(beta * n)
    (0..rvb.get_nvars()).for_each(|v| {
        let start = constant_ps.len();
        var_starts.push(start);
        rvb.constant_ops_on_var(v, constant_ps);
        debug_assert!(
            constant_ps.iter().cloned().all(|p| {
                let op = rvb.get_node_ref(p).unwrap().get_op_ref();
                op.get_vars().len() == 1
            }),
            "RVB cluster only supports constant ops with a single variable."
        );
        var_lengths.push(constant_ps.len() - start);
        if constant_ps.len() == *var_starts.last().unwrap() {
            vars_with_zero_ops.push(v);
        }
    });
}

/// Get returns n with chance 1/2^(n+1)
/// Chance of failure is 1/2^(2^64) and should therefore be acceptable.
pub fn contiguous_bits<R: Rng>(r: &mut R) -> usize {
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

fn calculate_mult(
    bonds_before: &BondContainer<usize>,
    bonds_after: &BondContainer<usize>,
    n: usize,
) -> f64 {
    let close = (bonds_before.get_total_weight() - bonds_after.get_total_weight()).abs()
        < std::f64::EPSILON;
    if n == 0 || close {
        1.0
    } else {
        let new_mult =
            (bonds_after.get_total_weight() / bonds_before.get_total_weight()).powi(n as i32);
        debug_assert!({
            let valid = new_mult >= 0.;
            if !valid {
                println!(
                    "Negative multiplier: {}\t{}\t{}",
                    bonds_after.get_total_weight(),
                    bonds_before.get_total_weight(),
                    n
                );
            }
            valid
        });
        new_mult
    }
}

#[cfg(test)]
mod sc_tests {
    use super::*;
    use crate::sse::fast_ops::*;
    use smallvec::smallvec;

    #[test]
    fn test_overlapping_regions_simple() {
        let cutoff = 10;
        let flips = [0, 2, 4, 6, 8];
        let p_start = 1;
        let p_end = 7;
        let overlaps = find_overlapping_starts(p_start, p_end, cutoff, &flips).collect::<Vec<_>>();
        println!("{:?}", overlaps);
        // [0 - 6]
        assert_eq!(overlaps, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_overlapping_regions() {
        let cutoff = 10;
        let flips = [0, 2, 4, 6, 8];
        let p_start = 5;
        let p_end = 7;
        let overlaps = find_overlapping_starts(p_start, p_end, cutoff, &flips).collect::<Vec<_>>();
        println!("{:?}", overlaps);
        assert_eq!(overlaps, vec![2, 3]);
    }

    #[test]
    fn test_wrap_around() {
        let cutoff = 10;
        let flips = [0, 2, 4, 6, 8];
        let p_start = 7;
        let p_end = 1;
        let overlaps = find_overlapping_starts(p_start, p_end, cutoff, &flips).collect::<Vec<_>>();
        println!("{:?}", overlaps);
        assert_eq!(overlaps, vec![3, 4, 0]);
    }

    #[test]
    fn test_remove_dups() {
        let mut v = vec![0, 0, 1, 2, 3, 3];
        remove_doubles(&mut v);
        assert_eq!(v, vec![1, 2])
    }

    #[test]
    fn test_remove_dups_again() {
        let mut v = vec![0, 0, 1, 2, 2, 3];
        remove_doubles(&mut v);
        assert_eq!(v, vec![1, 3])
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

        fn bond_mag(&self, _b: usize) -> f64 {
            1.0
        }
    }

    fn large_joined_manager() -> (FastOps, EN) {
        (
            FastOps::new_from_ops(
                2,
                vec![
                    (
                        0,
                        FastOp::offdiagonal(
                            smallvec![0],
                            2,
                            smallvec![false],
                            smallvec![false],
                            true,
                        ),
                    ),
                    (
                        1,
                        FastOp::offdiagonal(
                            smallvec![1],
                            3,
                            smallvec![false],
                            smallvec![false],
                            true,
                        ),
                    ),
                    (
                        2,
                        FastOp::offdiagonal(
                            smallvec![0],
                            2,
                            smallvec![false],
                            smallvec![false],
                            true,
                        ),
                    ),
                    (
                        3,
                        FastOp::offdiagonal(
                            smallvec![1],
                            3,
                            smallvec![false],
                            smallvec![false],
                            true,
                        ),
                    ),
                    (
                        4,
                        FastOp::diagonal(smallvec![0, 1], 0, smallvec![false, false], false),
                    ),
                    (
                        5,
                        FastOp::offdiagonal(
                            smallvec![0],
                            2,
                            smallvec![false],
                            smallvec![false],
                            true,
                        ),
                    ),
                    (
                        6,
                        FastOp::offdiagonal(
                            smallvec![1],
                            3,
                            smallvec![false],
                            smallvec![false],
                            true,
                        ),
                    ),
                    (
                        7,
                        FastOp::offdiagonal(
                            smallvec![0],
                            2,
                            smallvec![false],
                            smallvec![false],
                            true,
                        ),
                    ),
                    (
                        8,
                        FastOp::offdiagonal(
                            smallvec![1],
                            3,
                            smallvec![false],
                            smallvec![false],
                            true,
                        ),
                    ),
                ]
                .into_iter(),
            ),
            EN {
                bonds_for_var: vec![vec![0, 1], vec![0, 1]],
                bonds: vec![((0, 1), true), ((0, 1), false)],
            },
        )
    }

    #[test]
    fn run_two_joined_check_flip_p() {
        let (mut manager, edges) = large_joined_manager();
        debug_print_diagonal(&manager, &[false, false]);

        let p = calculate_flip_prob(
            &mut manager,
            (&[false, false], &mut [false, false]),
            (&mut [false, false], &[3, 6]),
            &[Some(2), Some(1)],
            (&[0, 1], |v| Some(v)),
            (
                |b, sa, sb| {
                    let pref = edges.bond_prefers_aligned(b);
                    let aligned = sa == sb;
                    if aligned == pref {
                        0.0
                    } else {
                        1.0
                    }
                },
                |_| 1.0,
                &edges,
            ),
        );
        println!("{}", p);
    }
}
