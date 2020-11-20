use crate::memory::allocator::Factory;
use crate::sse::*;
use rand::Rng;

/// Resonating bond update.
pub trait RVBClusterUpdater: RVBUpdater + DiagonalSubsection + Factory<Vec<Option<usize>>> {
    /// Perform a resonating bond update.
    fn rvb_cluster_update<R: Rng, EN: EdgeNavigator>(
        &mut self,
        edges: &EN,
        state: &mut [bool],
        updates: usize,
        rng: &mut R,
    ) -> usize {
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
            let choice = rng.gen_range(0, constant_ps.len() + vars_with_zero_ops.len());
            let (v, flip) = if choice < constant_ps.len() {
                let res = var_starts.binary_search(&choice);
                let v = match res {
                    Err(i) => i - 1,
                    Ok(i) => i,
                };
                (v, Some(choice))
            } else {
                let choice = choice - constant_ps.len();
                (vars_with_zero_ops[choice], None)
            };

            let mut cluster_vars: Vec<usize> = self.get_instance();
            let mut cluster_flips: Vec<Option<usize>> = self.get_instance();
            let mut boundary_vars: Vec<usize> = self.get_instance();
            let mut boundary_flips_pos: Vec<Option<usize>> = self.get_instance();

            let cluster_size = contiguous_bits(rng) + 1;
            build_cluster(
                cluster_size,
                (v, flip),
                (self.get_nvars(), self.get_cutoff()),
                (&mut cluster_vars, &mut cluster_flips),
                (&mut boundary_vars, &mut boundary_flips_pos),
                (&var_starts, &var_lengths),
                &constant_ps,
                edges,
                self,
                rng,
            );
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
                edges,
            );
            let should_mutate = if p_to_flip > 1.0 {
                true
            } else {
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
                    edges,
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
fn mutate_graph<RVB: RVBClusterUpdater + ?Sized, VS, EN: EdgeNavigator + ?Sized, R: Rng>(
    rvb: &mut RVB,
    (state, substate): (&[bool], &mut [bool]),
    (cluster_state, cluster_flips): (&mut [bool], &[usize]),
    boundary_tops: &[Option<usize>], // top for each of vars.
    (vars, var_to_subvar): (&[usize], VS),
    edges: &EN,
    rng: &mut R,
) where
    VS: Fn(usize) -> Option<usize>,
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

    let is_sat = |bond: usize, substate: &[bool]| -> bool {
        let p_aligned = edges.bond_prefers_aligned(bond);
        let (a, b) = edges.vars_for_bond(bond);
        let a = var_to_subvar(a).unwrap();
        let b = var_to_subvar(b).unwrap();
        let aligned = substate[a] == substate[b];
        aligned == p_aligned
    };

    // Now we have a series of pairs.
    let mut sat_bonds: BondContainer<usize> = rvb.get_instance();
    let mut unsat_bonds: BondContainer<usize> = rvb.get_instance();
    let mut next_cluster_index = 0;

    vars.iter()
        .cloned()
        .filter(|v| cluster_state[var_to_subvar(*v).unwrap()])
        .for_each(|v| {
            edges.bonds_for_var(v).iter().cloned().for_each(|b| {
                let ov = edges.other_var_for_bond(v, b).unwrap();
                if !cluster_state[var_to_subvar(ov).unwrap()] {
                    if is_sat(b, substate) {
                        sat_bonds.insert(b);
                    } else {
                        unsat_bonds.insert(b);
                    }
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

                let mut args = rvb.get_empty_args(SubvarAccess::VARLIST(&vars));
                rvb.fill_args_at_p_with_hint(from, &mut args, vars, boundary_tops.iter().cloned());

                let acc = (
                    next_cluster_index,
                    &mut sat_bonds,
                    &mut unsat_bonds,
                    substate,
                    cluster_state,
                    rng,
                );
                let ret = rvb.mutate_subsection_ops(
                    from,
                    until,
                    acc,
                    |_, op, p, acc| {
                        let (
                            mut next_cluster_index,
                            sat_bonds,
                            unsat_bonds,
                            substate,
                            cluster_state,
                            mut rng,
                        ) = acc;

                        debug_assert!(sat_bonds.iter().cloned().all(|b| is_sat(b, substate)));
                        debug_assert!(unsat_bonds.iter().cloned().all(|b| !is_sat(b, substate)));

                        let in_sat = sat_bonds.contains(&op.get_bond());
                        let in_unsat = unsat_bonds.contains(&op.get_bond());
                        let at_next_cluster_flip = next_cluster_index < cluster_flips.len()
                            && p == cluster_flips[next_cluster_index];
                        let newop = if in_sat || in_unsat {
                            // Rotatable ops must be diagonal.
                            debug_assert!(op.is_diagonal());

                            // Need to rotate
                            let new_bond = if in_sat {
                                unsat_bonds.get_random(&mut rng).unwrap()
                            } else {
                                sat_bonds.get_random(&mut rng).unwrap()
                            };

                            let (new_a, new_b) = edges.vars_for_bond(*new_bond);
                            let vars = RVB::Op::make_vars([new_a, new_b].iter().cloned());
                            let state = RVB::Op::make_substate(
                                [new_a, new_b]
                                    .iter()
                                    .cloned()
                                    .map(|v| var_to_subvar(v).unwrap())
                                    .map(|subvar| substate[subvar]),
                            );
                            let new_op =
                                RVB::Op::diagonal(vars, *new_bond, state, op.is_constant());

                            Some(Some(new_op))
                        } else if at_next_cluster_flip {
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
                            // Having changed substate now sat_bonds and unsat_bonds are invalid
                            // near this op, luckily flipping the cluster will remove all the
                            // invalid bonds.

                            toggle_cluster_and_bonds(
                                op,
                                cluster_state,
                                substate,
                                is_sat,
                                (sat_bonds, unsat_bonds),
                                &var_to_subvar,
                                edges,
                            );
                            toggle_state_and_bonds(
                                &new_op,
                                substate,
                                sat_bonds,
                                unsat_bonds,
                                &var_to_subvar,
                                edges,
                            );

                            next_cluster_index += 1;
                            Some(Some(new_op))
                        } else {
                            // Flip appropriate inputs/outputs.
                            // If any are out, then all are out - otherwise would be in sat or unsat
                            debug_assert!(
                                {
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
                                },
                                "All variables must be in or out, and at least one of those two."
                            );

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
                            } else if any_in_cluster {
                                let new_op = op.clone_and_edit_in_out_symmetric(|state| {
                                    state.iter_mut().for_each(|b| *b = !*b);
                                });

                                if !op.is_diagonal() {
                                    // Update state and bonds.
                                    toggle_state_and_bonds(
                                        &new_op,
                                        substate,
                                        sat_bonds,
                                        unsat_bonds,
                                        &var_to_subvar,
                                        edges,
                                    );
                                }

                                Some(Some(new_op))
                            } else {
                                if !op.is_diagonal() {
                                    // Update state and bonds.
                                    toggle_state_and_bonds(
                                        op,
                                        substate,
                                        sat_bonds,
                                        unsat_bonds,
                                        &var_to_subvar,
                                        edges,
                                    );
                                }

                                None
                            }
                        };

                        (
                            newop,
                            (
                                next_cluster_index,
                                sat_bonds,
                                unsat_bonds,
                                substate,
                                cluster_state,
                                rng,
                            ),
                        )
                    },
                    Some(args),
                );
                next_cluster_index = ret.0;
                let substate = ret.3;
                let cluster_state = ret.4;
                let rng = ret.5;
                (substate, cluster_state, rng)
            },
        );
    rvb.return_instance(sat_bonds);
    rvb.return_instance(unsat_bonds);

    rvb.return_instance(jump_to);
    rvb.return_instance(continue_until);
}

fn calculate_flip_prob<RVB: RVBClusterUpdater + ?Sized, VS, EN: EdgeNavigator + ?Sized>(
    rvb: &mut RVB,
    (state, substate): (&[bool], &mut [bool]),
    (cluster_state, cluster_flips): (&mut [bool], &[usize]),
    boundary_tops: &[Option<usize>], // top for each of vars.
    (vars, var_to_subvar): (&[usize], VS),
    edges: &EN,
) -> f64
where
    VS: Fn(usize) -> Option<usize>,
{
    let mut cluster_size = cluster_state.iter().cloned().filter(|x| *x).count();
    let mut psel = rvb.get_first_p();
    let mut next_cluster_index = 0;
    let mut mult = 1.0;

    let is_sat = |bond: usize, substate: &[bool]| -> bool {
        let p_aligned = edges.bond_prefers_aligned(bond);
        let (a, b) = edges.vars_for_bond(bond);
        let a = var_to_subvar(a).unwrap();
        let b = var_to_subvar(b).unwrap();
        let aligned = substate[a] == substate[b];
        aligned == p_aligned
    };

    let mut sat_bonds: BondContainer<usize> = rvb.get_instance();
    let mut unsat_bonds: BondContainer<usize> = rvb.get_instance();
    let mut n_sat = 0;
    let mut n_unsat = 0;
    if cluster_size != 0 {
        vars.iter()
            .cloned()
            .filter(|v| cluster_state[var_to_subvar(*v).unwrap()])
            .for_each(|v| {
                edges.bonds_for_var(v).iter().cloned().for_each(|b| {
                    let ov = edges.other_var_for_bond(v, b).unwrap();
                    if !cluster_state[var_to_subvar(ov).unwrap()] {
                        if is_sat(b, substate) {
                            sat_bonds.insert(b);
                        } else {
                            unsat_bonds.insert(b);
                        }
                    }
                })
            });
    }

    // Jump to cluster start, requires propagating state. Since not inside a cluster it's
    // safe to use just the boundary.
    while let Some(mut p) = psel {
        // Skip ahead.
        if cluster_size == 0 {
            debug_assert_eq!(sat_bonds.len(), 0);
            debug_assert_eq!(unsat_bonds.len(), 0);
            debug_assert_eq!(n_sat, 0);
            debug_assert_eq!(n_unsat, 0);
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
        debug_assert!(sat_bonds.iter().cloned().all(|b| is_sat(b, substate)));
        debug_assert!(unsat_bonds.iter().cloned().all(|b| !is_sat(b, substate)));

        let node = rvb.get_node_ref(p).unwrap();
        let op = node.get_op_ref();
        let is_cluster_bound =
            next_cluster_index < cluster_flips.len() && p == cluster_flips[next_cluster_index];
        let will_change_bonds = !op.is_diagonal() || is_cluster_bound;

        if will_change_bonds {
            // Commit the counts so far.
            mult *= calculate_mult(&sat_bonds, &unsat_bonds, n_sat, n_unsat);
            n_sat = 0;
            n_unsat = 0;
            // Break early if we reach 0 probability.
            if mult < std::f64::EPSILON {
                break;
            }
        }

        debug_assert!(
            {
                let any_in_cluster = op
                    .get_vars()
                    .iter()
                    .filter_map(|v| var_to_subvar(*v))
                    .any(|subvar| cluster_state[subvar]);
                let any_in_bound = op
                    .get_vars()
                    .iter()
                    .filter_map(|v| var_to_subvar(*v))
                    .any(|subvar| !cluster_state[subvar]);
                let b = op.get_bond();
                match (any_in_cluster, any_in_bound) {
                    (true, true) => sat_bonds.contains(&b) || unsat_bonds.contains(&b),
                    _ => true,
                }
            },
            "If op spans cluster and boundary it should appear in one of the bond sets."
        );

        // Count which bond it belongs to.
        let b = op.get_bond();
        if sat_bonds.contains(&b) {
            n_sat += 1;
        } else if unsat_bonds.contains(&b) {
            n_unsat += 1;
        }

        // We are at a cluster boundary, flips cluster state and bonds.
        if is_cluster_bound {
            debug_assert!(op.is_constant());
            cluster_size = (cluster_size as i64
                + toggle_cluster_and_bonds(
                    op,
                    cluster_state,
                    substate,
                    is_sat,
                    (&mut sat_bonds, &mut unsat_bonds),
                    &var_to_subvar,
                    edges,
                )) as usize;
            next_cluster_index += 1;
        }

        if !op.is_diagonal() {
            // Update state and bonds.
            toggle_state_and_bonds(
                op,
                substate,
                &mut sat_bonds,
                &mut unsat_bonds,
                &var_to_subvar,
                edges,
            );
        }
        // Move on
        psel = rvb.get_next_p(node);
    }
    // Commit remaining stuff.
    mult *= calculate_mult(&sat_bonds, &unsat_bonds, n_sat, n_unsat);

    rvb.return_instance(sat_bonds);
    rvb.return_instance(unsat_bonds);

    mult
}

fn toggle_cluster_and_bonds<O: Op, F, SAT, EN>(
    op: &O,
    cluster_state: &mut [bool],
    substate: &[bool],
    is_sat: SAT,
    (sat_bonds, unsat_bonds): (&mut BondContainer<usize>, &mut BondContainer<usize>),
    var_to_subvar: F,
    edges: &EN,
) -> i64
where
    F: Fn(usize) -> Option<usize>,
    SAT: Fn(usize, &[bool]) -> bool,
    EN: EdgeNavigator + ?Sized,
{
    let toggle = |bonds: &mut BondContainer<usize>, b: usize| {
        if bonds.contains(&b) {
            bonds.remove(&b);
        } else {
            bonds.insert(b);
        }
    };
    let mut cluster_delta = 0;
    op.get_vars()
        .iter()
        .filter_map(|v| var_to_subvar(*v).map(|subvar| (*v, subvar)))
        .for_each(|(v, subvar)| {
            cluster_state[subvar] = !cluster_state[subvar];
            cluster_delta = if cluster_state[subvar] {
                cluster_delta + 1
            } else {
                cluster_delta - 1
            };
            // Update bonds.
            edges.bonds_for_var(v).iter().cloned().for_each(|b| {
                if is_sat(b, substate) {
                    toggle(sat_bonds, b)
                } else {
                    toggle(unsat_bonds, b)
                };
            });
        });
    cluster_delta
}

fn toggle_state_and_bonds<O: Op, F, EN>(
    op: &O,
    substate: &mut [bool],
    sat_bonds: &mut BondContainer<usize>,
    unsat_bonds: &mut BondContainer<usize>,
    var_to_subvar: F,
    edges: &EN,
) where
    F: Fn(usize) -> Option<usize>,
    EN: EdgeNavigator + ?Sized,
{
    op.get_vars()
        .iter()
        .cloned()
        .zip(op.get_inputs().iter().cloned())
        .zip(op.get_outputs().iter().cloned())
        .filter_map(
            |((v, bin), bout)| {
                if bin == bout {
                    None
                } else {
                    Some((v, bout))
                }
            },
        )
        .filter_map(|(v, bout)| {
            var_to_subvar(v)
                .zip(Some(bout))
                .map(|(subvar, bout)| (v, subvar, bout))
        })
        .for_each(|(v, subvar, b)| {
            substate[subvar] = b;
            edges.bonds_for_var(v).iter().for_each(|b| {
                // Swap bonds.
                if sat_bonds.contains(b) {
                    sat_bonds.remove(b);
                    unsat_bonds.insert(*b);
                } else if unsat_bonds.contains(b) {
                    unsat_bonds.remove(b);
                    sat_bonds.insert(*b);
                }
            })
        });
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

fn build_cluster<Fact: ?Sized, EN, R: Rng>(
    mut cluster_size: usize,
    (init_var, init_flip): (usize, Option<usize>),
    (nvars, cutoff): (usize, usize),
    (cluster_vars, cluster_flips): (&mut Vec<usize>, &mut Vec<Option<usize>>),
    (boundary_vars, boundary_flips_pos): (&mut Vec<usize>, &mut Vec<Option<usize>>),
    (var_starts, var_lengths): (&[usize], &[usize]),
    constant_ps: &[usize],
    edges: &EN,
    fact: &mut Fact,
    rng: &mut R,
) where
    Fact: Factory<Vec<bool>> + Factory<Vec<usize>> + Factory<Vec<Option<usize>>>,
    EN: EdgeNavigator,
{
    let mut empty_var_in_cluster: Vec<bool> = fact.get_instance();
    let mut op_p_in_cluster: Vec<bool> = fact.get_instance();
    empty_var_in_cluster.resize(nvars, false);
    op_p_in_cluster.resize(cutoff, false);

    boundary_vars.push(init_var);
    boundary_flips_pos.push(init_flip);
    if let Some(init_flip) = init_flip {
        op_p_in_cluster[constant_ps[init_flip]] = true;
    } else {
        empty_var_in_cluster[init_var] = true;
    }

    while cluster_size > 0 && !boundary_vars.is_empty() {
        let to_pop = rng.gen_range(0, boundary_vars.len());
        let v = pop_index(boundary_vars, to_pop).unwrap();
        let flip = pop_index(boundary_flips_pos, to_pop).unwrap();

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
            if !op_p_in_cluster[constant_ps[flip_dec]] {
                op_p_in_cluster[constant_ps[flip_dec]] = true;

                boundary_vars.push(v);
                boundary_flips_pos.push(Some(flip_dec));
            }
            if !op_p_in_cluster[constant_ps[flip_inc]] {
                op_p_in_cluster[constant_ps[flip_inc]] = true;

                boundary_vars.push(v);
                boundary_flips_pos.push(Some(flip_inc));
            }
        }

        // Add neighbors to what we just added.
        edges.bonds_for_var(v).iter().for_each(|b| {
            let ov = edges.other_var_for_bond(v, *b).unwrap();
            if var_lengths[ov] == 0 {
                if !empty_var_in_cluster[ov] {
                    empty_var_in_cluster[ov] = true;

                    boundary_vars.push(ov);
                    boundary_flips_pos.push(None);
                }
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
                            if !op_p_in_cluster[constant_ps[flip_pos]] {
                                op_p_in_cluster[constant_ps[flip_pos]] = true;

                                boundary_vars.push(ov);
                                boundary_flips_pos.push(Some(flip_pos));
                            }
                        })
                } else {
                    // Add them all.
                    pis.for_each(|pi| {
                        if !op_p_in_cluster[constant_ps[pi]] {
                            op_p_in_cluster[constant_ps[pi]] = true;

                            boundary_vars.push(ov);
                            boundary_flips_pos.push(Some(pi));
                        }
                    })
                }
            }
        });

        cluster_size -= 1;
    }

    fact.return_instance(empty_var_in_cluster);
    fact.return_instance(op_p_in_cluster);
}

fn pop_index<T>(v: &mut Vec<T>, index: usize) -> Option<T> {
    let len = v.len();
    if index < len {
        v.swap(index, len - 1);
        v.pop()
    } else {
        None
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
    RVB: RVBClusterUpdater + ?Sized,
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
            &edges,
        );
        println!("{}", p);
    }
}
