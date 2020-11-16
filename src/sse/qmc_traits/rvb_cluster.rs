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
    ) {
        // TODO: can parallelize this!
        let mut var_starts: Vec<usize> = self.get_instance();
        let mut var_lengths: Vec<usize> = self.get_instance();
        let mut constant_ps: Vec<usize> = self.get_instance();
        // This helps us sample evenly.
        let mut vars_with_zero_ops: Vec<usize> = self.get_instance();

        // O(beta * n)
        (0..self.get_nvars()).for_each(|v| {
            let start = constant_ps.len();
            var_starts.push(start);
            self.constant_ops_on_var(v, &mut constant_ps);
            var_lengths.push(constant_ps.len() - start);
            if constant_ps.len() == *var_starts.last().unwrap() {
                vars_with_zero_ops.push(v);
            }
        });

        let mut cluster_vars: Vec<usize> = self.get_instance();
        let mut cluster_flips: Vec<Option<usize>> = self.get_instance();
        let mut boundary_vars: Vec<usize> = self.get_instance();
        let mut boundary_flips_pos: Vec<Option<usize>> = self.get_instance();
        let mut cluster_starting_state: Vec<bool> = self.get_instance();
        let mut cluster_toggle_ps: Vec<usize> = self.get_instance();

        let mut subvars: Vec<usize> = self.get_instance();
        let mut var_to_subvar: Vec<Option<usize>> = self.get_instance();
        let mut substate: Vec<bool> = self.get_instance();

        for _ in 0..updates {
            cluster_flips.clear();
            cluster_vars.clear();
            boundary_flips_pos.clear();
            boundary_vars.clear();
            cluster_toggle_ps.clear();
            cluster_starting_state.clear();
            var_to_subvar.clear();
            var_to_subvar.resize(self.get_nvars(), None);
            substate.clear();

            // Pick starting flip.
            let choice = rng.gen_range(0, constant_ps.len() + vars_with_zero_ops.len());
            let (v, flip) = if choice < constant_ps.len() {
                pick_starting_flip(&var_starts, choice)
            } else {
                let choice = choice - constant_ps.len();
                (vars_with_zero_ops[choice], None)
            };

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

            cluster_starting_state.resize(self.get_nvars(), false);
            cluster_vars
                .iter()
                .cloned()
                .zip(cluster_flips.iter().cloned())
                .for_each(|(v, fi)| {
                    if let Some(fi) = fi {
                        let vstart = var_starts[v];
                        let fi_rel = fi - vstart;

                        if fi_rel + 1 > var_lengths[v] {
                            cluster_starting_state[v] = true;
                            cluster_toggle_ps.push(constant_ps[fi]);
                            cluster_toggle_ps.push(constant_ps[vstart]);
                        } else {
                            cluster_toggle_ps.push(constant_ps[fi]);
                            cluster_toggle_ps.push(constant_ps[fi + 1]);
                        }
                    } else {
                        cluster_starting_state[v] = true;
                    }
                });

            // Should be able to infer substate based solely on boundary + cluster.
            // First get all vars in boundary or in cluster.
            subvars.extend(cluster_vars.iter().chain(boundary_vars.iter()).cloned());
            subvars.sort_unstable();
            subvars.dedup();
            subvars
                .iter()
                .cloned()
                .enumerate()
                .for_each(|(subvar, var)| var_to_subvar[var] = Some(subvar));

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
                                subvar_boundary_tops[subvar] = Some(*bfp)
                            }
                        }
                        (Some(_), None) | (None, None) => subvar_boundary_tops[subvar] = *bfp,
                        (None, Some(_)) => unreachable!(),
                    };
                });

            // Now lets get the cluster boundaries.
            cluster_toggle_ps.sort_unstable();
            remove_doubles(&mut cluster_toggle_ps);

            let p_to_flip = calculate_flip_prob(
                self,
                &mut substate,
                &mut cluster_starting_state,
                &cluster_toggle_ps,
                &subvar_boundary_tops,
                &subvars,
                edges,
                |v| var_to_subvar[v],
            );
            let should_mutate = if p_to_flip > 1.0 {
                true
            } else {
                rng.gen_bool(p_to_flip)
            };

            if should_mutate {
                // Great, mutate the graph.
            }

            self.return_instance(subvar_boundary_tops);

            // TODO finish this
            // Need to first check relevant region to get p to flip
            // then need to use boundary to make args to mutate
        }
        self.return_instance(var_to_subvar);
        self.return_instance(substate);
        self.return_instance(subvars);

        self.return_instance(cluster_flips);
        self.return_instance(cluster_vars);
        self.return_instance(boundary_flips_pos);
        self.return_instance(boundary_vars);

        self.return_instance(cluster_toggle_ps);
        self.return_instance(cluster_starting_state);

        self.return_instance(vars_with_zero_ops);
        self.return_instance(constant_ps);
        self.return_instance(var_lengths);
        self.return_instance(var_starts);
    }
}

fn calculate_flip_prob<RVB: RVBClusterUpdater + ?Sized, VS, EN: EdgeNavigator + ?Sized>(
    rvb: &mut RVB,
    substate: &mut [bool],
    cluster_state: &mut [bool],
    cluster_flips: &[usize],         // ps in sorted order
    boundary_tops: &[Option<usize>], // top for each of vars.
    vars: &[usize],
    edges: &EN,
    var_to_subvar: VS,
) -> f64
where
    VS: Fn(usize) -> Option<usize>,
{
    let mut cluster_size = cluster_state.iter().cloned().filter(|x| *x).count();
    let mut psel = Some(0);
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

        if op.is_diagonal() {
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
        } else {
            // Commit the counts so far.
            mult *= calculate_mult(&sat_bonds, &unsat_bonds, n_sat, n_unsat);
            n_sat = 0;
            n_unsat = 0;
            // Break early if we reach 0 probability.
            if mult < std::f64::EPSILON {
                break;
            }

            // Update state and bonds.
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

        // We are at a cluster boundary, flips cluster state and bonds.
        if p == cluster_flips[next_cluster_index] {
            debug_assert!(op.is_constant());
            debug_assert!(!op.is_diagonal());
            // Don't need to calculate mult stuff since this is necessarily not a diagonal op.
            op.get_vars()
                .iter()
                .filter_map(|v| var_to_subvar(*v).map(|subvar| (*v, subvar)))
                .for_each(|(v, subvar)| {
                    cluster_state[subvar] = !cluster_state[subvar];
                    cluster_size = if cluster_state[subvar] {
                        cluster_size + 1
                    } else {
                        cluster_size - 1
                    };
                    // Update bonds.
                    edges.bonds_for_var(v).iter().for_each(|b| {
                        let toggle_bonds = if is_sat(*b, substate) {
                            &mut sat_bonds
                        } else {
                            &mut unsat_bonds
                        };
                        if toggle_bonds.contains(b) {
                            toggle_bonds.remove(b);
                        } else {
                            toggle_bonds.insert(*b);
                        };
                    });
                });
            next_cluster_index += 1;
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

fn remove_doubles<T: Eq + Copy>(v: &mut Vec<T>) {
    let mut ii = 0;
    let mut jj = 0;
    while jj < v.len() - 1 {
        if v[jj] == v[jj + 1] {
            jj += 2;
        } else {
            v[ii] = v[jj];
            ii += 1;
            jj += 1;
        }
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

        cluster_vars.push(v);
        cluster_flips.push(flip);

        // Add neighbors to what we just added.
        edges.bonds_for_var(v).iter().for_each(|b| {
            let ov = edges.other_var_for_bond(v, *b).unwrap();
            if var_lengths[ov] == 0 && !empty_var_in_cluster[ov] {
                empty_var_in_cluster[ov] = true;
                boundary_vars.push(ov);
                boundary_flips_pos.push(None);
            } else {
                let pis = var_starts[ov]..var_starts[ov] + var_lengths[ov];
                if let Some(flip) = flip {
                    let pstart = constant_ps[flip];
                    let pend = constant_ps[(flip + 1) % var_lengths[ov]];
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

fn pick_starting_flip(var_starts: &[usize], choice: usize) -> (usize, Option<usize>) {
    let res = var_starts.binary_search(&choice);
    match res {
        Err(i) => (i - 1, Some(choice)),
        Ok(mut i) => {
            while i < var_starts.len() - 1 && var_starts[i + 1] == choice {
                i += 1;
            }
            (i, Some(choice))
        }
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
            p_start == p_end || {
                let check_start = (*p + cutoff - lowest_ps) % cutoff;
                let next_p = flip_positions[(*ip + 1) % flip_positions.len()];
                let check_end = (next_p + cutoff - lowest_ps) % cutoff;

                let has_overlap_start = check_start < offset_p_start && offset_p_start < check_end;
                let has_start_within = offset_p_start < check_start && check_start < offset_p_end;
                has_overlap_start || has_start_within
            }
        })
        .map(|(_, ip)| ip)
}

//
// struct FilterUsingState<T, V, It, P>
// where
//     It: Iterator<Item = T>,
//     P: Fn(T, &V) -> (bool, V),
// {
//     state: V,
//     iter: It,
//     pred: P,
// }
// impl<T, V, It, P> FilterUsingState<T, V, It, P>
// where
//     It: Iterator<Item = T>,
//     P: Fn(T, &V) -> (bool, V),
// {
//     fn new(state: V, iter: It, pred: P) -> Self {
//         Self { state, iter, pred }
//     }
// }
//
// impl<T, V, It, P> Iterator for FilterUsingState<T, V, It, P>
// where
//     It: Iterator<Item = T>,
//     P: Fn(T, &V) -> (bool, V),
// {
//     type Item = T;
//
//     fn next(&mut self) -> Option<Self::Item> {
//         for t in self.iter {
//             let (c, new_state) = (self.pred)(t, &self.state);
//             self.state = new_state;
//             if c {
//                 return Some(t);
//             }
//         }
//         None
//     }
// }

#[cfg(test)]
mod sc_tests {
    use super::*;

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
}
