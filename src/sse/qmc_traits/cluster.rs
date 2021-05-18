use crate::sse::qmc_traits::directed_loop::*;
use crate::sse::qmc_traits::op_container::*;
use crate::sse::qmc_types::*;
use crate::util::allocator::{Factory, StackTuplizer};
use rand::Rng;

/// Add cluster updates to LoopUpdater.
pub trait ClusterUpdater:
    LoopUpdater
    + Factory<Vec<bool>>
    + Factory<Vec<usize>>
    + Factory<Vec<OpIndex>>
    + Factory<Vec<Option<usize>>>
    + Factory<Vec<OpSide>>
{
    /// Flip each cluster in the graph using an rng instance, add to state changes in acc. Use this
    /// version if there's ising symmetry in your graph.
    fn flip_each_cluster_ising_symmetry_rng<R: Rng>(
        &mut self,
        prob: f64,
        rng: &mut R,
        state: &mut [bool],
    ) {
        self.flip_each_cluster_rng(prob, rng, state, None::<fn(&Self::Node) -> f64>)
    }

    /// Flip each cluster in the graph using an rng instance, add to state changes in acc. You can
    /// provide a function to adjust the weights for flipping each cluster based off the nodes
    /// within it. Edges of the cluster will always be constant ops but the total weight change
    /// inside the cluster will affect the probability of acceptance (multiplied by a global `prob`
    /// for each cluster). This function should take a node reference and return the ratio of the
    /// new weight divided by old weight if a global spin flip were to take place.
    /// # Rust language note:
    /// To call with none and keep types happy you must use `None::<fn(&Self::Node) -> f64>` or
    /// call the helper function `flip_each_cluster_ising_symmetry_rng` which does it for you.
    fn flip_each_cluster_rng<R: Rng, F>(
        &mut self,
        prob: f64,
        rng: &mut R,
        state: &mut [bool],
        weight_change_on_global_flip: Option<F>,
    ) where
        F: Fn(&Self::Node) -> f64,
    {
        if self.get_n() == 0 {
            return;
        }

        let last_p = self.get_last_p().unwrap();
        let mut boundaries = StackTuplizer::<Option<usize>, Option<usize>>::new(self);
        boundaries.resize_each(last_p + 1, || None, || None);

        let constant_op_loc = self.find_constant_op();
        let n_clusters = if let Some(constant_op_loc) = constant_op_loc {
            // Expand to edges of cluster
            let mut frontier = StackTuplizer::<OpIndex, OpSide>::new(self);
            frontier.push((constant_op_loc, OpSide::Outputs));
            frontier.push((constant_op_loc, OpSide::Inputs));

            let mut cluster_num = 1;
            loop {
                while let Some((loc, frontier_side)) = frontier.pop() {
                    match boundaries.get(loc.into()) {
                        Some((Some(_), Some(_))) => { /* This was hit by another cluster expansion. */
                        }
                        Some(_) => {
                            expand_whole_cluster::<Self>(
                                self,
                                loc,
                                (0, frontier_side),
                                cluster_num,
                                &mut boundaries,
                                &mut frontier,
                            );
                            cluster_num += 1;
                        }
                        None => unreachable!(),
                    }
                }
                // Check if any site ops are not yet set to a cluster.
                let unmapped_loc = boundaries
                    .iter()
                    .enumerate()
                    .map(|(loc, x)| -> (OpIndex, _) { (loc.into(), x) })
                    .find_map(|(loc, (a, b))| match (a, b) {
                        (None, None) => Some(loc),
                        (Some(_), None) | (None, Some(_)) => unreachable!(),
                        _ => None,
                    });
                if let Some(loc) = unmapped_loc {
                    frontier.push((loc, OpSide::Outputs));
                    frontier.push((loc, OpSide::Inputs));
                } else {
                    break;
                }
            }
            frontier.dissolve(self);
            cluster_num
        } else {
            // The whole thing is one cluster.
            boundaries.iter_mut().enumerate().for_each(|(p, (a, b))| {
                if self.get_node_ref(p).is_some() {
                    *a = Some(0);
                    *b = Some(0);
                }
            });
            1
        };

        let mut flips: Vec<bool> = self.get_instance();
        // If the ising symmetry is broken calculate out weight terms.
        if let Some(f) = weight_change_on_global_flip {
            let mut flips_weights: Vec<f64> = self.get_instance();
            flips_weights.resize(n_clusters, 1.0);
            boundaries
                .iter()
                .enumerate()
                .filter_map(|(p, clust)| match clust {
                    (Some(a), Some(b)) => Some((p, (a, b))),
                    (None, None) => None,
                    _ => unreachable!(),
                })
                .for_each(|(p, (input_cluster, output_cluster))| {
                    let node = self.get_node_ref(p).unwrap();
                    // If entire cluster will flip, then see weight changes.
                    if input_cluster == output_cluster {
                        flips_weights[*input_cluster] *= f(node)
                    }
                });
            flips.extend(
                flips_weights
                    .iter()
                    .cloned()
                    .map(|c| rng.gen_bool(c * prob)),
            );
            self.return_instance(flips_weights);
        } else {
            flips.extend((0..n_clusters).map(|_| rng.gen_bool(prob)));
        };
        boundaries
            .iter()
            .enumerate()
            .filter_map(|(p, clust)| match clust {
                (Some(a), Some(b)) => Some((p, (a, b))),
                (None, None) => None,
                _ => unreachable!(),
            })
            .for_each(|(p, (input_cluster, output_cluster))| {
                if flips[*input_cluster] {
                    let node = self.get_node_mut(p).unwrap();
                    let op = node.get_op_mut();
                    flip_state_for_op(op, OpSide::Inputs);
                    // Mark state changes if needed.
                    let node = self.get_node_ref(p).unwrap();
                    let op = node.get_op_ref();
                    (0..op.get_vars().len()).for_each(|relvar| {
                        let prev_p = self.get_previous_prel_for_rel_var(relvar, node);
                        if prev_p.is_none() {
                            state[op.get_vars()[relvar]] = op.get_inputs()[relvar];
                        }
                    });
                }
                if flips[*output_cluster] {
                    let node = self.get_node_mut(p).unwrap();
                    let op = node.get_op_mut();
                    flip_state_for_op(op, OpSide::Outputs)
                }
            });
        boundaries.dissolve(self);
        self.return_instance(flips);
        self.post_cluster_update_hook();
    }

    /// Find a site with a constant op.
    fn find_constant_op(&self) -> Option<OpIndex> {
        let mut loc = self.get_first_loc();
        while let Some(node_loc) = loc {
            let node = self.get_node_ref_loc(node_loc);
            if is_valid_cluster_edge_op(node.get_op_ref()) {
                return Some(node_loc);
            } else {
                loc = self.get_next_loc(node);
            }
        }
        None
    }

    /// Called after an update.
    fn post_cluster_update_hook(&mut self) {}
}

/// Expand a cluster at a given p and leg.
fn expand_whole_cluster<C: ClusterUpdater + ?Sized>(
    c: &mut C,
    loc: OpIndex,
    leg: Leg,
    cluster_num: usize,
    boundaries: &mut StackTuplizer<Option<usize>, Option<usize>>,
    frontier: &mut StackTuplizer<OpIndex, OpSide>,
) {
    let mut interior_frontier = StackTuplizer::<OpIndex, Leg>::new(c);

    let node = c.get_node_ref_loc(loc);
    let op = node.get_op_ref();
    if !is_valid_cluster_edge_op(op) {
        // Add all legs
        debug_assert_eq!(boundaries.at(loc.into()), (&None, &None));
        let inputs_legs = (0..op.get_vars().len()).map(|v| (v, OpSide::Inputs));
        let outputs_legs = (0..op.get_vars().len()).map(|v| (v, OpSide::Outputs));
        let all_legs = inputs_legs.chain(outputs_legs);
        interior_frontier.extend(all_legs.map(|l| (loc, l)));
    } else {
        debug_assert_eq!(op.get_vars().len(), 1);
        interior_frontier.push((loc, leg))
    };

    while let Some((loc, leg)) = interior_frontier.pop() {
        set_boundary(loc, leg.1, cluster_num, boundaries);
        let node = c.get_node_ref_loc(loc);
        let op = node.get_op_ref();
        let relvar = leg.0;
        let var = op.get_vars()[relvar];
        let ((next_loc, next_node), next_leg) = match leg.1 {
            OpSide::Inputs => {
                let prev_p = c.get_previous_prel_for_rel_var(relvar, node);
                let PRel {
                    loc: prev_loc,
                    relv: prev_relv,
                } = prev_p.unwrap_or_else(|| c.get_last_prel_for_var(var).unwrap());
                let prev_node = c.get_node_ref_loc(prev_loc);
                ((prev_loc, prev_node), (prev_relv, OpSide::Outputs))
            }
            OpSide::Outputs => {
                let next_p = c.get_next_prel_for_rel_var(relvar, node);
                let PRel {
                    loc: next_loc,
                    relv: next_relv,
                } = next_p.unwrap_or_else(|| c.get_first_prel_for_var(var).unwrap());
                let next_node = c.get_node_ref_loc(next_loc);
                ((next_loc, next_node), (next_relv, OpSide::Inputs))
            }
        };

        // If we hit a cluster edge, add to frontier and mark in boundary.
        if is_valid_cluster_edge_op(next_node.get_op_ref()) {
            if !set_boundary(next_loc, next_leg.1, cluster_num, boundaries) {
                frontier.push((next_loc, next_leg.1.reverse()))
            }
        } else {
            // Allow (None, None), (Some(c), None) or (None, Some(c))
            // For (None, None) just set c==cluster_num as a hack.
            let (a, b) = boundaries.at(next_loc.into());
            match ((*a, *b), cluster_num) {
                ((None, None), c) | ((Some(c), None), _) | ((None, Some(c)), _)
                    if c == cluster_num =>
                {
                    set_boundaries(next_loc, cluster_num, boundaries);

                    let next_op = next_node.get_op_ref();
                    let inputs_legs = (0..next_op.get_vars().len()).map(|v| (v, OpSide::Inputs));
                    let outputs_legs = (0..next_op.get_vars().len()).map(|v| (v, OpSide::Outputs));
                    let new_legs = inputs_legs.chain(outputs_legs).filter(|l| *l != next_leg);

                    interior_frontier.extend(new_legs.map(|leg| (next_loc, leg)));
                }
                _ => (),
            }
        }
    }
    interior_frontier.dissolve(c);
}

/// Valid cluster edges are constant and have a single variable, in the future we can rewrite
/// the cluster logic to handle multi-spin cluster edges but that requires some fancy boundary
/// logic.
fn is_valid_cluster_edge_op<O: Op>(op: &O) -> bool {
    is_valid_cluster_edge(op.is_constant(), op.get_vars().len())
}

/// Valid cluster edges are constant and have a single variable, in the future we can rewrite
/// the cluster logic to handle multi-spin cluster edges but that requires some fancy boundary
/// logic.
#[inline]
pub fn is_valid_cluster_edge(is_constant: bool, nvars: usize) -> bool {
    is_constant && nvars == 1
}

// Returns true if both sides have clusters attached.
fn set_boundary(
    loc: OpIndex,
    sel: OpSide,
    cluster_num: usize,
    boundaries: &mut StackTuplizer<Option<usize>, Option<usize>>,
) -> bool {
    let t = boundaries.at(loc.into());
    let res = match (sel, t) {
        (OpSide::Inputs, (None, t1)) => (Some(cluster_num), *t1),
        (OpSide::Outputs, (t0, None)) => (*t0, Some(cluster_num)),
        // Now being careful
        (OpSide::Inputs, (Some(c), t1)) if *c == cluster_num => (Some(cluster_num), *t1),
        (OpSide::Outputs, (t0, Some(c))) if *c == cluster_num => (*t0, Some(cluster_num)),
        _ => unreachable!(),
    };
    boundaries.set(loc.into(), res);
    matches!(boundaries.at(loc.into()), (Some(_), Some(_)))
}

fn set_boundaries(
    loc: OpIndex,
    cluster_num: usize,
    boundaries: &mut StackTuplizer<Option<usize>, Option<usize>>,
) {
    set_boundary(loc, OpSide::Inputs, cluster_num, boundaries);
    set_boundary(loc, OpSide::Outputs, cluster_num, boundaries);
}

fn flip_state_for_op<O: Op>(op: &mut O, side: OpSide) {
    op.edit_in_out(|ins, outs| match side {
        OpSide::Inputs => ins.iter_mut().for_each(|b| *b = !*b),
        OpSide::Outputs => outs.iter_mut().for_each(|b| *b = !*b),
    });
}
