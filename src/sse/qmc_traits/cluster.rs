use crate::sse::qmc_traits::directed_loop::*;
use crate::sse::qmc_traits::op_container::*;
use crate::sse::qmc_types::*;
use rand::Rng;

/// Add cluster updates to LoopUpdater.
pub trait ClusterUpdater: LoopUpdater {
    /// Flip each cluster in the graph using an rng instance, add to state changes in acc.
    fn flip_each_cluster_rng<R: Rng>(&mut self, prob: f64, rng: &mut R, state: &mut [bool]) {
        if self.get_n() == 0 {
            return;
        }

        let last_p = self.get_last_p().unwrap();
        let mut boundaries = self.get_boundaries_alloc(last_p + 1);

        let constant_op_p = self.find_constant_op();
        let n_clusters = if let Some(constant_op_p) = constant_op_p {
            // Expand to edges of cluster
            let mut frontier = self.get_frontier_alloc();
            frontier.push((constant_op_p, OpSide::Outputs));
            frontier.push((constant_op_p, OpSide::Inputs));

            let mut cluster_num = 1;
            loop {
                while let Some((p, frontier_side)) = frontier.pop() {
                    match boundaries.get(p) {
                        Some((Some(_), Some(_))) => { /* This was hit by another cluster expansion. */
                        }
                        Some(_) => {
                            expand_whole_cluster::<Self>(
                                self,
                                p,
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
                let unmapped_p = boundaries.iter().enumerate().find_map(|(p, (a, b))| {
                    self.get_node_ref(p).and_then(|_node| match (a, b) {
                        (None, None) => Some(p),
                        (Some(_), None) | (None, Some(_)) => unreachable!(),
                        _ => None,
                    })
                });
                if let Some(p) = unmapped_p {
                    frontier.extend_from_slice(&[(p, OpSide::Outputs), (p, OpSide::Inputs)])
                } else {
                    break;
                }
            }
            self.return_frontier_alloc(frontier);
            cluster_num
        } else {
            // The whole thing is one cluster.
            boundaries.iter_mut().enumerate().for_each(|(p, v)| {
                if self.get_node_ref(p).is_some() {
                    v.0 = Some(0);
                    v.1 = Some(0);
                }
            });
            1
        };

        let mut flips = self.get_flip_alloc();
        flips.extend((0..n_clusters).map(|_| rng.gen_bool(prob)));
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
                        let prev_p = self.get_previous_p_for_rel_var(relvar, node);
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
        self.return_boundaries_alloc(boundaries);
        self.return_flip_alloc(flips);
        self.post_cluster_update_hook();
    }

    /// Find a site with a constant op.
    fn find_constant_op(&self) -> Option<usize> {
        let mut p = self.get_first_p();
        while let Some(node_p) = p {
            let node = self.get_node_ref(node_p).unwrap();
            if is_valid_cluster_edge_op(node.get_op_ref()) {
                return Some(node_p);
            } else {
                p = self.get_next_p(node);
            }
        }
        None
    }

    // Overwrite these functions to reduce the number of memory allocs in the cluster step.
    /// Get an allocation for the frontier.
    fn get_frontier_alloc(&mut self) -> Vec<(usize, OpSide)> {
        vec![]
    }

    /// Get an allocation for the interior frontier.
    fn get_interior_frontier_alloc(&mut self) -> Vec<(usize, Leg)> {
        vec![]
    }

    /// Get an allocation for the bounaries.
    fn get_boundaries_alloc(&mut self, size: usize) -> Vec<(Option<usize>, Option<usize>)> {
        vec![(None, None); size]
    }

    /// Get an allocation for the spin flips.
    fn get_flip_alloc(&mut self) -> Vec<bool> {
        vec![]
    }

    /// Return an alloc.
    fn return_frontier_alloc(&mut self, _frontier: Vec<(usize, OpSide)>) {}

    /// Return an alloc.
    fn return_interior_frontier_alloc(&mut self, _interior_frontier: Vec<(usize, Leg)>) {}

    /// Return an alloc.
    fn return_boundaries_alloc(&mut self, _boundaries: Vec<(Option<usize>, Option<usize>)>) {}

    /// Return an alloc.
    fn return_flip_alloc(&mut self, _flips: Vec<bool>) {}

    /// Called after an update.
    fn post_cluster_update_hook(&mut self) {}
}

/// Expand a cluster at a given p and leg.
fn expand_whole_cluster<C: ClusterUpdater + ?Sized>(
    c: &mut C,
    p: usize,
    leg: Leg,
    cluster_num: usize,
    boundaries: &mut [(Option<usize>, Option<usize>)],
    frontier: &mut Vec<(usize, OpSide)>,
) {
    let mut interior_frontier = c.get_interior_frontier_alloc();
    let node = c.get_node_ref(p).unwrap();
    if node.get_op().get_vars().len() > 1 {
        // Add all legs
        assert_eq!(boundaries[p], (None, None));
        let op = node.get_op_ref();
        let inputs_legs = (0..op.get_vars().len()).map(|v| (v, OpSide::Inputs));
        let outputs_legs = (0..op.get_vars().len()).map(|v| (v, OpSide::Outputs));
        let all_legs = inputs_legs.chain(outputs_legs);
        interior_frontier.extend(all_legs.map(|l| (p, l)));
    } else {
        interior_frontier.push((p, leg))
    };

    while let Some((p, leg)) = interior_frontier.pop() {
        set_boundary(p, leg.1, cluster_num, boundaries);
        let node = c.get_node_ref(p).unwrap();
        let op = node.get_op_ref();
        let relvar = leg.0;
        let var = op.get_vars()[relvar];
        let ((next_p, next_node), next_leg) = match leg.1 {
            OpSide::Inputs => {
                let prev_p = c.get_previous_p_for_rel_var(relvar, node);
                let prev_p = prev_p.unwrap_or_else(|| c.get_last_p_for_var(var).unwrap());
                let prev_node = c.get_node_ref(prev_p).unwrap();
                let new_rel_var = prev_node.get_op_ref().index_of_var(var).unwrap();
                ((prev_p, prev_node), (new_rel_var, OpSide::Outputs))
            }
            OpSide::Outputs => {
                let next_p = c.get_next_p_for_rel_var(relvar, node);
                let next_p = next_p.unwrap_or_else(|| c.get_first_p_for_var(var).unwrap());
                let next_node = c.get_node_ref(next_p).unwrap();
                let new_rel_var = next_node.get_op_ref().index_of_var(var).unwrap();
                ((next_p, next_node), (new_rel_var, OpSide::Inputs))
            }
        };

        // If we hit a cluster edge, add to frontier and mark in boundary.
        if is_valid_cluster_edge_op(next_node.get_op_ref()) {
            if !set_boundary(next_p, next_leg.1, cluster_num, boundaries) {
                frontier.push((next_p, next_leg.1.reverse()))
            }
        } else {
            // Allow (None, None), (Some(c), None) or (None, Some(c))
            // For (None, None) just set c==cluster_num as a hack.
            match (boundaries[next_p], cluster_num) {
                ((None, None), c) | ((Some(c), None), _) | ((None, Some(c)), _)
                    if c == cluster_num =>
                {
                    set_boundaries(next_p, cluster_num, boundaries);

                    let next_op = next_node.get_op_ref();
                    let inputs_legs = (0..next_op.get_vars().len()).map(|v| (v, OpSide::Inputs));
                    let outputs_legs = (0..next_op.get_vars().len()).map(|v| (v, OpSide::Outputs));
                    let new_legs = inputs_legs.chain(outputs_legs).filter(|l| *l != next_leg);

                    interior_frontier.extend(new_legs.map(|leg| (next_p, leg)));
                }
                _ => (),
            }
        }
    }
    c.return_interior_frontier_alloc(interior_frontier);
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
    p: usize,
    sel: OpSide,
    cluster_num: usize,
    boundaries: &mut [(Option<usize>, Option<usize>)],
) -> bool {
    let t = &boundaries[p];
    boundaries[p] = match (sel, t) {
        (OpSide::Inputs, (None, t1)) => (Some(cluster_num), *t1),
        (OpSide::Outputs, (t0, None)) => (*t0, Some(cluster_num)),
        // Now being careful
        (OpSide::Inputs, (Some(c), t1)) if *c == cluster_num => (Some(cluster_num), *t1),
        (OpSide::Outputs, (t0, Some(c))) if *c == cluster_num => (*t0, Some(cluster_num)),
        _ => unreachable!(),
    };
    matches!(boundaries[p], (Some(_), Some(_)))
}

fn set_boundaries(p: usize, cluster_num: usize, boundaries: &mut [(Option<usize>, Option<usize>)]) {
    set_boundary(p, OpSide::Inputs, cluster_num, boundaries);
    set_boundary(p, OpSide::Outputs, cluster_num, boundaries);
}

fn flip_state_for_op<O: Op>(op: &mut O, side: OpSide) {
    match side {
        OpSide::Inputs => op.get_inputs_mut().iter_mut().for_each(|b| *b = !*b),
        OpSide::Outputs => op.get_outputs_mut().iter_mut().for_each(|b| *b = !*b),
    }
}