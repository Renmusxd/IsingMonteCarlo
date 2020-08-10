use crate::sse::qmc_traits::*;
use crate::sse::qmc_types::{Leg, OpSide};
#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

/// A fast op container.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct FastOps {
    pub(crate) ops: Vec<Option<FastOpNode>>,
    pub(crate) n: usize,
    pub(crate) p_ends: Option<(usize, usize)>,
    pub(crate) var_ends: Vec<Option<(usize, usize)>>,

    frontier: Option<Vec<(usize, OpSide)>>,
    interior_frontier: Option<Vec<(usize, Leg)>>,
    boundaries: Option<Vec<(Option<usize>, Option<usize>)>>,
    flips: Option<Vec<bool>>,
    // Reusable vector
    last_vars_alloc: Option<Vec<Option<usize>>>,
}

type FastOp = BasicOp<SmallVec<[usize; 2]>, SmallVec<[bool; 2]>>;
type LinkVars = SmallVec<[Option<usize>; 2]>;

/// A node which contains ops for FastOps.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct FastOpNode {
    pub(crate) op: FastOp,
    pub(crate) previous_p: Option<usize>,
    pub(crate) next_p: Option<usize>,
    pub(crate) previous_for_vars: LinkVars,
    pub(crate) next_for_vars: LinkVars,
}

impl FastOpNode {
    fn new(op: FastOp, previous_for_vars: LinkVars, next_for_vars: LinkVars) -> Self {
        let nvars = op.get_vars().len();
        assert_eq!(previous_for_vars.len(), nvars);
        assert_eq!(next_for_vars.len(), nvars);
        Self {
            op,
            previous_p: None,
            next_p: None,
            previous_for_vars,
            next_for_vars,
        }
    }
}

impl OpNode<FastOp> for FastOpNode {
    fn get_op(&self) -> FastOp {
        self.op.clone()
    }

    fn get_op_ref(&self) -> &FastOp {
        &self.op
    }

    fn get_op_mut(&mut self) -> &mut FastOp {
        &mut self.op
    }
}

impl DiagonalUpdater for FastOps {
    fn mutate_ps<F, T>(&mut self, cutoff: usize, t: T, f: F) -> T
    where
        F: Fn(&Self, Option<&Self::Op>, T) -> (Option<Option<Self::Op>>, T),
    {
        if cutoff > self.ops.len() {
            self.ops.resize(cutoff, None)
        };

        let last_p: Option<usize> = None;
        let mut last_vars = self.last_vars_alloc.take().unwrap();
        last_vars.clear();
        last_vars.resize(self.var_ends.len(), None);

        let (t, _, last_vars) = (0..cutoff).fold(
            (t, last_p, last_vars),
            |(t, mut last_p, mut last_vars), p| {
                let op_ref = self.get_pth(p);
                let (new_op, t) = f(&self, op_ref, t);

                if let Some(new_op) = new_op {
                    if let Some(node_ref) = self.ops[p].take() {
                        // Uninstall the old op.
                        // If there's a previous p, point it towards the next one
                        if let Some(last_p) = last_p {
                            let node = self.ops[last_p].as_mut().unwrap();
                            node.next_p = node_ref.next_p;
                        } else {
                            // No previous p, so we are removing the head. Make necessary adjustments.
                            self.p_ends = if let Some((head, tail)) = self.p_ends {
                                assert_eq!(head, p);
                                if tail == p {
                                    // We are removing the head and the tail.
                                    None
                                } else {
                                    // Keep tail as is, change the head.
                                    Some((node_ref.next_p.unwrap(), tail))
                                }
                            } else {
                                unreachable!()
                            }
                        }
                        // If there's a next p, point it towards the previous one
                        let next_p_node = node_ref.next_p.and_then(|p| self.ops[p].as_mut());
                        if let Some(next_p_node) = next_p_node {
                            next_p_node.previous_p = last_p
                        } else {
                            // No next p, so we are removing the tail. Adjust.
                            self.p_ends = if let Some((head, tail)) = self.p_ends {
                                assert_eq!(tail, p);
                                if head == p {
                                    // This should have been handled above.
                                    unreachable!()
                                } else {
                                    Some((head, node_ref.previous_p.unwrap()))
                                }
                            } else {
                                // Normally not allowed, but could have been set to None up above.
                                None
                            }
                        }

                        // Now do the same for variables.
                        let vars = &node_ref.op.get_vars();
                        vars.iter().cloned().enumerate().for_each(|(i, v)| {
                            // Check the previous node using this variable.
                            if let Some(prev_p_for_v) = last_vars[v] {
                                let prev_p_for_v_ref = self.ops[prev_p_for_v].as_mut().unwrap();
                                let prev_rel_indx = prev_p_for_v_ref.op.index_of_var(v).unwrap();

                                prev_p_for_v_ref.next_for_vars[prev_rel_indx] =
                                    node_ref.next_for_vars[i];
                            } else {
                                // This was the first one, need to edit vars list.
                                self.var_ends[v] = if let Some((head, tail)) = self.var_ends[v] {
                                    assert_eq!(head, p);
                                    if tail == p {
                                        // We are removing the head and the tail.
                                        None
                                    } else {
                                        // We are removing the head, keeping the tail.
                                        Some((node_ref.next_for_vars[i].unwrap(), tail))
                                    }
                                } else {
                                    unreachable!()
                                }
                            }

                            // Check the next nodes using this variable.
                            if let Some(next_p_for_v) = node_ref.next_for_vars[i] {
                                let next_p_for_v_ref = self.ops[next_p_for_v].as_mut().unwrap();
                                let next_rel_index = next_p_for_v_ref.op.index_of_var(v).unwrap();

                                next_p_for_v_ref.previous_for_vars[next_rel_index] = last_vars[v];
                            } else {
                                self.var_ends[v] = if let Some((head, tail)) = self.var_ends[v] {
                                    assert_eq!(tail, p);
                                    if head == p {
                                        // Should have been caught already.
                                        unreachable!()
                                    } else {
                                        // We are removing the tail, keeping the head.
                                        Some((head, node_ref.previous_for_vars[i].unwrap()))
                                    }
                                } else {
                                    // Could have been set to none previously.
                                    None
                                }
                            }
                        });
                        self.n -= 1;
                    }

                    if let Some(new_op) = new_op {
                        // Install the new one
                        let (prevs, nexts): (LinkVars, LinkVars) = new_op
                            .get_vars()
                            .iter()
                            .cloned()
                            .map(|v| -> (Option<usize>, Option<usize>) {
                                let prev_p_for_v = if let Some(prev_p_for_v) = last_vars[v] {
                                    // If there's a previous p for this variable, provide it
                                    Some(prev_p_for_v)
                                } else {
                                    // Otherwise return None
                                    None
                                };

                                let next_p_for_v = if let Some(prev_p_for_v) = last_vars[v] {
                                    // If there's a previous node for the var, check its next entry.
                                    let prev_node_for_v = self.ops[prev_p_for_v].as_ref().unwrap();
                                    let indx = prev_node_for_v.op.index_of_var(v).unwrap();
                                    prev_node_for_v.next_for_vars[indx]
                                } else if let Some((head, _)) = self.var_ends[v] {
                                    // Otherwise just look at the head (this is the new head).
                                    assert_eq!(prev_p_for_v, None);
                                    Some(head)
                                } else {
                                    // This is the new tail.
                                    None
                                };

                                (prev_p_for_v, next_p_for_v)
                            })
                            .unzip();

                        // Now adjust other nodes and ends
                        prevs
                            .iter()
                            .zip(new_op.get_vars().iter())
                            .for_each(|(prev, v)| {
                                if let Some(prev) = prev {
                                    let prev_node = self.ops[*prev].as_mut().unwrap();
                                    let indx = prev_node.op.index_of_var(*v).unwrap();
                                    prev_node.next_for_vars[indx] = Some(p);
                                } else {
                                    self.var_ends[*v] =
                                        if let Some((head, tail)) = self.var_ends[*v] {
                                            assert!(head >= p);
                                            Some((p, tail))
                                        } else {
                                            Some((p, p))
                                        }
                                }
                            });

                        nexts
                            .iter()
                            .zip(new_op.get_vars().iter())
                            .for_each(|(next, v)| {
                                if let Some(next) = next {
                                    let next_node = self.ops[*next].as_mut().unwrap();
                                    let indx = next_node.op.index_of_var(*v).unwrap();
                                    next_node.previous_for_vars[indx] = Some(p);
                                } else {
                                    self.var_ends[*v] =
                                        if let Some((head, tail)) = self.var_ends[*v] {
                                            assert!(tail <= p);
                                            Some((head, p))
                                        } else {
                                            Some((p, p))
                                        }
                                }
                            });

                        let mut node_ref = FastOpNode::new(new_op, prevs, nexts);
                        node_ref.previous_p = last_p;
                        node_ref.next_p = if let Some(last_p) = last_p {
                            let last_p_node = self.ops[last_p].as_ref().unwrap();
                            last_p_node.next_p
                        } else if let Some((head, _)) = self.p_ends {
                            Some(head)
                        } else {
                            None
                        };

                        // Based on what these were set to, adjust the p_ends and neighboring nodes.
                        if let Some(prev) = node_ref.previous_p {
                            let prev_node = self.ops[prev].as_mut().unwrap();
                            prev_node.next_p = Some(p);
                        } else {
                            self.p_ends = if let Some((head, tail)) = self.p_ends {
                                assert!(head >= p);
                                Some((p, tail))
                            } else {
                                Some((p, p))
                            }
                        };

                        if let Some(next) = node_ref.next_p {
                            let next_node = self.ops[next].as_mut().unwrap();
                            next_node.previous_p = Some(p);
                        } else {
                            self.p_ends = if let Some((head, tail)) = self.p_ends {
                                assert!(tail <= p);
                                Some((head, p))
                            } else {
                                Some((p, p))
                            }
                        };
                        self.ops[p] = Some(node_ref);
                        self.n += 1;
                    }
                }

                if let Some(op) = self.ops[p].as_ref().map(|r| &r.op) {
                    op.get_vars()
                        .iter()
                        .cloned()
                        .for_each(|v| last_vars[v] = Some(p));
                    last_p = Some(p)
                }

                (t, last_p, last_vars)
            },
        );
        self.last_vars_alloc = Some(last_vars);
        t
    }
}

impl OpContainerConstructor for FastOps {
    fn new(nvars: usize) -> Self {
        Self {
            ops: vec![],
            n: 0,
            p_ends: None,
            var_ends: vec![None; nvars],
            frontier: Some(vec![]),
            interior_frontier: Some(vec![]),
            boundaries: Some(vec![]),
            flips: Some(vec![]),
            last_vars_alloc: Some(vec![]),
        }
    }
}

impl OpContainer for FastOps {
    type Op = FastOp;

    fn get_cutoff(&self) -> usize {
        self.ops.len()
    }

    fn set_cutoff(&mut self, cutoff: usize) {
        if cutoff > self.ops.len() {
            self.ops.resize(cutoff, None)
        }
    }

    fn get_n(&self) -> usize {
        self.n
    }

    fn get_nvars(&self) -> usize {
        self.var_ends.len()
    }

    fn get_pth(&self, p: usize) -> Option<&Self::Op> {
        if p < self.ops.len() {
            self.ops[p].as_ref().map(|opnode| &opnode.op)
        } else {
            None
        }
    }
}

impl LoopUpdater for FastOps {
    type Node = FastOpNode;

    fn get_node_ref(&self, p: usize) -> Option<&FastOpNode> {
        self.ops[p].as_ref()
    }

    fn get_node_mut(&mut self, p: usize) -> Option<&mut FastOpNode> {
        self.ops[p].as_mut()
    }

    fn get_first_p(&self) -> Option<usize> {
        self.p_ends.map(|(p, _)| p)
    }

    fn get_last_p(&self) -> Option<usize> {
        self.p_ends.map(|(_, p)| p)
    }

    fn get_first_p_for_var(&self, var: usize) -> Option<usize> {
        self.var_ends[var].map(|(p, _)| p)
    }

    fn get_last_p_for_var(&self, var: usize) -> Option<usize> {
        self.var_ends[var].map(|(_, p)| p)
    }

    fn get_previous_p(&self, node: &FastOpNode) -> Option<usize> {
        node.previous_p
    }

    fn get_next_p(&self, node: &FastOpNode) -> Option<usize> {
        node.next_p
    }

    fn get_previous_p_for_rel_var(&self, revar: usize, node: &FastOpNode) -> Option<usize> {
        node.previous_for_vars[revar]
    }

    fn get_next_p_for_rel_var(&self, revar: usize, node: &FastOpNode) -> Option<usize> {
        node.next_for_vars[revar]
    }

    fn get_nth_p(&self, n: usize) -> usize {
        let n = n % self.n;
        let init = self.p_ends.map(|(head, _)| head).unwrap();
        (0..n).fold(init, |p, _| self.ops[p].as_ref().unwrap().next_p.unwrap())
    }
}

impl ClusterUpdater for FastOps {
    // No need for logic here, just reuse some allocations.
    fn get_frontier_alloc(&mut self) -> Vec<(usize, OpSide)> {
        self.frontier.take().unwrap()
    }

    fn get_interior_frontier_alloc(&mut self) -> Vec<(usize, Leg)> {
        self.interior_frontier.take().unwrap()
    }

    fn get_boundaries_alloc(&mut self, size: usize) -> Vec<(Option<usize>, Option<usize>)> {
        let mut boundaries = self.boundaries.take().unwrap();
        boundaries.resize(size, (None, None));
        boundaries
    }

    fn get_flip_alloc(&mut self) -> Vec<bool> {
        self.flips.take().unwrap()
    }

    // Put all the clears in the returns so that serialization doesn't save useless data.
    fn return_frontier_alloc(&mut self, mut frontier: Vec<(usize, OpSide)>) {
        frontier.clear();
        self.frontier = Some(frontier);
    }

    fn return_interior_frontier_alloc(&mut self, mut interior_frontier: Vec<(usize, Leg)>) {
        interior_frontier.clear();
        self.interior_frontier = Some(interior_frontier)
    }

    fn return_boundaries_alloc(&mut self, mut boundaries: Vec<(Option<usize>, Option<usize>)>) {
        boundaries.clear();
        self.boundaries = Some(boundaries)
    }

    fn return_flip_alloc(&mut self, mut flips: Vec<bool>) {
        flips.clear();
        self.flips = Some(flips)
    }
}
