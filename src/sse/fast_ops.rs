use crate::sse::qmc_traits::*;
use crate::sse::qmc_types::Op;
use smallvec::SmallVec;

pub struct FastOps {
    ops: Vec<Option<FastOpNode>>,
    n: usize,
    p_ends: Option<(usize, usize)>,
    var_ends: Vec<Option<(usize, usize)>>,
}

type LinkVars = SmallVec<[Option<usize>; 2]>;

#[derive(Clone, Debug)]
pub struct FastOpNode {
    op: Op,
    previous_p: Option<usize>,
    next_p: Option<usize>,
    previous_for_vars: LinkVars,
    next_for_vars: LinkVars,
}

impl FastOpNode {
    fn new(op: Op, previous_for_vars: LinkVars, next_for_vars: LinkVars) -> Self {
        let nvars = op.vars.len();
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

impl OpNode for FastOpNode {
    fn get_op(&self) -> Op {
        self.op.clone()
    }

    fn get_op_ref(&self) -> &Op {
        &self.op
    }

    fn get_op_mut(&mut self) -> &mut Op {
        &mut self.op
    }
}

impl DiagonalUpdater for FastOps {
    fn set_pth(&mut self, _p: usize, _op: Option<Op>) -> Option<Op> {
        unimplemented!()
    }

    /// This is actually what's called, if you override this you may leave set_pth unimplemented.
    /// Folds across the p values, passing T down.
    fn mutate_ps<F, T>(&mut self, cutoff: usize, t: T, f: F) -> T
    where
        F: Fn(&Self, Option<&Op>, T) -> (Option<Option<Op>>, T),
    {
        if cutoff > self.ops.len() {
            self.ops.resize_with(cutoff, || None)
        };

        let last_p: Option<usize> = None;
        let last_vars: Vec<Option<usize>> = vec![None; self.var_ends.len()];

        let (t, _, _) = (0..cutoff).fold(
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
                        let vars = &node_ref.op.vars;
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
                            .vars
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
                        prevs.iter().zip(new_op.vars.iter()).for_each(|(prev, v)| {
                            if let Some(prev) = prev {
                                let prev_node = self.ops[*prev].as_mut().unwrap();
                                let indx = prev_node.op.index_of_var(*v).unwrap();
                                prev_node.next_for_vars[indx] = Some(p);
                            } else {
                                self.var_ends[*v] = if let Some((head, tail)) = self.var_ends[*v] {
                                    assert!(head >= p);
                                    Some((p, tail))
                                } else {
                                    Some((p, p))
                                }
                            }
                        });

                        nexts.iter().zip(new_op.vars.iter()).for_each(|(next, v)| {
                            if let Some(next) = next {
                                let next_node = self.ops[*next].as_mut().unwrap();
                                let indx = next_node.op.index_of_var(*v).unwrap();
                                next_node.previous_for_vars[indx] = Some(p);
                            } else {
                                self.var_ends[*v] = if let Some((head, tail)) = self.var_ends[*v] {
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
                    op.vars.iter().cloned().for_each(|v| last_vars[v] = Some(p));
                    last_p = Some(p)
                }

                (t, last_p, last_vars)
            },
        );
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
        }
    }
}

impl OpContainer for FastOps {
    fn set_cutoff(&mut self, cutoff: usize) {
        if cutoff > self.ops.len() {
            self.ops.resize_with(cutoff, || None)
        }
    }

    fn get_n(&self) -> usize {
        self.n
    }

    fn get_nvars(&self) -> usize {
        self.var_ends.len()
    }

    fn get_pth(&self, p: usize) -> Option<&Op> {
        self.ops[p].as_ref().map(|opnode| &opnode.op)
    }

    fn weight<H>(&self, h: H) -> f64
    where
        H: Fn(&[usize], usize, &[bool], &[bool]) -> f64,
    {
        let mut t = 1.0;
        let mut p = self.p_ends.map(|(p, _)| p);
        while p.is_some() {
            let op = self.ops[p.unwrap()].as_ref().unwrap();
            t *= h(&op.op.vars, op.op.bond, &op.op.inputs, &op.op.outputs);
            p = op.next_p;
        }
        t
    }
}

impl LoopUpdater<FastOpNode> for FastOps {
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

impl ClusterUpdater<FastOpNode> for FastOps {}

impl ConvertsToDiagonal<FastOps> for FastOps {
    fn convert_to_diagonal(self) -> FastOps {
        self
    }
}

impl ConvertsToLooper<FastOpNode, FastOps> for FastOps {
    fn convert_to_looper(self) -> FastOps {
        self
    }
}
