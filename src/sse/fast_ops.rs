use crate::sse::qmc_traits::*;
use crate::sse::qmc_types::{Leg, OpSide};
#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};
use smallvec::{smallvec, SmallVec};

/// Underlying op for storing graph data, good for 2-variable ops.
pub type FastOp = BasicOp<SmallVec<[usize; 2]>, SmallVec<[bool; 2]>>;
/// A default implementation of the FastOps container, good for 2-variable ops.
pub type FastOps = FastOpsTemplate<FastOp>;
/// A default implementation of the FastOpNode container, good for 2-variable ops.
pub type FastOpNode = FastOpNodeTemplate<FastOp>;

/// Underlying op for storing graph data, good for 2-variable ops.
#[cfg(feature = "const_generics")]
pub type FastOpN<const N: usize> = BasicOp<SmallVec<[usize; N]>, SmallVec<[bool; N]>>;
/// A default implementation of the FastOps container, good for 2-variable ops.
#[cfg(feature = "const_generics")]
pub type FastOpsN<const N: usize> = FastOpsTemplate<FastOpN<N>>;
/// A default implementation of the FastOpNode container, good for 2-variable ops.
#[cfg(feature = "const_generics")]
pub type FastOpNodeN<const N: usize> = FastOpNodeTemplate<FastOpN<N>>;

/// A fast op container.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct FastOpsTemplate<O: Op> {
    pub(crate) ops: Vec<Option<FastOpNodeTemplate<O>>>,
    pub(crate) n: usize,
    pub(crate) p_ends: Option<(usize, usize)>,
    pub(crate) var_ends: Vec<Option<(usize, usize)>>,

    // For each variable check if flipped by an offdiagonal op.
    // for use with semiclassical updates.
    memoized_offdiagonal_ops: Option<Vec<bool>>,

    frontier: Option<Vec<(usize, OpSide)>>,
    interior_frontier: Option<Vec<(usize, Leg)>>,
    boundaries: Option<Vec<(Option<usize>, Option<usize>)>>,
    flips: Option<Vec<bool>>,
    leg_and_weight: Option<Vec<(Leg, f64)>>,
    // Reusable vector
    last_vars_alloc: Option<Vec<Option<usize>>>,
    // Semi classical updates
    classical_vars_alloc: Option<Vec<bool>>,
    classical_boundary_alloc: Option<Vec<(usize, bool)>>,
    classical_sat_alloc: Option<BondContainer<usize>>,
    classical_broken_alloc: Option<BondContainer<usize>>,
    // Optional bond counting.
    bond_counters: Option<Vec<usize>>,
}

impl<O: Op + Clone> FastOpsTemplate<O> {
    fn new_from_nvars_and_nbonds(nvars: usize, nbonds: Option<usize>) -> Self {
        Self {
            ops: vec![],
            n: 0,
            p_ends: None,
            var_ends: vec![None; nvars],
            memoized_offdiagonal_ops: Some(vec![false; nvars]),
            frontier: Some(vec![]),
            interior_frontier: Some(vec![]),
            boundaries: Some(vec![]),
            flips: Some(vec![]),
            leg_and_weight: Some(vec![]),
            last_vars_alloc: Some(vec![]),
            classical_vars_alloc: Some(vec![]),
            classical_boundary_alloc: Some(vec![]),
            classical_sat_alloc: Some(BondContainer::default()),
            classical_broken_alloc: Some(BondContainer::default()),
            bond_counters: nbonds.map(|nbonds| vec![0; nbonds]),
        }
    }

    fn new_from_nvars(nvars: usize) -> Self {
        Self::new_from_nvars_and_nbonds(nvars, None)
    }

    /// Make a new Manager from an interator of ops, and number of variables.
    pub fn new_from_ops<I: Iterator<Item = (usize, O)>>(nvars: usize, ps_and_ops: I) -> Self {
        let mut man = Self::new_from_nvars(nvars);
        man.clear_and_install_ops(ps_and_ops);
        man
    }

    fn clear_and_install_ops<I: Iterator<Item = (usize, O)>>(&mut self, ps_and_ops: I) {
        let nvars = self.var_ends.len();
        let ps_and_ops = ps_and_ops.collect::<Vec<_>>();
        if ps_and_ops.is_empty() {
            return;
        }
        let opslen = ps_and_ops.iter().map(|(p, _)| p).max().unwrap() + 1;
        self.ops.clear();
        self.ops.resize_with(opslen, || None);
        self.var_ends.clear();
        self.var_ends.resize_with(nvars, || None);
        self.p_ends = None;

        let last_p: Option<usize> = None;
        let mut last_vars = self.last_vars_alloc.take().unwrap();
        last_vars.clear();
        last_vars.resize(nvars, None);

        let (last_p, last_vars) =
            ps_and_ops
                .into_iter()
                .fold((last_p, last_vars), |(last_p, mut last_vars), (p, op)| {
                    if let Some(last_p) = last_p {
                        let last_node = self.ops[last_p].as_mut().unwrap();
                        last_node.next_p = Some(p);
                    } else {
                        self.p_ends = Some((p, p))
                    }

                    let previous_for_vars = op
                        .get_vars()
                        .iter()
                        .cloned()
                        .map(|v| {
                            if let Some(last_p) = last_vars[v] {
                                let node = self.ops[last_p].as_mut().unwrap();
                                let relv = node.get_op_ref().index_of_var(v).unwrap();
                                node.next_for_vars[relv] = Some(p);
                            } else {
                                self.var_ends[v] = Some((p, p));
                            }
                            let last_v = last_vars[v];
                            last_vars[v] = Some(p); // fine since vars cant be repeated.

                            last_v
                        })
                        .collect();

                    let n_opvars = op.get_vars().len();
                    let mut node = FastOpNodeTemplate::<O>::new(
                        op,
                        previous_for_vars,
                        smallvec![None; n_opvars],
                    );
                    node.previous_p = last_p;
                    self.ops[p] = Some(node);
                    self.n += 1;

                    (Some(p), last_vars)
                });
        if let Some((_, p_end)) = self.p_ends.as_mut() {
            *p_end = last_p.unwrap()
        }
        self.var_ends
            .iter_mut()
            .zip(last_vars.iter())
            .for_each(|(ends, last_v)| {
                if let Some((_, v_end)) = ends {
                    *v_end = last_v.unwrap();
                }
            });

        self.last_vars_alloc = Some(last_vars);
    }

    fn update_offdiagonal_lookup(&mut self) {
        let mut has_flips = self.memoized_offdiagonal_ops.take().unwrap();
        has_flips.iter_mut().for_each(|b| *b = false);
        let mut vars_left = self.get_nvars();
        let mut p = self.p_ends.map(|(start, _)| start);
        while let Some(node_p) = p {
            let node = self.ops[node_p].as_ref().unwrap();
            let op = node.get_op_ref();
            if !op.is_diagonal() {
                let eqs = op
                    .get_inputs()
                    .iter()
                    .zip(op.get_outputs().iter())
                    .map(|(input, output)| input == output);
                let _ = op
                    .get_vars()
                    .iter()
                    .cloned()
                    .zip(eqs)
                    .filter(|(_, eq)| !*eq)
                    .try_for_each(|(var, _)| {
                        if !has_flips[var] {
                            vars_left -= 1;
                            has_flips[var] = true;
                        }
                        if vars_left == 0 {
                            Err(())
                        } else {
                            Ok(())
                        }
                    });
            }
            p = node.next_p;
        }
        self.memoized_offdiagonal_ops = Some(has_flips);
    }
}

type LinkVars = SmallVec<[Option<usize>; 2]>;

/// A node which contains ops for FastOps.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct FastOpNodeTemplate<O: Op> {
    pub(crate) op: O,
    pub(crate) previous_p: Option<usize>,
    pub(crate) next_p: Option<usize>,
    pub(crate) previous_for_vars: LinkVars,
    pub(crate) next_for_vars: LinkVars,
}

impl<O: Op> FastOpNodeTemplate<O> {
    fn new(op: O, previous_for_vars: LinkVars, next_for_vars: LinkVars) -> Self {
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

impl<O: Op + Clone> OpNode<O> for FastOpNodeTemplate<O> {
    fn get_op(&self) -> O {
        self.op.clone()
    }

    fn get_op_ref(&self) -> &O {
        &self.op
    }

    fn get_op_mut(&mut self) -> &mut O {
        &mut self.op
    }
}

impl<O: Op + Clone> DiagonalUpdater for FastOpsTemplate<O> {
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
                                debug_assert_eq!(head, p);
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
                                debug_assert_eq!(tail, p);
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
                                    debug_assert_eq!(head, p);
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
                                    debug_assert_eq!(tail, p);
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
                        if let Some(bond_counters) = self.bond_counters.as_mut() {
                            let bond = node_ref.op.get_bond();
                            bond_counters[bond] -= 1;
                        }
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
                                    debug_assert_eq!(prev_p_for_v, None);
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
                                            debug_assert!(head >= p);
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
                                            debug_assert!(tail <= p);
                                            Some((head, p))
                                        } else {
                                            Some((p, p))
                                        }
                                }
                            });

                        let mut node_ref = FastOpNodeTemplate::<O>::new(new_op, prevs, nexts);
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
                                debug_assert!(head >= p);
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
                                debug_assert!(tail <= p);
                                Some((head, p))
                            } else {
                                Some((p, p))
                            }
                        };
                        if let Some(bond_counters) = self.bond_counters.as_mut() {
                            let bond = node_ref.op.get_bond();
                            bond_counters[bond] += 1;
                        }
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

    fn iterate_ops<F, T>(&self, mut t: T, f: F) -> T
    where
        F: Fn(&Self, &Self::Op, T) -> T,
    {
        let mut p = self.p_ends.map(|(start, _)| start);
        while let Some(node_p) = p {
            let node = self.ops[node_p].as_ref().unwrap();
            t = f(self, node.get_op_ref(), t);
            p = node.next_p;
        }
        t
    }
}

impl<O: Op + Clone> OpContainerConstructor for FastOpsTemplate<O> {
    fn new(nvars: usize) -> Self {
        Self::new_from_nvars(nvars)
    }

    fn new_with_bonds(nvars: usize, nbonds: usize) -> Self {
        Self::new_from_nvars_and_nbonds(nvars, Some(nbonds))
    }
}

impl<O: Op + Clone> OpContainer for FastOpsTemplate<O> {
    type Op = O;

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

impl<O: Op + Clone> LoopUpdater for FastOpsTemplate<O> {
    type Node = FastOpNodeTemplate<O>;

    fn get_node_ref(&self, p: usize) -> Option<&Self::Node> {
        self.ops[p].as_ref()
    }

    fn get_node_mut(&mut self, p: usize) -> Option<&mut Self::Node> {
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

    fn get_previous_p(&self, node: &Self::Node) -> Option<usize> {
        node.previous_p
    }

    fn get_next_p(&self, node: &Self::Node) -> Option<usize> {
        node.next_p
    }

    fn get_previous_p_for_rel_var(&self, revar: usize, node: &Self::Node) -> Option<usize> {
        node.previous_for_vars[revar]
    }

    fn get_next_p_for_rel_var(&self, revar: usize, node: &Self::Node) -> Option<usize> {
        node.next_for_vars[revar]
    }

    fn get_nth_p(&self, n: usize) -> usize {
        let n = n % self.n;
        let init = self.p_ends.map(|(head, _)| head).unwrap();
        (0..n).fold(init, |p, _| self.ops[p].as_ref().unwrap().next_p.unwrap())
    }

    fn get_leg_weight_alloc(&mut self) -> Vec<(Leg, f64)> {
        self.leg_and_weight.take().unwrap()
    }

    fn return_leg_weight_alloc(&mut self, mut alloc: Vec<(Leg, f64)>) {
        alloc.clear();
        self.leg_and_weight = Some(alloc)
    }

    fn post_loop_update_hook(&mut self) {
        self.update_offdiagonal_lookup()
    }
}

impl<O: Op + Clone> ClusterUpdater for FastOpsTemplate<O> {
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

    fn post_cluster_update_hook(&mut self) {
        self.update_offdiagonal_lookup()
    }
}

impl<O: Op + Clone> ClassicalLoopUpdater for FastOpsTemplate<O> {
    fn var_ever_flips(&self, var: usize) -> bool {
        self.memoized_offdiagonal_ops.as_ref().unwrap()[var]
    }

    fn count_ops_on_border(
        &self,
        sat_set: &BondContainer<usize>,
        broken_set: &BondContainer<usize>,
    ) -> (usize, usize) {
        if let Some(bond_counters) = self.bond_counters.as_ref() {
            let sats = sat_set
                .iter()
                .cloned()
                .map(|bond| bond_counters[bond])
                .sum();
            let brokens = broken_set
                .iter()
                .cloned()
                .map(|bond| bond_counters[bond])
                .sum();
            (sats, brokens)
        } else {
            count_using_iter_ops(self, sat_set, broken_set)
        }
    }

    fn get_var_alloc(&mut self, nvars: usize) -> Vec<bool> {
        let mut class = self.classical_vars_alloc.take().unwrap();
        class.resize(nvars, false);
        class
    }

    fn return_var_alloc(&mut self, mut alloc: Vec<bool>) {
        alloc.clear();
        self.classical_vars_alloc = Some(alloc)
    }

    fn get_boundary_alloc(&mut self) -> Vec<(usize, bool)> {
        self.classical_boundary_alloc.take().unwrap()
    }

    fn return_boundary_alloc(&mut self, mut alloc: Vec<(usize, bool)>) {
        alloc.clear();
        self.classical_boundary_alloc = Some(alloc)
    }

    fn get_sat_alloc(&mut self) -> BondContainer<usize> {
        self.classical_sat_alloc.take().unwrap()
    }

    fn get_broken_alloc(&mut self) -> BondContainer<usize> {
        self.classical_broken_alloc.take().unwrap()
    }

    fn return_sat_alloc(&mut self, mut alloc: BondContainer<usize>) {
        alloc.clear();
        self.classical_sat_alloc = Some(alloc);
    }

    fn return_broken_alloc(&mut self, mut alloc: BondContainer<usize>) {
        alloc.clear();
        self.classical_broken_alloc = Some(alloc);
    }
}
