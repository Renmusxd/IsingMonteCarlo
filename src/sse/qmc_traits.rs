use crate::sse::qmc_types::*;
use rand::Rng;
use smallvec::SmallVec;
use std::cmp::min;

pub trait OpNode {
    fn get_op(&self) -> Op;
    fn get_op_ref(&self) -> &Op;
    fn get_op_mut(&mut self) -> &mut Op;
}

pub trait OpContainerConstructor {
    fn new(nvars: usize) -> Self;
}

pub trait OpContainer {
    fn get_cutoff(&self) -> usize;
    fn set_cutoff(&mut self, cutoff: usize);
    fn get_n(&self) -> usize;
    fn get_nvars(&self) -> usize;
    fn get_pth(&self, p: usize) -> Option<&Op>;

    fn verify(&self, state: &[bool]) -> bool {
        let mut rolling_state = state.to_vec();
        for p in 0..self.get_cutoff() {
            let op = self.get_pth(p);
            if let Some(op) = op {
                for (v, inp) in op.vars.iter().zip(op.inputs.iter()) {
                    if rolling_state[*v] != *inp {
                        return false;
                    }
                }
                op.vars.iter().zip(op.outputs.iter()).for_each(|(v, out)| {
                    rolling_state[*v] = *out;
                })
            }
        }
        rolling_state
            .into_iter()
            .zip(state.iter().cloned())
            .all(|(a, b)| a == b)
    }
}

pub struct Hamiltonian<
    'a,
    H: Fn(&[usize], usize, &[bool], &[bool]) -> f64,
    E: Fn(usize) -> &'a [usize],
> {
    pub(crate) hamiltonian: H,
    pub(crate) num_edges: usize,
    pub(crate) edge_fn: E,
}

impl<'a, H: Fn(&[usize], usize, &[bool], &[bool]) -> f64, E: Fn(usize) -> &'a [usize]>
    Hamiltonian<'a, H, E>
{
    pub fn new(hamiltonian: H, edge_fn: E, num_edges: usize) -> Self {
        Hamiltonian {
            hamiltonian,
            edge_fn,
            num_edges,
        }
    }
}

pub trait DiagonalUpdater: OpContainer {
    fn set_pth(&mut self, p: usize, op: Option<Op>) -> Option<Op>;

    /// This is actually what's called, if you override this you may leave set_pth unimplemented.
    /// Folds across the p values, passing T down. Mutates op if returned values is Some(...)
    fn mutate_ps<F, T>(&mut self, cutoff: usize, t: T, f: F) -> T
    where
        F: Fn(&Self, Option<&Op>, T) -> (Option<Option<Op>>, T),
    {
        (0..cutoff).fold(t, |t, p| {
            let op = self.get_pth(p);
            let (op, t) = f(&self, op, t);
            if let Some(op) = op {
                self.set_pth(p, op);
            }
            t
        })
    }

    /// This is actually what's called, if you override this you may leave set_pth unimplemented.
    /// Folds across the p values, passing T down.
    fn iterate_ps<F, T>(&self, t: T, f: F) -> T
    where
        F: Fn(&Self, Option<&Op>, T) -> T,
    {
        let cutoff = self.get_cutoff();
        (0..cutoff).fold(t, |t, p| {
            let op = self.get_pth(p);
            f(&self, op, t)
        })
    }

    fn make_diagonal_update<'b, H, E>(
        &mut self,
        cutoff: usize,
        beta: f64,
        state: &[bool],
        hamiltonian: &Hamiltonian<'b, H, E>,
    ) where
        H: Fn(&[usize], usize, &[bool], &[bool]) -> f64,
        E: Fn(usize) -> &'b [usize],
    {
        self.make_diagonal_update_with_rng(
            cutoff,
            beta,
            state,
            hamiltonian,
            &mut rand::thread_rng(),
        )
    }

    fn make_diagonal_update_with_rng<'b, H, E, R: Rng>(
        &mut self,
        cutoff: usize,
        beta: f64,
        state: &[bool],
        hamiltonian: &Hamiltonian<'b, H, E>,
        rng: &mut R,
    ) where
        H: Fn(&[usize], usize, &[bool], &[bool]) -> f64,
        E: Fn(usize) -> &'b [usize],
    {
        let mut state = state.to_vec();
        self.make_diagonal_update_with_rng_and_state_ref(cutoff, beta, &mut state, hamiltonian, rng)
    }

    fn make_diagonal_update_with_rng_and_state_ref<'b, H, E, R: Rng>(
        &mut self,
        cutoff: usize,
        beta: f64,
        state: &mut [bool],
        hamiltonian: &Hamiltonian<'b, H, E>,
        rng: &mut R,
    ) where
        H: Fn(&[usize], usize, &[bool], &[bool]) -> f64,
        E: Fn(usize) -> &'b [usize],
    {
        self.mutate_ps(cutoff, (state, rng), |s, op, (state, rng)| {
            let op = Self::metropolis_single_diagonal_update(
                op,
                cutoff,
                s.get_n(),
                beta,
                state,
                hamiltonian,
                rng,
            );
            (op, (state, rng))
        });
    }

    fn metropolis_single_diagonal_update<'b, H, E, R: Rng>(
        op: Option<&Op>,
        cutoff: usize,
        n: usize,
        beta: f64,
        state: &mut [bool],
        hamiltonian: &Hamiltonian<'b, H, E>,
        rng: &mut R,
    ) -> Option<Option<Op>>
    where
        H: Fn(&[usize], usize, &[bool], &[bool]) -> f64,
        E: Fn(usize) -> &'b [usize],
    {
        let b = match op {
            None => rng.gen_range(0, hamiltonian.num_edges),
            Some(op) if op.is_diagonal() => op.bond,
            Some(Op { vars, outputs, .. }) => {
                vars.iter()
                    .zip(outputs.iter())
                    .for_each(|(v, b)| state[*v] = *b);
                return None;
            }
        };
        let vars = (hamiltonian.edge_fn)(b);
        let substate = vars
            .iter()
            .map(|v| state[*v])
            .collect::<SmallVec<[bool; 2]>>();
        let mat_element = (hamiltonian.hamiltonian)(vars, b, &substate, &substate);

        // This is based on equations 19a and 19b of arXiv:1909.10591v1 from 23 Sep 2019
        let numerator = beta * (hamiltonian.num_edges as f64) * mat_element;
        let denominator = (cutoff - n) as f64;

        match op {
            None => {
                if numerator > denominator || rng.gen_bool(numerator / denominator) {
                    let op = Op::diagonal(vars, b, substate);
                    Some(Some(op))
                } else {
                    None
                }
            }
            Some(op) if op.is_diagonal() => {
                let denominator = denominator + 1.0;
                if denominator > numerator || rng.gen_bool(denominator / numerator) {
                    Some(None)
                } else {
                    None
                }
            }
            _ => None,
        }
    }
}

pub enum LoopResult {
    Return,
    Iterate(usize, Leg),
}

pub trait LoopUpdater<Node: OpNode>: OpContainer {
    fn get_node_ref(&self, p: usize) -> Option<&Node>;
    fn get_node_mut(&mut self, p: usize) -> Option<&mut Node>;

    fn get_first_p(&self) -> Option<usize>;
    fn get_last_p(&self) -> Option<usize>;
    fn get_first_p_for_var(&self, var: usize) -> Option<usize>;
    fn get_last_p_for_var(&self, var: usize) -> Option<usize>;

    fn get_previous_p(&self, node: &Node) -> Option<usize>;
    fn get_next_p(&self, node: &Node) -> Option<usize>;

    fn get_previous_p_for_rel_var(&self, relvar: usize, node: &Node) -> Option<usize>;
    fn get_next_p_for_rel_var(&self, relvar: usize, node: &Node) -> Option<usize>;

    fn get_previous_p_for_var(&self, var: usize, node: &Node) -> Result<Option<usize>, ()> {
        let relvar = node.get_op_ref().index_of_var(var);
        if let Some(relvar) = relvar {
            Ok(self.get_previous_p_for_rel_var(relvar, node))
        } else {
            Err(())
        }
    }
    fn get_next_p_for_var(&self, var: usize, node: &Node) -> Result<Option<usize>, ()> {
        let relvar = node.get_op_ref().index_of_var(var);
        if let Some(relvar) = relvar {
            Ok(self.get_next_p_for_rel_var(relvar, node))
        } else {
            Err(())
        }
    }

    fn get_nth_p(&self, n: usize) -> usize {
        let acc = self
            .get_first_p()
            .map(|p| (p, self.get_node_ref(p).unwrap()))
            .unwrap();
        (0..n)
            .fold(acc, |(_, opnode), _| {
                let p = self.get_next_p(opnode).unwrap();
                (p, self.get_node_ref(p).unwrap())
            })
            .0
    }

    fn does_var_have_ops(&self, var: usize) -> bool {
        self.get_first_p_for_var(var).is_some()
    }

    fn make_loop_update<H>(
        &mut self,
        initial_n: Option<usize>,
        hamiltonian: H,
    ) -> Vec<(usize, bool)>
    where
        H: Fn(&[usize], usize, &[bool], &[bool]) -> f64,
    {
        self.make_loop_update_with_rng(initial_n, hamiltonian, &mut rand::thread_rng())
    }

    fn make_loop_update_with_rng<H, R: Rng>(
        &mut self,
        initial_n: Option<usize>,
        hamiltonian: H,
        rng: &mut R,
    ) -> Vec<(usize, bool)>
    where
        H: Fn(&[usize], usize, &[bool], &[bool]) -> f64,
    {
        let h = |op: &Op, entrance: Leg, exit: Leg| -> f64 {
            let (inputs, outputs) = adjust_states(op.inputs.clone(), op.outputs.clone(), entrance);
            let (inputs, outputs) = adjust_states(inputs, outputs, exit);
            // Call the supplied hamiltonian.
            hamiltonian(&op.vars, op.bond, &inputs, &outputs)
        };

        if self.get_n() > 0 {
            let initial_n = initial_n
                .map(|n| min(n, self.get_n()))
                .unwrap_or_else(|| rng.gen_range(0, self.get_n()));
            let nth_p = self.get_nth_p(initial_n);
            // Get starting leg for pth op.
            let op = self.get_node_ref(nth_p).unwrap();
            let n_vars = op.get_op_ref().vars.len();
            let initial_var = rng.gen_range(0, n_vars);
            let initial_direction = if rng.gen() {
                OpSide::Inputs
            } else {
                OpSide::Outputs
            };
            let initial_leg = (initial_var, initial_direction);

            let updates = self.apply_loop_update(
                (nth_p, initial_leg),
                nth_p,
                initial_leg,
                h,
                rng,
                vec![None; self.get_nvars()],
            );
            updates
                .into_iter()
                .enumerate()
                .fold(vec![], |mut acc, (i, v)| {
                    if let Some(v) = v {
                        acc.push((i, v))
                    };
                    acc
                })
        } else {
            vec![]
        }
    }

    fn apply_loop_update<H, R: Rng>(
        &mut self,
        initial_op_and_leg: (usize, Leg),
        mut sel_op_pos: usize,
        mut entrance_leg: Leg,
        h: H,
        rng: &mut R,
        mut acc: Vec<Option<bool>>,
    ) -> Vec<Option<bool>>
    where
        H: Copy + Fn(&Op, Leg, Leg) -> f64,
    {
        loop {
            let res = self.loop_body(
                initial_op_and_leg,
                sel_op_pos,
                entrance_leg,
                h,
                rng,
                &mut acc,
            );
            match res {
                LoopResult::Return => break acc,
                LoopResult::Iterate(new_sel_op_pos, new_entrance_leg) => {
                    sel_op_pos = new_sel_op_pos;
                    entrance_leg = new_entrance_leg;
                }
            }
        }
    }

    fn loop_body<H, R: Rng>(
        &mut self,
        initial_op_and_leg: (usize, Leg),
        sel_op_pos: usize,
        entrance_leg: Leg,
        h: H,
        rng: &mut R,
        acc: &mut [Option<bool>],
    ) -> LoopResult
    where
        H: Fn(&Op, Leg, Leg) -> f64,
    {
        let sel_opnode = self.get_node_mut(sel_op_pos).unwrap();
        let sel_op = sel_opnode.get_op();

        let inputs_legs = (0..sel_op.vars.len()).map(|v| (v, OpSide::Inputs));
        let outputs_legs = (0..sel_op.vars.len()).map(|v| (v, OpSide::Outputs));
        let legs = inputs_legs
            .chain(outputs_legs)
            .collect::<SmallVec<[(usize, OpSide); 4]>>();
        let weights = legs
            .iter()
            .map(|leg| h(&sel_op, entrance_leg, *leg))
            .collect::<SmallVec<[f64; 4]>>();
        let total_weight: f64 = weights.iter().sum();
        let choice = rng.gen_range(0.0, total_weight);
        let exit_leg = *weights
            .iter()
            .zip(legs.iter())
            .try_fold(choice, |c, (weight, leg)| {
                if c < *weight {
                    Err(leg)
                } else {
                    Ok(c - *weight)
                }
            })
            .unwrap_err();
        let (inputs, outputs) = adjust_states(
            sel_opnode.get_op_ref().inputs.clone(),
            sel_opnode.get_op_ref().outputs.clone(),
            entrance_leg,
        );
        let (inputs, outputs) = adjust_states(inputs, outputs, exit_leg);

        // Change the op now that we passed through.
        let sel_op_mut = sel_opnode.get_op_mut();
        sel_op_mut.inputs = inputs;
        sel_op_mut.outputs = outputs;

        // No longer need mutability.
        let sel_opnode = self.get_node_ref(sel_op_pos).unwrap();
        let sel_op = sel_opnode.get_op_ref();

        // Check if we closed the loop before going to next opnode.
        if (sel_op_pos, exit_leg) == initial_op_and_leg {
            LoopResult::Return
        } else {
            // Get the next opnode and entrance leg, let us know if it changes the initial/final.
            let (next_op_pos, var_to_match) = match exit_leg {
                (var, OpSide::Outputs) => {
                    let next_var_op = self.get_next_p_for_rel_var(var, sel_opnode);
                    let next = next_var_op.unwrap_or_else(|| {
                        acc[sel_op.vars[var]] = Some(sel_op.outputs[var]);
                        self.get_first_p_for_var(sel_op.vars[var]).unwrap()
                    });
                    (next, sel_op.vars[var])
                }
                (var, OpSide::Inputs) => {
                    let prev_var_op = self.get_previous_p_for_rel_var(var, sel_opnode);
                    let next = prev_var_op.unwrap_or_else(|| {
                        acc[sel_op.vars[var]] = Some(sel_op.inputs[var]);
                        self.get_last_p_for_var(sel_op.vars[var]).unwrap()
                    });
                    (next, sel_op.vars[var])
                }
            };

            let next_node = self.get_node_ref(next_op_pos).unwrap();
            let next_var_index = next_node.get_op_ref().index_of_var(var_to_match).unwrap();
            let new_entrance_leg = (next_var_index, exit_leg.1.reverse());

            // If back where we started, close loop and return state changes.
            if (next_op_pos, new_entrance_leg) == initial_op_and_leg {
                LoopResult::Return
            } else {
                LoopResult::Iterate(next_op_pos, new_entrance_leg)
            }
        }
    }
}

pub trait ClusterUpdater<Node: OpNode>: LoopUpdater<Node> {
    fn flip_each_cluster_rng<R: Rng>(&mut self, prob: f64, rng: &mut R) -> Vec<(usize, bool)> {
        let mut state_changes = vec![];
        self.flip_each_cluster_rng_to_acc(prob, rng, &mut state_changes);
        state_changes
    }

    fn flip_each_cluster_rng_to_acc<R: Rng>(
        &mut self,
        prob: f64,
        rng: &mut R,
        state_changes: &mut Vec<(usize, bool)>,
    ) {
        if self.get_n() == 0 {
            return;
        }

        let last_p = self.get_last_p().unwrap();
        let mut boundaries = self.get_boundaries_alloc(last_p + 1);

        let single_site_p = self.find_single_site();
        let n_clusters = if let Some(single_site_p) = single_site_p {
            // Expand to edges of cluster
            let mut frontier = self.get_frontier_alloc();
            frontier.push((single_site_p, OpSide::Outputs));
            frontier.push((single_site_p, OpSide::Inputs));

            let mut cluster_num = 1;
            loop {
                while let Some((p, frontier_side)) = frontier.pop() {
                    match boundaries.get(p) {
                        Some((Some(_), Some(_))) => { /* This was hit by another cluster expansion. */
                        }
                        Some(_) => {
                            self.expand_whole_cluster(
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
                    (0..op.vars.len()).for_each(|relvar| {
                        let prev_p = self.get_previous_p_for_rel_var(relvar, node);
                        if prev_p.is_none() {
                            state_changes.push((op.vars[relvar], op.inputs[relvar]));
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
    }

    fn expand_whole_cluster(
        &mut self,
        p: usize,
        leg: Leg,
        cluster_num: usize,
        boundaries: &mut [(Option<usize>, Option<usize>)],
        frontier: &mut Vec<(usize, OpSide)>,
    ) {
        let mut interior_frontier = self.get_interior_frontier_alloc();
        let node = self.get_node_ref(p).unwrap();
        if node.get_op().vars.len() > 1 {
            // Add all legs
            assert_eq!(boundaries[p], (None, None));
            let op = node.get_op_ref();
            let inputs_legs = (0..op.vars.len()).map(|v| (v, OpSide::Inputs));
            let outputs_legs = (0..op.vars.len()).map(|v| (v, OpSide::Outputs));
            let all_legs = inputs_legs.chain(outputs_legs);
            interior_frontier.extend(all_legs.map(|l| (p, l)));
        } else {
            interior_frontier.push((p, leg))
        };

        while let Some((p, leg)) = interior_frontier.pop() {
            set_boundary(p, leg.1, cluster_num, boundaries);
            let node = self.get_node_ref(p).unwrap();
            let op = node.get_op_ref();
            let relvar = leg.0;
            let var = op.vars[relvar];
            let ((next_p, next_node), next_leg) = match leg.1 {
                OpSide::Inputs => {
                    let prev_p = self.get_previous_p_for_rel_var(relvar, node);
                    let prev_p = prev_p.unwrap_or_else(|| self.get_last_p_for_var(var).unwrap());
                    let prev_node = self.get_node_ref(prev_p).unwrap();
                    let new_rel_var = prev_node.get_op_ref().index_of_var(var).unwrap();
                    ((prev_p, prev_node), (new_rel_var, OpSide::Outputs))
                }
                OpSide::Outputs => {
                    let next_p = self.get_next_p_for_rel_var(relvar, node);
                    let next_p = next_p.unwrap_or_else(|| self.get_first_p_for_var(var).unwrap());
                    let next_node = self.get_node_ref(next_p).unwrap();
                    let new_rel_var = next_node.get_op_ref().index_of_var(var).unwrap();
                    ((next_p, next_node), (new_rel_var, OpSide::Inputs))
                }
            };

            // If we hit a 1-site, add to frontier and mark in boundary.
            if next_node.get_op_ref().vars.len() == 1 {
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
                        let inputs_legs = (0..next_op.vars.len()).map(|v| (v, OpSide::Inputs));
                        let outputs_legs = (0..next_op.vars.len()).map(|v| (v, OpSide::Outputs));
                        let new_legs = inputs_legs.chain(outputs_legs).filter(|l| *l != next_leg);

                        interior_frontier.extend(new_legs.map(|leg| (next_p, leg)));
                    }
                    _ => (),
                }
            }
        }
        self.return_interior_frontier_alloc(interior_frontier);
    }

    fn find_single_site(&self) -> Option<usize> {
        let mut p = self.get_first_p();
        while let Some(node_p) = p {
            let node = self.get_node_ref(node_p).unwrap();
            if node.get_op_ref().vars.len() == 1 {
                return Some(node_p);
            } else {
                p = self.get_next_p(node);
            }
        }
        None
    }

    // Overwrite these functions to reduce the number of memory allocs in the cluster step.
    fn get_frontier_alloc(&mut self) -> Vec<(usize, OpSide)> {
        vec![]
    }

    fn get_interior_frontier_alloc(&mut self) -> Vec<(usize, Leg)> {
        vec![]
    }

    fn get_boundaries_alloc(&mut self, size: usize) -> Vec<(Option<usize>, Option<usize>)> {
        vec![(None, None); size]
    }

    fn get_flip_alloc(&mut self) -> Vec<bool> {
        vec![]
    }

    fn return_frontier_alloc(&mut self, _frontier: Vec<(usize, OpSide)>) {}

    fn return_interior_frontier_alloc(&mut self, _interior_frontier: Vec<(usize, Leg)>) {}

    fn return_boundaries_alloc(&mut self, _boundaries: Vec<(Option<usize>, Option<usize>)>) {}

    fn return_flip_alloc(&mut self, _flips: Vec<bool>) {}
}

pub trait ConvertsToDiagonal<D: DiagonalUpdater> {
    fn convert_to_diagonal(self) -> D;
}

pub trait ConvertsToLooper<N: OpNode, L: LoopUpdater<N>> {
    fn convert_to_looper(self) -> L;
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

fn flip_state_for_op(op: &mut Op, side: OpSide) {
    match side {
        OpSide::Inputs => op.inputs.iter_mut().for_each(|b| *b = !*b),
        OpSide::Outputs => op.outputs.iter_mut().for_each(|b| *b = !*b),
    }
}

pub(crate) fn debug_print_diagonal<D: DiagonalUpdater>(diagonal: &D, state: &[bool]) {
    let nvars = diagonal.get_nvars();
    for _ in 0..nvars {
        print!("=");
    }
    println!();
    for b in state {
        print!("{}", if *b { "1" } else { "0" });
    }
    println!();

    diagonal.iterate_ps(0, |_, op, p| {
        if let Some(op) = op {
            let mut last_var = 0;
            for (var, outp) in op.vars.iter().zip(op.outputs.iter()) {
                for _ in last_var..*var {
                    print!("|");
                }
                print!("{}", if *outp { 1 } else { 0 });
                last_var = var + 1;
            }
            for _ in last_var..nvars {
                print!("|");
            }
        } else {
            for _ in 0..nvars {
                print!("|");
            }
        }

        println!("\tp={}", p);
        p + 1
    });
}
