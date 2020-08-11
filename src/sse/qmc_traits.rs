use crate::sse::qmc_types::*;
use rand::Rng;
#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use std::cmp::min;
use std::iter::FromIterator;

/// Ops for holding SSE graph state.
pub trait Op {
    /// The list of op variables.
    type Vars: FromIterator<usize> + AsRef<[usize]> + AsMut<[usize]>;
    /// The list of op input and output states.
    type SubState: FromIterator<bool> + AsRef<[bool]> + AsMut<[bool]>;

    /// Make a diagonal op.
    fn diagonal<A, B>(vars: A, bond: usize, state: B) -> Self
    where
        A: Into<Self::Vars>,
        B: Into<Self::SubState>;

    /// Make an offdiagonal op.
    fn offdiagonal<A, B, C>(vars: A, bond: usize, inputs: B, outputs: C) -> Self
    where
        A: Into<Self::Vars>,
        B: Into<Self::SubState>,
        C: Into<Self::SubState>;

    /// Get the relative index of a variable.
    fn index_of_var(&self, var: usize) -> Option<usize>;

    /// Check if the op is diagonal (makes no state changes).
    fn is_diagonal(&self) -> bool {
        self.get_inputs() == self.get_outputs()
    }

    /// Get the set of variables used for this op.
    fn get_vars(&self) -> &[usize];

    /// Get the associated bond number for the op.
    fn get_bond(&self) -> usize;

    /// Get the input state for the op.
    fn get_inputs(&self) -> &[bool];

    /// Get the output state for the op.
    fn get_outputs(&self) -> &[bool];

    /// Get the input state for the op.
    fn get_inputs_mut(&mut self) -> &mut [bool];

    /// Get the output state for the op.
    fn get_outputs_mut(&mut self) -> &mut [bool];

    /// Get both the inputs and outputs for op.
    fn get_mut_inputs_and_outputs(&mut self) -> (&mut [bool], &mut [bool]);

    /// Get the input state for the op.
    fn clone_inputs(&self) -> Self::SubState;

    /// Get the output state for the op.
    fn clone_outputs(&self) -> Self::SubState;
}

/// A node for a loop updater to contain ops.
pub trait OpNode<O: Op> {
    /// Get the contained up
    fn get_op(&self) -> O;
    /// Get a reference to the contained op
    fn get_op_ref(&self) -> &O;
    /// Get a mutable reference to the contained op.
    fn get_op_mut(&mut self) -> &mut O;
}

/// The ability to construct a new OpContainer
pub trait OpContainerConstructor {
    /// Make a new container for nvars.
    fn new(nvars: usize) -> Self;
}

/// Contain and manage ops.
pub trait OpContainer {
    /// The op object to manage.
    type Op: Op;

    /// Get the cutoff for this container.
    fn get_cutoff(&self) -> usize;
    /// Set the cutoff for this container.
    fn set_cutoff(&mut self, cutoff: usize);
    /// Get the number of non-identity ops.
    fn get_n(&self) -> usize;
    /// Get the number of managed variables.
    fn get_nvars(&self) -> usize;
    /// Get the pth op, None is identity.
    fn get_pth(&self, p: usize) -> Option<&Self::Op>;
    /// Verify the integrity of the OpContainer.
    fn verify(&self, state: &[bool]) -> bool {
        let mut rolling_state = state.to_vec();
        for p in 0..self.get_cutoff() {
            let op = self.get_pth(p);
            if let Some(op) = op {
                for (v, inp) in op.get_vars().iter().zip(op.get_inputs().iter()) {
                    if rolling_state[*v] != *inp {
                        return false;
                    }
                }
                op.get_vars()
                    .iter()
                    .zip(op.get_outputs().iter())
                    .for_each(|(v, out)| {
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

/// A hamiltonian for the graph.
#[derive(Debug)]
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
    /// Construct a new hamiltonian with a function, edge lookup function, and the number of bonds.
    pub fn new(hamiltonian: H, edge_fn: E, num_edges: usize) -> Self {
        Hamiltonian {
            hamiltonian,
            edge_fn,
            num_edges,
        }
    }
}

/// Perform diagonal updates to an op container.
pub trait DiagonalUpdater: OpContainer {
    /// Folds across the p values, passing T down. Mutates op if returned values is Some(...)
    fn mutate_ps<F, T>(&mut self, cutoff: usize, t: T, f: F) -> T
    where
        F: Fn(&Self, Option<&Self::Op>, T) -> (Option<Option<Self::Op>>, T);

    /// Iterate through the ops and call f.
    fn iterate_ps<F, T>(&self, t: T, f: F) -> T
    where
        F: Fn(&Self, Option<&Self::Op>, T) -> T,
    {
        let cutoff = self.get_cutoff();
        (0..cutoff).fold(t, |t, p| {
            let op = self.get_pth(p);
            f(&self, op, t)
        })
    }

    /// Perform a diagonal update step with thread rng.
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

    /// Perform a diagonal update step.
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

    /// Perform a diagonal update step using in place edits to state.
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
            let op = metropolis_single_diagonal_update(
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
}

/// Perform a single metropolis update.
fn metropolis_single_diagonal_update<'b, O: Op, H, E, R: Rng>(
    op: Option<&O>,
    cutoff: usize,
    n: usize,
    beta: f64,
    state: &mut [bool],
    hamiltonian: &Hamiltonian<'b, H, E>,
    rng: &mut R,
) -> Option<Option<O>>
where
    H: Fn(&[usize], usize, &[bool], &[bool]) -> f64,
    E: Fn(usize) -> &'b [usize],
{
    let b = match op {
        None => rng.gen_range(0, hamiltonian.num_edges),
        Some(op) if op.is_diagonal() => op.get_bond(),
        Some(op) => {
            op.get_vars()
                .iter()
                .zip(op.get_outputs().iter())
                .for_each(|(v, b)| state[*v] = *b);
            return None;
        }
    };
    let vars = (hamiltonian.edge_fn)(b);
    let substate = vars.iter().map(|v| state[*v]).collect::<O::SubState>();
    let mat_element = (hamiltonian.hamiltonian)(vars, b, substate.as_ref(), substate.as_ref());

    // This is based on equations 19a and 19b of arXiv:1909.10591v1 from 23 Sep 2019
    // or A. W. Sandvik, Phys. Rev. B 59, 14157 (1999)
    let numerator = beta * (hamiltonian.num_edges as f64) * mat_element;
    let denominator = (cutoff - n) as f64;

    match op {
        None => {
            if numerator > denominator || rng.gen_bool(numerator / denominator) {
                let vars = vars.iter().cloned().collect::<O::Vars>();
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

/// Add loop updates to OpContainer.
pub trait LoopUpdater: OpContainer {
    /// The type used to contain the Op and handle movement around the worldlines.
    type Node: OpNode<Self::Op>;

    /// Get a ref to a node at position p
    fn get_node_ref(&self, p: usize) -> Option<&Self::Node>;
    /// Get a mutable ref to the node at position p
    fn get_node_mut(&mut self, p: usize) -> Option<&mut Self::Node>;

    /// Get the first occupied p if it exists.
    fn get_first_p(&self) -> Option<usize>;
    /// Get the last occupied p if it exists.
    fn get_last_p(&self) -> Option<usize>;
    /// Get the first p occupied which covers variable `var`.
    fn get_first_p_for_var(&self, var: usize) -> Option<usize>;
    /// Get the last p occupied which covers variable `var`.
    fn get_last_p_for_var(&self, var: usize) -> Option<usize>;

    /// Get the previous occupied p compared to `node`.
    fn get_previous_p(&self, node: &Self::Node) -> Option<usize>;
    /// Get the next occupied p compared to `node`.
    fn get_next_p(&self, node: &Self::Node) -> Option<usize>;

    /// Get the previous p for a given var, takes the relative var index in node.
    fn get_previous_p_for_rel_var(&self, relvar: usize, node: &Self::Node) -> Option<usize>;
    /// Get the next p for a given var, takes the relative var index in node.
    fn get_next_p_for_rel_var(&self, relvar: usize, node: &Self::Node) -> Option<usize>;

    /// Get the previous p for a given var.
    fn get_previous_p_for_var(&self, var: usize, node: &Self::Node) -> Result<Option<usize>, ()> {
        let relvar = node.get_op_ref().index_of_var(var);
        if let Some(relvar) = relvar {
            Ok(self.get_previous_p_for_rel_var(relvar, node))
        } else {
            Err(())
        }
    }
    /// Get the next p for a given var.
    fn get_next_p_for_var(&self, var: usize, node: &Self::Node) -> Result<Option<usize>, ()> {
        let relvar = node.get_op_ref().index_of_var(var);
        if let Some(relvar) = relvar {
            Ok(self.get_next_p_for_rel_var(relvar, node))
        } else {
            Err(())
        }
    }

    /// Get the nth occupied p.
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

    /// Returns if a given variable is covered by any ops.
    fn does_var_have_ops(&self, var: usize) -> bool {
        self.get_first_p_for_var(var).is_some()
    }

    /// Make a loop update to the graph with thread rng.
    fn make_loop_update<H>(&mut self, initial_n: Option<usize>, hamiltonian: H, state: &mut [bool])
    where
        H: Fn(&[usize], usize, &[bool], &[bool]) -> f64,
    {
        self.make_loop_update_with_rng(initial_n, hamiltonian, state, &mut rand::thread_rng())
    }

    /// Make a loop update to the graph.
    fn make_loop_update_with_rng<H, R: Rng>(
        &mut self,
        initial_n: Option<usize>,
        hamiltonian: H,
        state: &mut [bool],
        rng: &mut R,
    ) where
        H: Fn(&[usize], usize, &[bool], &[bool]) -> f64,
    {
        let h = |op: &Self::Op, entrance: Leg, exit: Leg| -> f64 {
            let mut inputs = op.clone_inputs();
            let mut outputs = op.clone_outputs();
            adjust_states(inputs.as_mut(), outputs.as_mut(), entrance);
            adjust_states(inputs.as_mut(), outputs.as_mut(), exit);
            // Call the supplied hamiltonian.
            hamiltonian(
                &op.get_vars(),
                op.get_bond(),
                inputs.as_ref(),
                outputs.as_ref(),
            )
        };

        if self.get_n() > 0 {
            let initial_n = initial_n
                .map(|n| min(n, self.get_n()))
                .unwrap_or_else(|| rng.gen_range(0, self.get_n()));
            let nth_p = self.get_nth_p(initial_n);
            // Get starting leg for pth op.
            let op = self.get_node_ref(nth_p).unwrap();
            let n_vars = op.get_op_ref().get_vars().len();
            let initial_var = rng.gen_range(0, n_vars);
            let initial_direction = if rng.gen() {
                OpSide::Inputs
            } else {
                OpSide::Outputs
            };
            let initial_leg = (initial_var, initial_direction);

            apply_loop_update(
                self,
                (nth_p, initial_leg),
                nth_p,
                initial_leg,
                h,
                state,
                rng,
            );
        }
    }
}

/// Allow recursive loop updates with a trampoline mechanic
#[derive(Debug, Clone, Copy)]
enum LoopResult {
    Return,
    Iterate(usize, Leg),
}

/// Apply loop update logic (call `make_loop_update` instead)
fn apply_loop_update<L: LoopUpdater + ?Sized, H, R: Rng>(
    l: &mut L,
    initial_op_and_leg: (usize, Leg),
    mut sel_op_pos: usize,
    mut entrance_leg: Leg,
    h: H,
    state: &mut [bool],
    rng: &mut R,
) where
    H: Copy + Fn(&L::Op, Leg, Leg) -> f64,
{
    loop {
        let res = loop_body(
            l,
            initial_op_and_leg,
            sel_op_pos,
            entrance_leg,
            h,
            state,
            rng,
        );
        match res {
            LoopResult::Return => break,
            LoopResult::Iterate(new_sel_op_pos, new_entrance_leg) => {
                sel_op_pos = new_sel_op_pos;
                entrance_leg = new_entrance_leg;
            }
        }
    }
}

/// Apply loop update logic (call `make_loop_update` instead)
fn loop_body<L: LoopUpdater + ?Sized, H, R: Rng>(
    l: &mut L,
    initial_op_and_leg: (usize, Leg),
    sel_op_pos: usize,
    entrance_leg: Leg,
    h: H,
    state: &mut [bool],
    rng: &mut R,
) -> LoopResult
where
    H: Fn(&L::Op, Leg, Leg) -> f64,
{
    let sel_opnode = l.get_node_mut(sel_op_pos).unwrap();
    let sel_op = sel_opnode.get_op();

    let inputs_legs = (0..sel_op.get_vars().len()).map(|v| (v, OpSide::Inputs));
    let outputs_legs = (0..sel_op.get_vars().len()).map(|v| (v, OpSide::Outputs));

    // TODO: Adjust this once const generics allow for more vars on stack.
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
    let mut inputs = sel_opnode.get_op_ref().clone_inputs();
    let mut outputs = sel_opnode.get_op_ref().clone_outputs();
    adjust_states(inputs.as_mut(), outputs.as_mut(), entrance_leg);

    // Change the op now that we passed through.
    let (inputs, outputs) = sel_opnode.get_op_mut().get_mut_inputs_and_outputs();
    adjust_states(inputs, outputs, exit_leg);

    // No longer need mutability.
    let sel_opnode = l.get_node_ref(sel_op_pos).unwrap();
    let sel_op = sel_opnode.get_op_ref();

    // Check if we closed the loop before going to next opnode.
    if (sel_op_pos, exit_leg) == initial_op_and_leg {
        LoopResult::Return
    } else {
        // Get the next opnode and entrance leg, let us know if it changes the initial/final.
        let (next_op_pos, var_to_match) = match exit_leg {
            (var, OpSide::Outputs) => {
                let next_var_op = l.get_next_p_for_rel_var(var, sel_opnode);
                let next = next_var_op.unwrap_or_else(|| {
                    // Adjust the state to reflect new output.
                    state[sel_op.get_vars()[var]] = sel_op.get_outputs()[var];
                    l.get_first_p_for_var(sel_op.get_vars()[var]).unwrap()
                });
                (next, sel_op.get_vars()[var])
            }
            (var, OpSide::Inputs) => {
                let prev_var_op = l.get_previous_p_for_rel_var(var, sel_opnode);
                let next = prev_var_op.unwrap_or_else(|| {
                    // Adjust the state to reflect new input.
                    state[sel_op.get_vars()[var]] = sel_op.get_inputs()[var];
                    l.get_last_p_for_var(sel_op.get_vars()[var]).unwrap()
                });
                (next, sel_op.get_vars()[var])
            }
        };

        let next_node = l.get_node_ref(next_op_pos).unwrap();
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

/// Add cluster updates to LoopUpdater.
pub trait ClusterUpdater: LoopUpdater {
    /// Flip each cluster in the graph using an rng instance, add to state changes in acc.
    fn flip_each_cluster_rng<R: Rng>(&mut self, prob: f64, rng: &mut R, state: &mut [bool]) {
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
    }

    /// Find a site with a single var op.
    fn find_single_site(&self) -> Option<usize> {
        let mut p = self.get_first_p();
        while let Some(node_p) = p {
            let node = self.get_node_ref(node_p).unwrap();
            if node.get_op_ref().get_vars().len() == 1 {
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

        // If we hit a 1-site, add to frontier and mark in boundary.
        if next_node.get_op_ref().get_vars().len() == 1 {
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
            for (var, outp) in op.get_vars().iter().zip(op.get_outputs().iter()) {
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

/// An standard op which covers a number of variables and changes the state from input to output.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct BasicOp<Vars, SubState>
where
    Vars: FromIterator<usize> + AsRef<[usize]> + AsMut<[usize]> + Clone,
    SubState: FromIterator<bool> + AsRef<[bool]> + AsMut<[bool]> + Clone,
{
    /// Variables involved in op
    vars: Vars,
    /// Bond number (index of op)
    bond: usize,
    /// Input state into op.
    inputs: SubState,
    /// Output state out of op.
    outputs: SubState,
}

impl<Vars, SubState> Op for BasicOp<Vars, SubState>
where
    Vars: FromIterator<usize> + AsRef<[usize]> + AsMut<[usize]> + Clone,
    SubState: FromIterator<bool> + AsRef<[bool]> + AsMut<[bool]> + Clone,
{
    type Vars = Vars;
    type SubState = SubState;

    fn diagonal<A, B>(vars: A, bond: usize, state: B) -> Self
    where
        A: Into<Self::Vars>,
        B: Into<Self::SubState>,
    {
        let outputs = state.into();
        Self {
            vars: vars.into(),
            bond,
            inputs: outputs.clone(),
            outputs,
        }
    }

    fn offdiagonal<A, B, C>(vars: A, bond: usize, inputs: B, outputs: C) -> Self
    where
        A: Into<Self::Vars>,
        B: Into<Self::SubState>,
        C: Into<Self::SubState>,
    {
        Self {
            vars: vars.into(),
            bond,
            inputs: inputs.into(),
            outputs: outputs.into(),
        }
    }

    fn index_of_var(&self, var: usize) -> Option<usize> {
        let res = self
            .vars
            .as_ref()
            .iter()
            .enumerate()
            .try_for_each(|(indx, v)| if *v == var { Err(indx) } else { Ok(()) });
        match res {
            Ok(_) => None,
            Err(v) => Some(v),
        }
    }

    fn get_vars(&self) -> &[usize] {
        self.vars.as_ref()
    }

    fn get_bond(&self) -> usize {
        self.bond
    }

    fn get_inputs(&self) -> &[bool] {
        self.inputs.as_ref()
    }

    fn get_outputs(&self) -> &[bool] {
        self.outputs.as_ref()
    }

    fn get_inputs_mut(&mut self) -> &mut [bool] {
        self.inputs.as_mut()
    }

    fn get_outputs_mut(&mut self) -> &mut [bool] {
        self.outputs.as_mut()
    }

    fn get_mut_inputs_and_outputs(&mut self) -> (&mut [bool], &mut [bool]) {
        (self.inputs.as_mut(), self.outputs.as_mut())
    }

    fn clone_inputs(&self) -> Self::SubState {
        self.inputs.clone()
    }

    fn clone_outputs(&self) -> Self::SubState {
        self.outputs.clone()
    }
}
