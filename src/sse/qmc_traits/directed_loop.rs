use crate::memory::allocator::{Factory, StackTuplizer};
use crate::sse::qmc_traits::op_container::*;
use crate::sse::qmc_types::*;
use rand::Rng;
use std::cmp::min;

/// The location in imaginary time (p) and the relative index of the variable.
pub type PRel = (usize, usize);

/// Add loop updates to OpContainer.
pub trait LoopUpdater: OpContainer + Factory<Vec<Leg>> + Factory<Vec<f64>> {
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
    /// Get the first p occupied which covers variable `var`, also returns the relative index.
    fn get_first_p_for_var(&self, var: usize) -> Option<PRel>;
    /// Get the last p occupied which covers variable `var`, also returns the relative index.
    fn get_last_p_for_var(&self, var: usize) -> Option<PRel>;

    /// Get the previous occupied p compared to `node`.
    fn get_previous_p(&self, node: &Self::Node) -> Option<usize>;
    /// Get the next occupied p compared to `node`.
    fn get_next_p(&self, node: &Self::Node) -> Option<usize>;

    /// Get the previous p for a given var, takes the relative var index in node. Also returns the
    /// new relative var index.
    fn get_previous_p_for_rel_var(
        &self,
        relvar: usize,
        node: &Self::Node,
    ) -> Option<(usize, usize)>;
    /// Get the next p for a given var, takes the relative var index in node. Also returns the new
    /// relative var index.
    fn get_next_p_for_rel_var(&self, relvar: usize, node: &Self::Node) -> Option<PRel>;

    /// Get the previous p for a given var.
    fn get_previous_p_for_var(&self, var: usize, node: &Self::Node) -> Result<Option<PRel>, ()> {
        let relvar = node.get_op_ref().index_of_var(var);
        if let Some(relvar) = relvar {
            Ok(self.get_previous_p_for_rel_var(relvar, node))
        } else {
            Err(())
        }
    }
    /// Get the next p for a given var.
    fn get_next_p_for_var(&self, var: usize, node: &Self::Node) -> Result<Option<PRel>, ()> {
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
        self.post_loop_update_hook();
    }

    /// Called after an update.
    fn post_loop_update_hook(&mut self) {}
}

/// Allow recursive loop updates with a trampoline mechanic
#[derive(Debug, Clone, Copy)]
enum LoopResult {
    Return,
    Iterate(usize, Leg),
}

/// Apply loop update logic
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

/// Apply loop update logic.
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
    let mut legs = StackTuplizer::<Leg, f64>::new(l);
    let sel_opnode = l.get_node_mut(sel_op_pos).unwrap();
    let sel_op = sel_opnode.get_op();

    let inputs_legs = (0..sel_op.get_vars().len()).map(|v| (v, OpSide::Inputs));
    let outputs_legs = (0..sel_op.get_vars().len()).map(|v| (v, OpSide::Outputs));

    legs.extend(
        inputs_legs
            .chain(outputs_legs)
            .map(|leg| (leg, h(&sel_op, entrance_leg, leg))),
    );

    let total_weight: f64 = legs.iter().map(|(_, w)| *w).sum();
    let choice = rng.gen_range(0.0, total_weight);
    let exit_leg = legs
        .iter()
        .try_fold(choice, |c, (leg, weight)| {
            if c < *weight {
                Err(*leg)
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
        let (next_p, next_rel) = match exit_leg {
            (var, OpSide::Outputs) => {
                let next_var_op = l.get_next_p_for_rel_var(var, sel_opnode);
                next_var_op.unwrap_or_else(|| {
                    // Adjust the state to reflect new output.
                    state[sel_op.get_vars()[var]] = sel_op.get_outputs()[var];
                    l.get_first_p_for_var(sel_op.get_vars()[var]).unwrap()
                })
            }
            (var, OpSide::Inputs) => {
                let prev_var_op = l.get_previous_p_for_rel_var(var, sel_opnode);
                prev_var_op.unwrap_or_else(|| {
                    // Adjust the state to reflect new input.
                    state[sel_op.get_vars()[var]] = sel_op.get_inputs()[var];
                    l.get_last_p_for_var(sel_op.get_vars()[var]).unwrap()
                })
            }
        };
        let new_entrance_leg = (next_rel, exit_leg.1.reverse());

        legs.dissolve(l);

        // If back where we started, close loop and return state changes.
        if (next_p, new_entrance_leg) == initial_op_and_leg {
            LoopResult::Return
        } else {
            LoopResult::Iterate(next_p, new_entrance_leg)
        }
    }
}
