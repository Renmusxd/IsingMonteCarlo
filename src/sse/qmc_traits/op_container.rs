#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::iter::FromIterator;

/// Ops for holding SSE graph state.
pub trait Op: Clone + Debug {
    /// The list of op variables.
    type Vars: FromIterator<usize> + AsRef<[usize]> + AsMut<[usize]> + Debug;
    /// The list of op input and output states.
    type SubState: FromIterator<bool> + AsRef<[bool]> + AsMut<[bool]> + Debug;

    /// Make a diagonal op.
    fn diagonal<A, B>(vars: A, bond: usize, state: B, constant: bool) -> Self
    where
        A: Into<Self::Vars>,
        B: Into<Self::SubState>;

    /// Make an offdiagonal op.
    fn offdiagonal<A, B, C>(vars: A, bond: usize, inputs: B, outputs: C, constant: bool) -> Self
    where
        A: Into<Self::Vars>,
        B: Into<Self::SubState>,
        C: Into<Self::SubState>;

    /// Make vars (this is here mostly due to rust bug 38078)
    fn make_vars<V: Iterator<Item = usize>>(vars: V) -> Self::Vars {
        vars.collect()
    }

    /// Make substate (this is here mostly due to rust bug 38078)
    fn make_substate<S: Iterator<Item = bool>>(state: S) -> Self::SubState {
        state.collect()
    }

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

    /// If the op is always a constant under any bit flip in input or output, then it can be used
    /// to mark the edges of clusters.
    fn is_constant(&self) -> bool;
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
    /// Make a new container for nvars giving a hint as to the number of bonds.
    fn new_with_bonds(nvars: usize, nbonds: usize) -> Self;
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
    /// Gets the count of `bonds` ops in the graph.
    fn get_count(&self, bond: usize) -> usize;
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

/// An standard op which covers a number of variables and changes the state from input to output.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct BasicOp<Vars, SubState>
where
    Vars: FromIterator<usize> + AsRef<[usize]> + AsMut<[usize]> + Clone + Debug,
    SubState: FromIterator<bool> + AsRef<[bool]> + AsMut<[bool]> + Clone + Debug,
{
    /// Variables involved in op
    vars: Vars,
    /// Bond number (index of op)
    bond: usize,
    /// Input state into op.
    inputs: SubState,
    /// Output state out of op.
    outputs: SubState,
    /// Is this op constant under bit flips?
    constant: bool,
}

impl<Vars, SubState> Op for BasicOp<Vars, SubState>
where
    Vars: FromIterator<usize> + AsRef<[usize]> + AsMut<[usize]> + Clone + Debug,
    SubState: FromIterator<bool> + AsRef<[bool]> + AsMut<[bool]> + Clone + Debug,
{
    type Vars = Vars;
    type SubState = SubState;

    fn diagonal<A, B>(vars: A, bond: usize, state: B, constant: bool) -> Self
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
            constant,
        }
    }

    fn offdiagonal<A, B, C>(vars: A, bond: usize, inputs: B, outputs: C, constant: bool) -> Self
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
            constant,
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

    fn is_constant(&self) -> bool {
        self.constant
    }
}
