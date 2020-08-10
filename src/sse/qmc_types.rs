#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

/// The list of op variables.
pub type Vars = SmallVec<[usize; 2]>;
/// The list of op input and output states.
pub type SubState = SmallVec<[bool; 2]>;

/// An op which covers a number of variables and changes the state from input to output.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct Op {
    /// Variables involved in op
    vars: Vars,
    /// Bond number (index of op)
    bond: usize,
    /// Input state into op.
    inputs: SubState,
    /// Output state out of op.
    outputs: SubState,
}

impl Op {
    /// Make a diagonal op.
    pub fn diagonal<A, B>(vars: A, bond: usize, state: B) -> Self
    where
        A: Into<Vars>,
        B: Into<SubState>,
    {
        let outputs = state.into();
        Self {
            vars: vars.into(),
            bond,
            inputs: outputs.clone(),
            outputs,
        }
    }

    /// Make an offdiagonal op.
    pub fn offdiagonal<A, B, C>(vars: A, bond: usize, inputs: B, outputs: C) -> Self
    where
        A: Into<Vars>,
        B: Into<SubState>,
        C: Into<SubState>,
    {
        Self {
            vars: vars.into(),
            bond,
            inputs: inputs.into(),
            outputs: outputs.into(),
        }
    }

    /// Get the relative index of a variable.
    pub fn index_of_var(&self, var: usize) -> Option<usize> {
        let res =
            self.vars
                .iter()
                .enumerate()
                .try_for_each(|(indx, v)| if *v == var { Err(indx) } else { Ok(()) });
        match res {
            Ok(_) => None,
            Err(v) => Some(v),
        }
    }

    /// Check if the op is diagonal (makes no state changes).
    pub fn is_diagonal(&self) -> bool {
        self.inputs == self.outputs
    }

    /// Get the set of variables used for this op.
    pub fn get_vars(&self) -> &[usize] {
        &self.vars
    }

    /// Get the associated bond number for the op.
    pub fn get_bond(&self) -> usize {
        self.bond
    }

    /// Get the input state for the op.
    pub fn get_inputs(&self) -> &[bool] {
        &self.inputs
    }

    /// Get the output state for the op.
    pub fn get_outputs(&self) -> &[bool] {
        &self.outputs
    }

    /// Get the input state for the op.
    pub fn get_inputs_mut(&mut self) -> &mut [bool] {
        &mut self.inputs
    }

    /// Get the output state for the op.
    pub fn get_outputs_mut(&mut self) -> &mut [bool] {
        &mut self.outputs
    }

    /// Get both the inputs and outputs for op.
    pub fn get_mut_inputs_and_outputs(&mut self) -> (&mut [bool], &mut [bool]) {
        (&mut self.inputs, &mut self.outputs)
    }

    /// Get the input state for the op.
    pub fn clone_inputs(&self) -> SubState {
        self.inputs.clone()
    }

    /// Get the output state for the op.
    pub fn clone_outputs(&self) -> SubState {
        self.outputs.clone()
    }
}

/// Enum detailing which side of the op, input or output.
#[derive(PartialEq, Eq, Clone, Copy, Debug)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub enum OpSide {
    /// <p side
    Inputs,
    /// >p side
    Outputs,
}

impl OpSide {
    /// Swap from Inputs to Outputs
    pub fn reverse(self) -> Self {
        match self {
            OpSide::Inputs => OpSide::Outputs,
            OpSide::Outputs => OpSide::Inputs,
        }
    }
}

/// A leg is a relative variable on a given side of the op.
pub(crate) type Leg = (usize, OpSide);

/// Toggle input or output states at the location given by `leg`.
pub(crate) fn adjust_states(before: &mut [bool], after: &mut [bool], leg: Leg) {
    match leg {
        (var, OpSide::Inputs) => {
            before[var] = !before[var];
        }
        (var, OpSide::Outputs) => {
            after[var] = !after[var];
        }
    };
}
