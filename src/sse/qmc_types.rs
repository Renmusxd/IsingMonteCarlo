#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use std::ops::IndexMut;

type Vars = SmallVec<[usize; 2]>;
type SubState = SmallVec<[bool; 2]>;

/// An op which covers a number of variables and changes the state from input to output.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct Op {
    /// Variables involved in op
    pub vars: Vars,
    /// Bond number (index of op)
    pub bond: usize,
    /// Input state into op.
    pub inputs: SubState,
    /// Output state out of op.
    pub outputs: SubState,
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
pub(crate) fn adjust_states<V>(mut before: V, mut after: V, leg: Leg) -> (V, V)
where
    V: IndexMut<usize, Output = bool>,
{
    match leg {
        (var, OpSide::Inputs) => {
            before[var] = !before[var];
        }
        (var, OpSide::Outputs) => {
            after[var] = !after[var];
        }
    };
    (before, after)
}
