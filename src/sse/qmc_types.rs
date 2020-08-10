#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};

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
