use smallvec::SmallVec;
use std::ops::IndexMut;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Op {
    pub vars: SmallVec<[usize; 2]>,
    pub bond: usize,
    pub inputs: SmallVec<[bool; 2]>,
    pub outputs: SmallVec<[bool; 2]>,
}

impl Op {
    pub fn diagonal<A, B>(vars: A, bond: usize, state: B) -> Self
    where
        A: Into<SmallVec<[usize; 2]>>,
        B: Into<SmallVec<[bool; 2]>>,
    {
        let outputs = state.into();
        Self {
            vars: vars.into(),
            bond,
            inputs: outputs.clone(),
            outputs,
        }
    }

    pub fn offdiagonal<A, B, C>(vars: A, bond: usize, inputs: B, outputs: C) -> Self
    where
        A: Into<SmallVec<[usize; 2]>>,
        B: Into<SmallVec<[bool; 2]>>,
        C: Into<SmallVec<[bool; 2]>>,
    {
        Self {
            vars: vars.into(),
            bond,
            inputs: inputs.into(),
            outputs: outputs.into(),
        }
    }

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

    pub fn is_diagonal(&self) -> bool {
        self.inputs == self.outputs
    }
}

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub enum OpSide {
    Inputs,
    Outputs,
}

impl OpSide {
    pub fn reverse(self) -> Self {
        match self {
            OpSide::Inputs => OpSide::Outputs,
            OpSide::Outputs => OpSide::Inputs,
        }
    }
}

pub type Leg = (usize, OpSide);

pub fn adjust_states<V>(mut before: V, mut after: V, leg: Leg) -> (V, V)
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
