#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Op {
    pub vars: Vec<usize>,
    pub bond: usize,
    pub inputs: Vec<bool>,
    pub outputs: Vec<bool>,
}

impl Op {
    pub fn diagonal(vars: Vec<usize>, bond: usize, state: Vec<bool>) -> Self {
        Self {
            vars,
            bond,
            inputs: state.clone(),
            outputs: state,
        }
    }

    pub fn offdiagonal(
        vars: Vec<usize>,
        bond: usize,
        inputs: Vec<bool>,
        outputs: Vec<bool>,
    ) -> Self {
        Self {
            vars,
            bond,
            inputs,
            outputs,
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

pub fn adjust_states(
    mut before: Vec<bool>,
    mut after: Vec<bool>,
    leg: Leg,
) -> (Vec<bool>, Vec<bool>) {
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
