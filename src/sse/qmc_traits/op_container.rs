#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::iter::FromIterator;

/// Location in imaginary time guarenteed to have an operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct OpIndex {
    index: usize,
}

impl Into<usize> for OpIndex {
    fn into(self) -> usize {
        self.index
    }
}
impl From<usize> for OpIndex {
    fn from(i: usize) -> Self {
        Self { index: i }
    }
}

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
    fn make_vars<V>(vars: V) -> Self::Vars
    where
        V: IntoIterator<Item = usize>,
    {
        vars.into_iter().collect()
    }

    /// Make substate (this is here mostly due to rust bug 38078)
    fn make_substate<S>(state: S) -> Self::SubState
    where
        S: IntoIterator<Item = bool>,
    {
        state.into_iter().collect()
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

    /// Clone the op and edit inputs and outputs.
    fn clone_and_edit_in_out<F>(&self, f: F) -> Self
    where
        F: Fn(&mut [bool], &mut [bool]);

    /// Clone the op and edit inputs and outputs. Must edit states symmetrically (diagonal stays
    /// diagonal, offdiagonal stays offdiagonal).
    fn clone_and_edit_in_out_symmetric<F>(&self, f: F) -> Self
    where
        F: Fn(&mut [bool]);

    /// Edit inputs and outputs.
    fn edit_in_out<F>(&mut self, f: F)
    where
        F: Fn(&mut [bool], &mut [bool]);

    /// Edit inputs and outputs. Must edit states symmetrically (diagonal stays
    /// diagonal, offdiagonal stays offdiagonal).
    fn edit_in_out_symmetric<F>(&mut self, f: F)
    where
        F: Fn(&mut [bool]);

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
    /// Gets the count of `bond` ops in the graph.
    fn get_count(&self, bond: usize) -> usize;

    /// Iterate through the imaginary time states of the opcontainer.
    fn itime_fold<F, T>(&self, state: &mut [bool], fold_fn: F, init: T) -> T
    where
        F: Fn(T, &[bool]) -> T;

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

/// Holds op inputs and outputs as diagonal or offdiagonal.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub enum OpType<SubState>
where
    SubState: Clone + Debug + FromIterator<bool> + AsRef<[bool]> + AsMut<[bool]>,
{
    /// A diagonal op.
    Diagonal(SubState),
    /// An offdiagonal op.
    Offdiagonal(SubState, SubState),
}

impl<SubState> OpType<SubState>
where
    SubState: Clone + Debug + FromIterator<bool> + AsRef<[bool]> + AsMut<[bool]>,
{
    fn edit_states<F>(&mut self, f: F)
    where
        F: Fn(&mut [bool], &mut [bool]),
    {
        let (inputs, outputs) = match self {
            OpType::Diagonal(state) => {
                let mut inputs = state.clone();
                let mut outputs = state.clone();
                f(inputs.as_mut(), outputs.as_mut());
                (inputs, outputs)
            }
            OpType::Offdiagonal(inputs, outputs) => {
                let mut inputs = inputs.clone();
                let mut outputs = outputs.clone();
                f(inputs.as_mut(), outputs.as_mut());
                (inputs, outputs)
            }
        };

        *self = if inputs.as_ref() == outputs.as_ref() {
            Self::Diagonal(inputs)
        } else {
            Self::Offdiagonal(inputs, outputs)
        };
    }

    fn edit_states_symmetric<F>(&mut self, f: F)
    where
        F: Fn(&mut [bool]),
    {
        match self {
            OpType::Diagonal(state) => {
                f(state.as_mut());
            }
            OpType::Offdiagonal(inputs, outputs) => {
                f(inputs.as_mut());
                f(outputs.as_mut());
            }
        };
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
    /// Input and Output state.
    in_out: OpType<SubState>,
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
        Self {
            vars: vars.into(),
            bond,
            in_out: OpType::Diagonal(state.into()),
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
            in_out: OpType::Offdiagonal(inputs.into(), outputs.into()),
            constant,
        }
    }

    fn is_diagonal(&self) -> bool {
        matches!(&self.in_out, OpType::Diagonal(_))
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
        match &self.in_out {
            OpType::Diagonal(state) => state.as_ref(),
            OpType::Offdiagonal(state, _) => state.as_ref(),
        }
    }

    fn get_outputs(&self) -> &[bool] {
        match &self.in_out {
            OpType::Diagonal(state) => state.as_ref(),
            OpType::Offdiagonal(_, state) => state.as_ref(),
        }
    }

    fn clone_and_edit_in_out<F>(&self, f: F) -> Self
    where
        F: Fn(&mut [bool], &mut [bool]),
    {
        let (mut inputs, mut outputs) = match &self.in_out {
            OpType::Diagonal(state) => {
                let inputs = state.clone();
                let outputs = state.clone();
                (inputs, outputs)
            }
            OpType::Offdiagonal(inputs, outputs) => {
                let inputs = inputs.clone();
                let outputs = outputs.clone();
                (inputs, outputs)
            }
        };
        f(inputs.as_mut(), outputs.as_mut());
        let all_eq = inputs.as_ref() == outputs.as_ref();
        let in_out = if all_eq {
            OpType::Diagonal(inputs)
        } else {
            OpType::Offdiagonal(inputs, outputs)
        };
        Self {
            vars: self.vars.clone(),
            bond: self.bond,
            in_out,
            constant: self.constant,
        }
    }

    fn clone_and_edit_in_out_symmetric<F>(&self, f: F) -> Self
    where
        F: Fn(&mut [bool]),
    {
        let in_out = match &self.in_out {
            OpType::Diagonal(state) => {
                let mut inputs = state.clone();
                f(inputs.as_mut());
                OpType::Diagonal(inputs)
            }
            OpType::Offdiagonal(inputs, outputs) => {
                let mut inputs = inputs.clone();
                let mut outputs = outputs.clone();
                f(inputs.as_mut());
                f(outputs.as_mut());
                OpType::Offdiagonal(inputs, outputs)
            }
        };
        Self {
            vars: self.vars.clone(),
            bond: self.bond,
            in_out,
            constant: self.constant,
        }
    }

    fn clone_inputs(&self) -> Self::SubState {
        match &self.in_out {
            OpType::Diagonal(state) => state.clone(),
            OpType::Offdiagonal(state, _) => state.clone(),
        }
    }

    fn clone_outputs(&self) -> Self::SubState {
        match &self.in_out {
            OpType::Diagonal(state) => state.clone(),
            OpType::Offdiagonal(_, state) => state.clone(),
        }
    }

    fn is_constant(&self) -> bool {
        self.constant
    }

    fn edit_in_out<F>(&mut self, f: F)
    where
        F: Fn(&mut [bool], &mut [bool]),
    {
        self.in_out.edit_states(f)
    }

    fn edit_in_out_symmetric<F>(&mut self, f: F)
    where
        F: Fn(&mut [bool]),
    {
        self.in_out.edit_states_symmetric(f)
    }
}
