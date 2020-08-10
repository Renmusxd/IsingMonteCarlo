use crate::sse::arena::*;
use crate::sse::qmc_traits::*;
#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

/// A simple implementation of a diagonal op container.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct SimpleOpDiagonal {
    pub(crate) ops: Vec<Option<SimpleOp>>,
    n: usize,
    nvars: usize,
    arena: Arena<Option<usize>>,
}

impl SimpleOpDiagonal {
    /// Set the minimum size of the container.
    pub(crate) fn set_min_size(&mut self, n: usize) {
        if self.ops.len() < n {
            self.ops.resize(n, None)
        }
    }

    /// Debug print the container.
    pub fn debug_print<H>(&self, h: H)
    where
        H: Fn(&[usize], usize, &[bool], &[bool]) -> f64,
    {
        fn lines_for(a: usize, b: usize) {
            (a..b).for_each(|_| {
                print!("| ");
            });
        }

        print!("    \t");
        lines_for(0, self.nvars);
        println!();
        let empty: [usize; 0] = [];
        self.ops.iter().enumerate().for_each(|(p, op)| {
            let (vars, d): (&[usize], bool) = match op {
                Some(op) => (&op.get_vars(), op.is_diagonal()),
                None => (&empty, false),
            };
            print!("{:>5}\t", p);
            let last = vars.iter().fold(0, |acc, v| {
                lines_for(acc, *v);
                if d {
                    print!("O ");
                } else {
                    print!("X ");
                }
                *v + 1
            });
            lines_for(last, self.nvars);
            match op {
                Some(op) => println!(
                    "\t{:?}\t{}",
                    op,
                    h(
                        &op.get_vars(),
                        op.get_bond(),
                        &op.get_inputs(),
                        &op.get_outputs()
                    )
                ),
                None => println!(),
            };
        })
    }

    /// Set the pth op.
    pub fn set_pth(&mut self, p: usize, op: Option<SimpleOp>) -> Option<SimpleOp> {
        self.set_min_size(p + 1);
        let temp = self.ops[p].take();
        self.ops[p] = op;
        temp
    }
}

impl OpContainerConstructor for SimpleOpDiagonal {
    fn new(nvars: usize) -> Self {
        Self {
            ops: vec![],
            n: 0,
            nvars,
            arena: Arena::new(None),
        }
    }
}

impl OpContainer for SimpleOpDiagonal {
    type Op = SimpleOp;

    fn get_cutoff(&self) -> usize {
        self.ops.len()
    }

    fn set_cutoff(&mut self, cutoff: usize) {
        self.set_min_size(cutoff)
    }

    fn get_n(&self) -> usize {
        self.n
    }

    fn get_nvars(&self) -> usize {
        self.nvars
    }

    fn get_pth(&self, p: usize) -> Option<&Self::Op> {
        if p >= self.ops.len() {
            None
        } else {
            self.ops[p].as_ref()
        }
    }
}

impl DiagonalUpdater for SimpleOpDiagonal {
    fn mutate_ps<F, T>(&mut self, cutoff: usize, t: T, f: F) -> T
    where
        F: Fn(&Self, Option<&Self::Op>, T) -> (Option<Option<Self::Op>>, T),
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
}

/// An op node for the simple op container.
#[derive(Clone, Debug)]
pub struct SimpleOpNode {
    pub(crate) op: SimpleOp,
    pub(crate) previous_p: Option<usize>,
    pub(crate) next_p: Option<usize>,
    pub(crate) previous_for_vars: ArenaIndex,
    pub(crate) next_for_vars: ArenaIndex,
}

impl SimpleOpNode {
    fn new(op: SimpleOp, previous_for_vars: ArenaIndex, next_for_vars: ArenaIndex) -> Self {
        let nvars = op.get_vars().len();
        assert_eq!(previous_for_vars.size(), nvars);
        assert_eq!(next_for_vars.size(), nvars);
        Self {
            op,
            previous_p: None,
            next_p: None,
            previous_for_vars,
            next_for_vars,
        }
    }
}

impl OpNode<SimpleOp> for SimpleOpNode {
    fn get_op(&self) -> SimpleOp {
        self.op.clone()
    }

    fn get_op_ref(&self) -> &SimpleOp {
        &self.op
    }

    fn get_op_mut(&mut self) -> &mut SimpleOp {
        &mut self.op
    }
}

/// A simple implementation of the linked list op container for loop updates.
#[derive(Debug)]
pub struct SimpleOpLooper {
    ops: Vec<Option<SimpleOpNode>>,
    nth_ps: Vec<usize>,
    p_ends: Option<(usize, usize)>,
    var_ends: Vec<Option<(usize, usize)>>,
    arena: Arena<Option<usize>>,
}

impl OpContainer for SimpleOpLooper {
    type Op = SimpleOp;

    fn get_cutoff(&self) -> usize {
        self.ops.len()
    }

    fn set_cutoff(&mut self, cutoff: usize) {
        if cutoff > self.ops.len() {
            self.ops.resize(cutoff, None);
        }
    }

    fn get_n(&self) -> usize {
        self.nth_ps.len()
    }

    fn get_nvars(&self) -> usize {
        self.var_ends.len()
    }

    fn get_pth(&self, p: usize) -> Option<&Self::Op> {
        self.ops[p].as_ref().map(|opnode| &opnode.op)
    }
}

impl LoopUpdater for SimpleOpLooper {
    type Node = SimpleOpNode;

    fn get_node_ref(&self, p: usize) -> Option<&SimpleOpNode> {
        self.ops[p].as_ref()
    }

    fn get_node_mut(&mut self, p: usize) -> Option<&mut SimpleOpNode> {
        self.ops[p].as_mut()
    }

    fn get_first_p(&self) -> Option<usize> {
        self.p_ends.map(|(p, _)| p)
    }

    fn get_last_p(&self) -> Option<usize> {
        self.p_ends.map(|(_, p)| p)
    }

    fn get_first_p_for_var(&self, var: usize) -> Option<usize> {
        self.var_ends[var].map(|(p, _)| p)
    }

    fn get_last_p_for_var(&self, var: usize) -> Option<usize> {
        self.var_ends[var].map(|(_, p)| p)
    }

    fn get_previous_p(&self, node: &SimpleOpNode) -> Option<usize> {
        node.previous_p
    }

    fn get_next_p(&self, node: &SimpleOpNode) -> Option<usize> {
        node.next_p
    }

    fn get_previous_p_for_rel_var(&self, revar: usize, node: &SimpleOpNode) -> Option<usize> {
        self.arena[&node.previous_for_vars][revar]
    }

    fn get_next_p_for_rel_var(&self, revar: usize, node: &SimpleOpNode) -> Option<usize> {
        self.arena[&node.next_for_vars][revar]
    }

    fn get_nth_p(&self, n: usize) -> usize {
        self.nth_ps[n]
    }
}

impl ClusterUpdater for SimpleOpLooper {}

impl Into<SimpleOpDiagonal> for SimpleOpLooper {
    fn into(self) -> SimpleOpDiagonal {
        let n = self.get_n();
        let nvars = self.get_nvars();
        let ops = self
            .ops
            .into_iter()
            .map(|opnode| opnode.map(|opnode| opnode.op))
            .collect();
        let mut arena = self.arena;
        arena.clear();
        SimpleOpDiagonal {
            ops,
            n,
            nvars,
            arena,
        }
    }
}

impl Into<SimpleOpLooper> for SimpleOpDiagonal {
    fn into(self) -> SimpleOpLooper {
        let mut p_ends = None;
        let mut var_ends = vec![None; self.nvars];
        let mut arena = self.arena;
        let mut opnodes = self
            .ops
            .iter()
            .map(|op| {
                op.clone().map(|op| {
                    let previous_slice = arena.get_alloc(op.get_vars().len());
                    let next_slice = arena.get_alloc(op.get_vars().len());
                    SimpleOpNode::new(op, previous_slice, next_slice)
                })
            })
            .collect::<Vec<_>>();

        let nth_ps = self
            .ops
            .iter()
            .enumerate()
            .filter_map(|(n, op)| op.as_ref().map(|op| (n, op)))
            .map(|(p, op)| {
                match p_ends {
                    None => p_ends = Some((p, p)),
                    Some((_, last_p)) => {
                        let last_op = opnodes[last_p].as_mut().unwrap();
                        last_op.next_p = Some(p);
                        p_ends.as_mut().unwrap().1 = p;
                        let this_opnode = opnodes[p].as_mut().unwrap();
                        this_opnode.previous_p = Some(last_p);
                    }
                }
                op.get_vars()
                    .iter()
                    .cloned()
                    .enumerate()
                    .for_each(|(indx, v)| match var_ends.get(v) {
                        Some(None) => var_ends[v] = Some((p, p)),
                        Some(Some((_, last_p))) => {
                            let last_p = *last_p;
                            let last_op = opnodes[last_p].as_mut().unwrap();
                            let last_relvar = last_op.op.index_of_var(v).unwrap();
                            arena[&last_op.next_for_vars][last_relvar] = Some(p);
                            var_ends[v].as_mut().unwrap().1 = p;
                            let this_opnode = opnodes[p].as_mut().unwrap();
                            arena[&this_opnode.previous_for_vars][indx] = Some(last_p);
                        }
                        None => unreachable!(),
                    });
                p
            })
            .collect();

        SimpleOpLooper {
            ops: opnodes,
            nth_ps,
            p_ends,
            var_ends,
            arena,
        }
    }
}

/// An op which covers a number of variables and changes the state from input to output.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct SimpleOp {
    /// Variables involved in op
    vars: SmallVec<[usize; 2]>,
    /// Bond number (index of op)
    bond: usize,
    /// Input state into op.
    inputs: SmallVec<[bool; 2]>,
    /// Output state out of op.
    outputs: SmallVec<[bool; 2]>,
}

impl Op for SimpleOp {
    type Vars = SmallVec<[usize; 2]>;
    type SubState = SmallVec<[bool; 2]>;

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

    fn is_diagonal(&self) -> bool {
        self.inputs == self.outputs
    }

    fn get_vars(&self) -> &[usize] {
        &self.vars
    }

    fn get_bond(&self) -> usize {
        self.bond
    }

    fn get_inputs(&self) -> &[bool] {
        &self.inputs
    }

    fn get_outputs(&self) -> &[bool] {
        &self.outputs
    }

    fn get_inputs_mut(&mut self) -> &mut [bool] {
        &mut self.inputs
    }

    fn get_outputs_mut(&mut self) -> &mut [bool] {
        &mut self.outputs
    }

    fn get_mut_inputs_and_outputs(&mut self) -> (&mut [bool], &mut [bool]) {
        (&mut self.inputs, &mut self.outputs)
    }

    fn clone_inputs(&self) -> Self::SubState {
        self.inputs.clone()
    }

    fn clone_outputs(&self) -> Self::SubState {
        self.outputs.clone()
    }
}
