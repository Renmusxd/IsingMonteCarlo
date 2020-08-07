use crate::sse::fast_ops::FastOps;
use crate::sse::qmc_graph::QMCGraph;
use crate::sse::qmc_traits::*;
use crate::sse::simple_ops::SimpleOpDiagonal;
use rand::Rng;
use smallvec::{smallvec, SmallVec};

/// Allows retrieving states.
pub trait StateGetter {
    /// Get the state of the instance.
    fn get_state_ref(&self) -> &[bool];
}

/// Allows setting states.
pub trait StateSetter {
    /// Set the state of the instance.
    fn set_state(&mut self, state: Vec<bool>);
}

/// Allow getting the relative weight for a state compared to default.
pub trait OpWeights {
    /// Get the relative weight for a state compared to the current one.
    fn relative_weight_for_state<H>(&self, h: H, state: &mut [bool]) -> f64
    where
        H: Fn(&[usize], usize, &[bool], &[bool]) -> f64;
    /// Get the relative weight but skip finding the op with maximum vars.
    fn relative_weight_for_state_with_max_vars<H>(
        &self,
        h: H,
        state: &mut [bool],
        max_vars: usize,
    ) -> f64
    where
        H: Fn(&[usize], usize, &[bool], &[bool]) -> f64;
}

/// Operator with a hamiltonian.
pub trait OpHam {
    /// Take an input state and output state for a given set of variables and bond, output the
    /// value of <a|H|b>.
    fn hamiltonian(
        &self,
        vars: &[usize],
        bond: usize,
        input_state: &[bool],
        output_state: &[bool],
    ) -> f64;
}

impl<
        R: Rng,
        N: OpNode,
        M: OpContainerConstructor + DiagonalUpdater + Into<L> + OpWeights,
        L: LoopUpdater<N> + ClusterUpdater<N> + Into<M>,
    > OpWeights for QMCGraph<R, N, M, L>
{
    fn relative_weight_for_state<H>(&self, h: H, state: &mut [bool]) -> f64
    where
        H: Fn(&[usize], usize, &[bool], &[bool]) -> f64,
    {
        self.get_manager_ref().relative_weight_for_state(h, state)
    }

    fn relative_weight_for_state_with_max_vars<H>(
        &self,
        h: H,
        state: &mut [bool],
        max_vars: usize,
    ) -> f64
    where
        H: Fn(&[usize], usize, &[bool], &[bool]) -> f64,
    {
        self.get_manager_ref()
            .relative_weight_for_state_with_max_vars(h, state, max_vars)
    }
}

impl<
        R: Rng,
        N: OpNode,
        M: OpContainerConstructor + DiagonalUpdater + Into<L>,
        L: LoopUpdater<N> + ClusterUpdater<N> + Into<M>,
    > OpHam for QMCGraph<R, N, M, L>
{
    fn hamiltonian(
        &self,
        vars: &[usize],
        bond: usize,
        input_state: &[bool],
        output_state: &[bool],
    ) -> f64 {
        let haminfo = self.make_haminfo();
        Self::hamiltonian(&haminfo, vars, bond, input_state, output_state)
    }
}

impl<
        R: Rng,
        N: OpNode,
        M: OpContainerConstructor + DiagonalUpdater + Into<L>,
        L: LoopUpdater<N> + ClusterUpdater<N> + Into<M>,
    > StateGetter for QMCGraph<R, N, M, L>
{
    fn get_state_ref(&self) -> &[bool] {
        self.state_ref()
    }
}

impl<
        R: Rng,
        N: OpNode,
        M: OpContainerConstructor + DiagonalUpdater + Into<L> + StateSetter,
        L: LoopUpdater<N> + ClusterUpdater<N> + Into<M>,
    > StateSetter for QMCGraph<R, N, M, L>
{
    fn set_state(&mut self, state: Vec<bool>) {
        assert_eq!(self.state_ref().len(), state.len());
        self.state_mut()
            .iter_mut()
            .zip(state.iter())
            .for_each(|(o_s, n_s)| {
                *o_s = *n_s;
            });
        // Ensure all ops are updated
        self.get_manager_mut().set_state(state)
    }
}

impl StateSetter for FastOps {
    fn set_state(&mut self, state: Vec<bool>) {
        let mut state = state;

        let mut op_p = self.p_ends.map(|(p, _)| p);
        while let Some(p) = op_p {
            let op = self.get_node_mut(p).unwrap();

            for i in 0..op.op.vars.len() {
                let v = op.op.vars[i];
                let neq = op.op.inputs[i] != op.op.outputs[i];
                op.op.inputs[i] = state[v];
                if neq {
                    state[v] = !state[v];
                };
                op.op.outputs[i] = state[v];
            }
            op_p = op.next_p;
        }
    }
}

impl StateSetter for SimpleOpDiagonal {
    fn set_state(&mut self, state: Vec<bool>) {
        self.ops
            .iter_mut()
            .filter(|op| op.is_some())
            .map(|op| op.as_mut().unwrap())
            .fold(state, |mut state, op| {
                for i in 0..op.vars.len() {
                    let v = op.vars[i];
                    let neq = op.inputs[i] != op.outputs[i];
                    op.inputs[i] = state[v];
                    if neq {
                        state[v] = !state[v];
                    };
                    op.outputs[i] = state[v];
                }
                state
            });
    }
}

impl OpWeights for FastOps {
    fn relative_weight_for_state<H>(&self, h: H, state: &mut [bool]) -> f64
    where
        H: Fn(&[usize], usize, &[bool], &[bool]) -> f64,
    {
        let max_vars = self
            .ops
            .iter()
            .filter_map(|op| op.as_ref().map(|op| op.op.vars.len()))
            .max();
        if let Some(max_vars) = max_vars {
            self.relative_weight_for_state_with_max_vars(h, state, max_vars)
        } else {
            1.0
        }
    }

    fn relative_weight_for_state_with_max_vars<H>(
        &self,
        h: H,
        state: &mut [bool],
        max_vars: usize,
    ) -> f64
    where
        H: Fn(&[usize], usize, &[bool], &[bool]) -> f64,
    {
        let mut t = 1.0;
        let mut inputs: SmallVec<[bool; 2]> = smallvec!(false; max_vars);
        let mut outputs: SmallVec<[bool; 2]> = smallvec!(false; max_vars);

        let mut op_p = self.p_ends.map(|(p, _)| p);
        while let Some(p) = op_p {
            let op = self.get_node_ref(p).unwrap();

            for i in 0..op.op.vars.len() {
                let v = op.op.vars[i];
                inputs[i] = state[v];
                if op.op.inputs[i] != op.op.outputs[i] {
                    state[v] = !state[v];
                };
                outputs[i] = state[v];
            }
            let new_weight = h(&op.op.vars, op.op.bond, &inputs, &outputs);
            if new_weight == 0.0 {
                return 0.0;
            }
            let old_weight = h(&op.op.vars, op.op.bond, &op.op.inputs, &op.op.outputs);
            t *= new_weight / old_weight;
            op_p = op.next_p;
        }
        t
    }
}

impl OpWeights for SimpleOpDiagonal {
    fn relative_weight_for_state<H>(&self, h: H, state: &mut [bool]) -> f64
    where
        H: Fn(&[usize], usize, &[bool], &[bool]) -> f64,
    {
        let max_vars = self
            .ops
            .iter()
            .filter_map(|op| op.as_ref().map(|op| op.vars.len()))
            .max();
        if let Some(max_vars) = max_vars {
            self.relative_weight_for_state_with_max_vars(h, state, max_vars)
        } else {
            1.0
        }
    }

    fn relative_weight_for_state_with_max_vars<H>(
        &self,
        h: H,
        state: &mut [bool],
        max_vars: usize,
    ) -> f64
    where
        H: Fn(&[usize], usize, &[bool], &[bool]) -> f64,
    {
        let mut inputs: SmallVec<[bool; 2]> = smallvec!(false; max_vars);
        let mut outputs: SmallVec<[bool; 2]> = smallvec!(false; max_vars);

        self.ops
            .iter()
            .filter(|op| op.is_some())
            .map(|op| op.as_ref().unwrap())
            .try_fold((1.0, state), |(t, state), op| {
                let equality_iter = op.inputs.iter().zip(op.outputs.iter()).map(|(a, b)| a == b);

                op.vars
                    .iter()
                    .cloned()
                    .zip(equality_iter)
                    .zip(inputs.iter_mut())
                    .zip(outputs.iter_mut())
                    .for_each(|(((v, eq), input), output)| {
                        *input = state[v];
                        if !eq {
                            state[v] = !state[v];
                        };
                        *output = state[v];
                    });
                let new_weight = h(&op.vars, op.bond, &inputs, &outputs);
                if new_weight == 0.0 {
                    Err(())
                } else {
                    let old_weight = h(&op.vars, op.bond, &op.inputs, &op.outputs);
                    let t = t * (new_weight / old_weight);
                    Ok((t, state))
                }
            })
            .map(|(w, _)| w)
            .unwrap_or(0.0) // 0.0 weight if returned early.
    }
}
