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
    /// Returns the relative total graph weight of evaluating H1 versus H2: W(H1)/W(H2)
    fn relative_weight_for_hamiltonians<H1, H2>(&self, h1: H1, h2: H2) -> f64
    where
        H1: Fn(&[usize], usize, &[bool], &[bool]) -> f64,
        H2: Fn(&[usize], usize, &[bool], &[bool]) -> f64;

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
        M: OpContainerConstructor + DiagonalUpdater + Into<L> + OpWeights,
        L: ClusterUpdater + Into<M>,
    > OpWeights for QMCGraph<R, M, L>
{
    fn relative_weight_for_hamiltonians<H1, H2>(&self, h1: H1, h2: H2) -> f64
    where
        H1: Fn(&[usize], usize, &[bool], &[bool]) -> f64,
        H2: Fn(&[usize], usize, &[bool], &[bool]) -> f64,
    {
        self.get_manager_ref()
            .relative_weight_for_hamiltonians(h1, h2)
    }

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
        M: OpContainerConstructor + DiagonalUpdater + Into<L>,
        L: ClusterUpdater + Into<M>,
    > OpHam for QMCGraph<R, M, L>
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
        M: OpContainerConstructor + DiagonalUpdater + Into<L>,
        L: ClusterUpdater + Into<M>,
    > StateGetter for QMCGraph<R, M, L>
{
    fn get_state_ref(&self) -> &[bool] {
        self.state_ref()
    }
}

impl<
        R: Rng,
        M: OpContainerConstructor + DiagonalUpdater + Into<L> + StateSetter,
        L: ClusterUpdater + Into<M>,
    > StateSetter for QMCGraph<R, M, L>
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

            for i in 0..op.get_op_ref().get_vars().len() {
                let opref = op.get_op_mut();
                let v = opref.get_vars()[i];
                let neq = opref.get_inputs()[i] != opref.get_outputs()[i];

                let inputs = opref.get_inputs_mut();
                inputs[i] = state[v];
                if neq {
                    state[v] = !state[v];
                };
                let outputs = opref.get_outputs_mut();
                outputs[i] = state[v];
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
                for i in 0..op.get_vars().len() {
                    let v = op.get_vars()[i];
                    let neq = op.get_inputs()[i] != op.get_outputs()[i];
                    op.get_inputs_mut()[i] = state[v];
                    if neq {
                        state[v] = !state[v];
                    };
                    op.get_outputs_mut()[i] = state[v];
                }
                state
            });
    }
}

impl OpWeights for FastOps {
    fn relative_weight_for_hamiltonians<H1, H2>(&self, h1: H1, h2: H2) -> f64
    where
        H1: Fn(&[usize], usize, &[bool], &[bool]) -> f64,
        H2: Fn(&[usize], usize, &[bool], &[bool]) -> f64,
    {
        let mut t = 1.0;
        let mut op_p = self.p_ends.map(|(p, _)| p);
        while let Some(p) = op_p {
            let op = self.get_node_ref(p).unwrap();
            let w1 = h1(
                &op.op.get_vars(),
                op.op.get_bond(),
                &op.op.get_inputs(),
                &op.op.get_outputs(),
            );
            let w2 = h2(
                &op.op.get_vars(),
                op.op.get_bond(),
                &op.op.get_inputs(),
                &op.op.get_outputs(),
            );
            if w1 == 0.0 {
                return 0.0;
            }
            if w2 == 0.0 {
                return std::f64::INFINITY;
            }
            t *= w1 / w2;
            op_p = op.next_p;
        }
        t
    }

    fn relative_weight_for_state<H>(&self, h: H, state: &mut [bool]) -> f64
    where
        H: Fn(&[usize], usize, &[bool], &[bool]) -> f64,
    {
        let max_vars = self
            .ops
            .iter()
            .filter_map(|op| op.as_ref().map(|op| op.op.get_vars().len()))
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

            for i in 0..op.op.get_vars().len() {
                let v = op.op.get_vars()[i];
                inputs[i] = state[v];
                if op.op.get_inputs()[i] != op.op.get_outputs()[i] {
                    state[v] = !state[v];
                };
                outputs[i] = state[v];
            }
            let new_weight = h(&op.op.get_vars(), op.op.get_bond(), &inputs, &outputs);
            if new_weight == 0.0 {
                return 0.0;
            }
            let old_weight = h(
                &op.op.get_vars(),
                op.op.get_bond(),
                &op.op.get_inputs(),
                &op.op.get_outputs(),
            );
            t *= new_weight / old_weight;
            op_p = op.next_p;
        }
        t
    }
}

impl OpWeights for SimpleOpDiagonal {
    fn relative_weight_for_hamiltonians<H1, H2>(&self, h1: H1, h2: H2) -> f64
    where
        H1: Fn(&[usize], usize, &[bool], &[bool]) -> f64,
        H2: Fn(&[usize], usize, &[bool], &[bool]) -> f64,
    {
        let res = self
            .ops
            .iter()
            .filter(|op| op.is_some())
            .map(|op| op.as_ref().unwrap())
            .try_fold(1.0, |t, op| -> Result<f64, f64> {
                let w1 = h1(
                    &op.get_vars(),
                    op.get_bond(),
                    &op.get_inputs(),
                    &op.get_outputs(),
                );
                let w2 = h2(
                    &op.get_vars(),
                    op.get_bond(),
                    &op.get_inputs(),
                    &op.get_outputs(),
                );
                if w1 != 0.0 && w2 != 0.0 {
                    Ok(t * w1 / w2)
                } else if w1 == 0.0 {
                    Err(0.0)
                } else {
                    Err(std::f64::INFINITY)
                }
            });
        match res {
            Ok(f) => f,
            Err(f) => f,
        }
    }

    fn relative_weight_for_state<H>(&self, h: H, state: &mut [bool]) -> f64
    where
        H: Fn(&[usize], usize, &[bool], &[bool]) -> f64,
    {
        let max_vars = self
            .ops
            .iter()
            .filter_map(|op| op.as_ref().map(|op| op.get_vars().len()))
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
                let equality_iter = op
                    .get_inputs()
                    .iter()
                    .zip(op.get_outputs().iter())
                    .map(|(a, b)| a == b);

                op.get_vars()
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
                let new_weight = h(&op.get_vars(), op.get_bond(), &inputs, &outputs);
                if new_weight == 0.0 {
                    Err(())
                } else {
                    let old_weight = h(
                        &op.get_vars(),
                        op.get_bond(),
                        &op.get_inputs(),
                        &op.get_outputs(),
                    );
                    let t = t * (new_weight / old_weight);
                    Ok((t, state))
                }
            })
            .map(|(w, _)| w)
            .unwrap_or(0.0) // 0.0 weight if returned early.
    }
}
