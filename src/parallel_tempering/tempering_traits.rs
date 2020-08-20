use crate::sse::fast_ops::FastOps;
use crate::sse::qmc_ising::QMCIsingGraph;
use crate::sse::qmc_runner::ManagerRef;
use crate::sse::qmc_runner::QMC;
use crate::sse::qmc_traits::*;
use crate::sse::simple_ops::SimpleOpDiagonal;
use crate::sse::ClassicalLoopUpdater;
use rand::Rng;

/// Allows QMC objects to swap internal state and op managers
pub trait SwapManagers {
    /// Checks if graphs can be swapped. Transitive/Commutative properties apply.
    fn can_swap_graphs(&self, other: &Self) -> bool;

    /// Swap op graphs with another struct.
    fn swap_graphs(&mut self, other: &mut Self);

    /// Get the cutoff of the manager.
    fn get_op_cutoff(&self) -> usize;

    /// Set the cutoff of the manager.
    fn set_op_cutoff(&mut self, cutoff: usize);
}

/// Allows retrieving states.
pub trait StateGetter {
    /// Get the state of the instance.
    fn get_state_ref(&self) -> &[bool];
}

/// Allow getting the relative weight for a state compared to default.
pub trait OpWeights {
    /// Returns the relative total graph weight of evaluating H1 versus H2: W(H1)/W(H2)
    fn relative_weight_for_hamiltonians<H1, H2>(&self, h1: H1, h2: H2) -> f64
    where
        H1: Fn(&[usize], usize, &[bool], &[bool]) -> f64,
        H2: Fn(&[usize], usize, &[bool], &[bool]) -> f64;
}

/// Operator with a hamiltonian.
pub trait OpHam {
    /// Return true if the hamiltonians are equal.
    fn ham_eq(&self, other: &Self) -> bool;

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

impl<R, M, L> OpWeights for QMC<R, M, L>
where
    R: Rng,
    M: OpContainerConstructor + DiagonalUpdater + Into<L> + OpWeights,
    L: ClusterUpdater + Into<M> + OpWeights,
{
    fn relative_weight_for_hamiltonians<H1, H2>(&self, h1: H1, h2: H2) -> f64
    where
        H1: Fn(&[usize], usize, &[bool], &[bool]) -> f64,
        H2: Fn(&[usize], usize, &[bool], &[bool]) -> f64,
    {
        match self.get_manager_ref() {
            ManagerRef::DIAGONAL(a) => a.relative_weight_for_hamiltonians(h1, h2),
            ManagerRef::LOOPER(b) => b.relative_weight_for_hamiltonians(h1, h2),
        }
    }
}

impl<R, M, L> SwapManagers for QMC<R, M, L>
where
    R: Rng,
    M: OpContainerConstructor + DiagonalUpdater + Into<L> + OpWeights,
    L: ClusterUpdater + Into<M>,
{
    fn can_swap_graphs(&self, other: &Self) -> bool {
        self.can_swap_managers(other)
    }

    fn swap_graphs(&mut self, other: &mut Self) {
        self.swap_manager_and_state(other)
    }

    fn get_op_cutoff(&self) -> usize {
        self.get_cutoff()
    }

    fn set_op_cutoff(&mut self, cutoff: usize) {
        self.set_cutoff(cutoff);
    }
}

impl<R, M, L> SwapManagers for QMCIsingGraph<R, M, L>
where
    R: Rng,
    M: OpContainerConstructor + ClassicalLoopUpdater + Into<L> + OpWeights,
    L: ClusterUpdater + Into<M>,
{
    fn can_swap_graphs(&self, other: &Self) -> bool {
        self.can_swap_managers(other)
    }

    fn swap_graphs(&mut self, other: &mut Self) {
        self.swap_manager_and_state(other)
    }

    fn get_op_cutoff(&self) -> usize {
        self.get_cutoff()
    }

    fn set_op_cutoff(&mut self, cutoff: usize) {
        self.set_cutoff(cutoff);
    }
}

impl<R, M, L> OpWeights for QMCIsingGraph<R, M, L>
where
    R: Rng,
    M: OpContainerConstructor + ClassicalLoopUpdater + Into<L> + OpWeights,
    L: ClusterUpdater + Into<M>,
{
    fn relative_weight_for_hamiltonians<H1, H2>(&self, h1: H1, h2: H2) -> f64
    where
        H1: Fn(&[usize], usize, &[bool], &[bool]) -> f64,
        H2: Fn(&[usize], usize, &[bool], &[bool]) -> f64,
    {
        self.get_manager_ref()
            .relative_weight_for_hamiltonians(h1, h2)
    }
}

impl<
        'a,
        R: Rng,
        M: OpContainerConstructor + ClassicalLoopUpdater + Into<L>,
        L: ClusterUpdater + Into<M>,
    > OpHam for QMCIsingGraph<R, M, L>
{
    fn ham_eq(&self, other: &Self) -> bool {
        self.make_haminfo() == other.make_haminfo()
    }

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
        M: OpContainerConstructor + ClassicalLoopUpdater + Into<L>,
        L: ClusterUpdater + Into<M>,
    > StateGetter for QMCIsingGraph<R, M, L>
{
    fn get_state_ref(&self) -> &[bool] {
        self.state_ref()
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
}
