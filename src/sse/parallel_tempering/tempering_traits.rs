use crate::sse::fast_ops::FastOps;
use crate::sse::qmc_ising::QMCIsingGraph;
use crate::sse::qmc_runner::ManagerRef;
use crate::sse::qmc_runner::QMC;
use crate::sse::qmc_traits::*;
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
pub trait GraphWeights {
    /// Whether the hams are equal. ham_eq => relative_weight=1.0
    fn ham_eq(&self, other: &Self) -> bool;

    /// Returns the relative total graph weight of evaluating the hamiltonian from the other graph
    /// compared to the default one.
    fn relative_weight(&self, h: &Self) -> f64;
}

/// Allow getting the relative weight for a state compared to default.
pub trait OpWeights {
    /// Returns the relative total graph weight of evaluating H1 versus H2: W(H1)/W(H2)
    fn relative_weight_for_hamiltonians<H1, H2>(&self, h1: H1, h2: H2) -> f64
    where
        H1: Fn(&[usize], usize, &[bool], &[bool]) -> f64,
        H2: Fn(&[usize], usize, &[bool], &[bool]) -> f64;
}

impl<'a, R, M, L> GraphWeights for QMC<R, M, L>
where
    R: Rng,
    M: OpContainerConstructor + DiagonalUpdater + Into<L> + OpWeights,
    L: ClusterUpdater + Into<M> + OpWeights,
{
    fn ham_eq(&self, other: &Self) -> bool {
        self.get_bonds() == other.get_bonds()
    }

    fn relative_weight(&self, h: &Self) -> f64 {
        let ha = |_vars: &[usize], bond: usize, input_state: &[bool], output_state: &[bool]| {
            h.get_bonds()[bond].at(input_state, output_state).unwrap()
        };

        let hb = |_vars: &[usize], bond: usize, input_state: &[bool], output_state: &[bool]| {
            self.get_bonds()[bond]
                .at(input_state, output_state)
                .unwrap()
        };

        match self.get_manager_ref() {
            ManagerRef::DIAGONAL(m) => m.relative_weight_for_hamiltonians(ha, hb),
            ManagerRef::LOOPER(l) => l.relative_weight_for_hamiltonians(ha, hb),
        }
    }
}

impl<R, M, L> SwapManagers for QMC<R, M, L>
where
    R: Rng,
    M: OpContainerConstructor + DiagonalUpdater + Into<L>,
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
    M: OpContainerConstructor + ClassicalLoopUpdater + RVBUpdater + Into<L>,
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

impl<R, M, L> GraphWeights for QMCIsingGraph<R, M, L>
where
    R: Rng,
    M: OpContainerConstructor + ClassicalLoopUpdater + RVBUpdater + Into<L> + OpWeights,
    L: ClusterUpdater + Into<M>,
{
    fn ham_eq(&self, other: &Self) -> bool {
        self.make_haminfo() == other.make_haminfo()
    }

    fn relative_weight(&self, h: &Self) -> f64 {
        let bond_ratio = h
            .get_edges()
            .iter()
            .zip(self.get_edges().iter())
            .enumerate()
            .map(|(bond, ((_, ja), (_, jb)))| {
                (ja / jb).powi(self.get_manager_ref().get_count(bond) as i32)
            })
            .product::<f64>();
        let nedges = self.get_edges().len();
        let t_count = (0..self.get_nvars())
            .map(|v| self.get_manager_ref().get_count(v + nedges))
            .sum::<usize>() as i32;
        let transverse_ratio =
            (h.get_transverse_field() / self.get_transverse_field()).powi(t_count);

        bond_ratio * transverse_ratio
    }
}

impl<
        R: Rng,
        M: OpContainerConstructor + ClassicalLoopUpdater + RVBUpdater + Into<L>,
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
