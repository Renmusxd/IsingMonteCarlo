use crate::sse::fast_ops::FastOpsTemplate;
use crate::sse::qmc_ising::QMCIsingGraph;
use crate::sse::*;
use rand::Rng;

/// Enables debugging OpContainers
pub trait DebugOps: OpContainer {
    /// Count the number of diagonal and offdiagonal ops. Sum is self.get_n()
    fn count_diagonal_and_off(&self) -> (usize, usize) {
        let cutoff = self.get_cutoff();
        let mut diag = 0;
        let mut offdiag = 0;
        for p in 0..cutoff {
            let op = self.get_pth(p);
            if let Some(op) = op {
                if op.is_diagonal() {
                    diag += 1;
                } else {
                    offdiag += 1;
                }
            }
        }
        debug_assert_eq!(diag + offdiag, self.get_n());
        (diag, offdiag)
    }
}

/// Allows for debugging QMC instances given they have a debuggable OpContainer.
pub trait QMCDebug {
    /// The type of the debuggable manager.
    type M: DebugOps;
    /// The manager which can be debugged.
    fn get_debug_manager(&self) -> &Self::M;

    /// Count the number of diagonal and offdiagonal ops.
    fn count_diagonal_and_off(&self) -> (usize, usize) {
        self.get_debug_manager().count_diagonal_and_off()
    }
}

impl<O: Op> DebugOps for FastOpsTemplate<O> {}

impl<R, M, L> QMCDebug for QMCIsingGraph<R, M, L>
where
    R: Rng,
    M: OpContainerConstructor + ClassicalLoopUpdater + RVBUpdater + DebugOps + Into<L>,
    L: ClusterUpdater + Into<M>,
{
    type M = M;

    fn get_debug_manager(&self) -> &Self::M {
        self.get_manager_ref()
    }
}
