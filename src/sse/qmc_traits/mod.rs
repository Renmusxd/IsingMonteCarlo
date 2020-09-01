/// Cluster updates in imaginary time.
pub mod cluster;
/// Diagonal update moves.
pub mod diagonal;
/// SSE directed loop updates.
pub mod directed_loop;
/// Holding and manipulating ops.
pub mod op_container;
/// QMC timesteps and analysis.
pub mod qmc_stepper;
/// Resonating bonds
pub mod rvb;
/// Semi-classical SSE algorithms
pub mod semi_classical;

pub use cluster::*;
pub use diagonal::*;
pub use directed_loop::*;
pub use op_container::*;
pub use qmc_stepper::*;
pub use rvb::*;
pub use semi_classical::*;

/// Check integrity of a struct.
pub trait Verify {
    /// Check integrity.
    fn verify(&self) -> bool;
}
