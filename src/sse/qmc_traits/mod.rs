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

pub use cluster::*;
pub use diagonal::*;
pub use directed_loop::*;
pub use op_container::*;
pub use qmc_stepper::*;
pub use rvb::*;

/// Check integrity of a struct.
pub trait Verify {
    /// Check integrity.
    fn verify(&self) -> bool;
}
