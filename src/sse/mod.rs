//! A module with various QMC algorithms and traits.

pub use qmc_debug::*;
pub use qmc_ising::*;
pub use qmc_runner::*;
pub use qmc_traits::rvb::*;
pub use qmc_traits::semi_classical::*;
pub use qmc_traits::*;

/// Calculate the autcorrelation calculations for bonds.
#[cfg(feature = "autocorrelations")]
pub mod autocorrelations;

/// Clever operator management
pub mod fast_ops;

/// A generic QMC framework.
pub mod qmc_runner;

/// A QMC graph for easy TFIM.
pub mod qmc_ising;

/// Traits which, when implemented, run SSE.
pub mod qmc_traits;

/// QMC utility classes.
pub mod qmc_types;

/// Utilities for parallel tempering.
#[cfg(feature = "tempering")]
pub mod parallel_tempering;

/// Debugging stuff.
pub mod qmc_debug;

#[cfg(feature = "autocorrelations")]
pub use autocorrelations::*;

#[cfg(feature = "tempering")]
pub use parallel_tempering::*;
