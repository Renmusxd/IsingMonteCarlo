//! A module with various QMC algorithms and traits.

pub use qmc_ising::DefaultQMCIsingGraph;
pub use qmc_traits::*;

/// An arena for managing memory efficiently
pub(crate) mod arena;

/// Clever operator management
pub mod fast_ops;

/// A wrapper for QMC data.
pub mod qmc_ising;

/// Traits which, when implemented, run SSE.
pub mod qmc_traits;

/// QMC utility classes.
pub mod qmc_types;

/// A simpler operator management.
pub mod simple_ops;
