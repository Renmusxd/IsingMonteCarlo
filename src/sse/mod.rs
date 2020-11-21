//! A module with various QMC algorithms and traits.
//!
//! Simple functionality can be accessed through the `qmc_ising` and `qmc_runner` modules, the later
//! being a generalization of the former. Other modules are useful for building custom backends for
//! sse (see `fast_ops` for an example).
//!
//! # Basic Quantum Ising Example
//! ```
//! use qmc::sse::*;
//! use rand::prelude::*;
//!
//! // H = J_ij s_i s_j
//! let edges = vec![
//!   ((0, 1), -1.0), // ((i, j), J)
//!   ((1, 2), 1.0),
//!   ((2, 3), 1.0),
//!   ((3, 0), 1.0)
//! ];
//! let transverse = 1.0;
//! let beta = 1.0;
//!
//! // Make an ising model using default system prng.
//! let rng = rand::thread_rng();
//! let initial_state = Some(vec![false, false, false, false]);
//! let initial_cutoff = 4;
//! let mut g = DefaultQMCIsingGraph::<ThreadRng>::new_with_rng(edges, transverse, initial_cutoff, rng, initial_state);
//!
//! // Take timesteps
//! g.timesteps(1000, beta);
//!
//! // Take timesteps and sample states (as Vec<Vec<bool>>).
//! let (state, average_energy) = g.timesteps_sample(1000, beta, None);
//! ```
//!
//! # Arbitrary QMC
//! ```
//! use qmc::sse::*;
//! use rand::prelude::*;
//!
//! let rng = rand::thread_rng();
//! let nvars = 3;
//! let mut g = DefaultQMC::new(nvars, rng, false);
//! // sz + sx + 1
//! let mat = vec![2.0, 1.0,
//!                1.0, 0.0];
//! for i in 0 .. nvars {
//!   g.make_interaction(mat.clone(), vec![i]).unwrap()
//! }
//! // Some energy offsets for the macrostates.
//! let diag = (0 .. 2_u32.pow(nvars as u32))
//!                 .map(|i| i as f64)
//!                 .collect::<Vec<f64>>();
//! g.make_diagonal_interaction(diag, vec![0, 1, 2]).unwrap();
//! g.timesteps(1000, 1.0);
//! ```

pub use qmc_debug::*;
pub use qmc_ising::*;
pub use qmc_runner::*;
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

/// A default hamiltonian class for carrying around functions.
pub(crate) mod ham;

/// Debugging stuff.
pub mod qmc_debug;

#[cfg(feature = "autocorrelations")]
pub use autocorrelations::*;

#[cfg(feature = "tempering")]
pub use parallel_tempering::*;
