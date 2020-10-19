#![deny(
    missing_docs,
    unreachable_pub,
    missing_debug_implementations,
    missing_copy_implementations,
    trivial_casts,
    trivial_numeric_casts,
    unsafe_code,
    unused_import_braces,
    unused_qualifications
)]

//! `qmc` is a library for simulating classical and quantum ising systems on a lattice
//! using monte carlo methods.
//!
//! The sse library contains built-in classes to handle ising models, as well as classes which handle
//! arbitrary interactions.
//!
//! It also offers a few feature gated modules:
//! - parallel tempering system using the `tempering` or `parallel-tempering` feature gates.
//! - autocorrelation calculations on variables, bonds, or arbitrary values: use `autocorrelations`
//! - graph serialization using serde with the `serialize` feature.
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
//! let mut g = DefaultQMCIsingGraph::<ThreadRng>::new_with_rng(edges, transverse, 3, rng, None);
//!
//! // Take timesteps
//! g.timesteps(1000, beta);
//!
//! // Take timesteps and sample states (as Vec<Vec<bool>>).
//! let (state, average_energy) = g.timesteps_sample(1000, beta, None);
//! ```

#![cfg_attr(feature = "const_generics", feature(min_const_generics))]

/// A limited classical monte carlo library for ising models.
pub mod classical;
/// Memory management, useful for building custom QMC backends.
pub mod memory;
/// QMC algorithms and traits.
pub mod sse;
