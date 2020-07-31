#![deny(
    missing_docs,
    unreachable_pub,
    missing_debug_implementations,
    missing_copy_implementations,
    trivial_casts,
    trivial_numeric_casts,
    unsafe_code,
    unstable_features,
    unused_import_braces,
    unused_qualifications
)]

//! `ising_monte_carlo` is a library for simulating classical and quantum ising systems on a lattice
//! using monte carlo methods.

/// Classical monte carlo.
pub mod graph;
/// Utilities for parallel tempering.
#[cfg(feature = "tempering")]
pub mod parallel_tempering;
/// QMC algorithms and traits.
pub mod sse;
