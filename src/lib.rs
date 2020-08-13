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

//! `ising_monte_carlo` is a library for simulating classical and quantum ising systems on a lattice
//! using monte carlo methods.

#![cfg_attr(feature = "const_generics", allow(incomplete_features))]
#![cfg_attr(feature = "const_generics", feature(const_generics))]

/// Classical monte carlo.
pub mod classical;
/// Utilities for parallel tempering.
#[cfg(feature = "tempering")]
pub mod parallel_tempering;
/// QMC algorithms and traits.
pub mod sse;
