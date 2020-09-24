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

#![cfg_attr(feature = "const_generics", feature(min_const_generics))]

/// Classical monte carlo.
pub mod classical;
/// Memory management.
pub mod memory;
/// QMC algorithms and traits.
pub mod sse;
