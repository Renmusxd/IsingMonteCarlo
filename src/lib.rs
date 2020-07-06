///
/// `ising_monte_carlo` is a library for simulating classical and quantum ising systems on a lattice
/// using monte carlo methods.
///
pub mod graph;
#[cfg(feature = "tempering")]
pub mod parallel_tempering;
pub mod sse;
