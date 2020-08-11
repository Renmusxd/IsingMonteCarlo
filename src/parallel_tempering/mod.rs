//! A module with parallel tempering containers and algorithms. Enabled via the `tempering` and
//! `parallel-tempering` features.

/// A container to run parallel tempering on QMC graphs.
pub mod tempering_container;
/// Traits which allow for tempering.
pub mod tempering_traits;

pub use tempering_container::*;
pub use tempering_traits::*;

#[cfg(all(feature = "autocorrelations", feature = "parallel-tempering"))]
pub use tempering_container::rayon_tempering::autocorrelations::*;
#[cfg(feature = "parallel-tempering")]
pub use tempering_container::rayon_tempering::*;
#[cfg(feature = "serialize")]
pub use tempering_container::serialization::*;
