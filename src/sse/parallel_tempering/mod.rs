//! A module with parallel tempering containers and algorithms. Enabled via the `tempering` and
//! `parallel-tempering` features.
//! # Example
//! ```
//! use qmc::sse::*;
//! use rand::prelude::*;
//!
//! let rng1 = SmallRng::seed_from_u64(0u64);
//! let edges = vec![((0, 1), 1.0), ((1, 2), 1.0), ((2, 3), 1.0), ((3, 4), 1.0)];
//! let mut temper = new_with_rng::<SmallRng, _>(rng1);
//! for _ in 0..2 {
//!     let rng = SmallRng::seed_from_u64(0u64);
//!     let beta = 1.0;
//!     let qmc = DefaultQMCIsingGraph::<SmallRng>::new_with_rng(
//!         edges.clone(),
//!         0.1,
//!         0.,
//!         10,
//!         rng,
//!         None,
//!     );
//!     temper.add_qmc_stepper(qmc, beta).unwrap();
//! }
//! let results = temper.timesteps_sample(1000, 1, 1);
//! ```

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
