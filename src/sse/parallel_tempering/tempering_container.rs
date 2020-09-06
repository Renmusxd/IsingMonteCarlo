use crate::sse::fast_ops::FastOps;
use crate::sse::parallel_tempering::tempering_traits::*;
use crate::sse::qmc_ising::QMCIsingGraph;
use crate::sse::qmc_traits::*;
use itertools::Itertools;
use rand::prelude::ThreadRng;
use rand::Rng;
#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use std::cmp::min;

/// A tempering container using FastOps and FastOpNodes.
pub type DefaultTemperingContainer<R1, R2> = TemperingContainer<R1, QMCIsingGraph<R2, FastOps>>;

/// A container to perform parallel tempering.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct TemperingContainer<R, Q>
where
    R: Rng,
    Q: QMCStepper + GraphWeights + SwapManagers,
{
    // Graph and beta
    graphs: Vec<(Q, f64)>,
    rng: Option<R>,

    // Pairwise equality comparisons
    graph_ham_eq_a: Option<Vec<bool>>,
    graph_ham_eq_b: Option<Vec<bool>>,

    // Sort of a debug parameter to see how well swaps are going.
    total_swaps: u64,
}

/// Make a new parallel tempering container.
pub fn new_with_rng<R2: Rng, R: Rng>(rng: R) -> DefaultTemperingContainer<R, R2> {
    TemperingContainer::new(rng)
}

/// Make a new parallel tempering container.
pub fn new_thread_rng() -> DefaultTemperingContainer<ThreadRng, ThreadRng> {
    new_with_rng(rand::thread_rng())
}

impl<R, Q> TemperingContainer<R, Q>
where
    R: Rng,
    Q: QMCStepper + GraphWeights + SwapManagers,
{
    /// Make a new tempering container. All graphs will share this set of edges
    /// and start with this cutoff.
    pub fn new(rng: R) -> Self {
        Self {
            rng: Some(rng),
            graph_ham_eq_a: None,
            graph_ham_eq_b: None,
            graphs: vec![],
            total_swaps: 0,
        }
    }

    /// Add a QMC instance to the tempering container. Returns an Err if the added QMCStepper
    /// cannot be swapped with the existing steppers.
    pub fn add_qmc_stepper(&mut self, q: Q, beta: f64) -> Result<(), ()> {
        if !self.graphs.is_empty() && !self.graphs[0].0.can_swap_graphs(&q) {
            Err(())
        } else {
            self.graph_ham_eq_a = None;
            self.graph_ham_eq_b = None;
            self.graphs.push((q, beta));
            Ok(())
        }
    }

    /// Perform a series of qmc timesteps on each graph.
    pub fn timesteps(&mut self, t: usize) {
        self.graphs.iter_mut().for_each(|(g, beta)| {
            g.timesteps(t, *beta);
        })
    }

    fn make_first_subgraphs(&mut self) -> &mut [(Q, f64)] {
        if self.graphs.len() % 2 == 0 {
            self.graphs.as_mut_slice()
        } else {
            let n = self.graphs.len();
            &mut self.graphs[0..n - 1]
        }
    }

    fn make_second_subgraphs(&mut self) -> &mut [(Q, f64)] {
        if self.graphs.len() % 2 == 1 {
            &mut self.graphs[1..]
        } else {
            let n = self.graphs.len();
            &mut self.graphs[1..n - 1]
        }
    }

    fn make_ham_equalities(&mut self) {
        let graphs = self.make_first_subgraphs();
        self.graph_ham_eq_a = Some(Self::make_eqs_from_graphs(graphs));

        let graphs = self.make_second_subgraphs();
        self.graph_ham_eq_b = Some(Self::make_eqs_from_graphs(graphs));
    }

    fn make_eqs_from_graphs(graphs: &[(Q, f64)]) -> Vec<bool> {
        graphs
            .iter()
            .map(|(g, _)| g)
            .chunks(2)
            .into_iter()
            .map(unwrap_chunk)
            .map(|(ga, gb)| ga.ham_eq(gb))
            .collect()
    }

    /// Perform a tempering step.
    pub fn tempering_step(&mut self) {
        if self.graphs.len() <= 1 {
            return;
        }
        if self.graph_ham_eq_a.is_none() || self.graph_ham_eq_b.is_none() {
            self.make_ham_equalities()
        }

        let max_cutoff = self
            .graphs
            .iter()
            .map(|(g, _)| g.get_op_cutoff())
            .max()
            .unwrap();
        self.graphs
            .iter_mut()
            .for_each(|(g, _)| g.set_op_cutoff(max_cutoff));
        let mut rng = self.rng.take().unwrap();

        if rng.gen_bool(0.5) {
            self.tempering_a(&mut rng);
            self.tempering_b(&mut rng);
        } else {
            self.tempering_b(&mut rng);
            self.tempering_a(&mut rng);
        }

        self.rng = Some(rng);
    }

    fn tempering_a<R1: Rng>(&mut self, rng: R1) {
        let hameqs = self.graph_ham_eq_a.take().unwrap();
        let graphs = self.make_first_subgraphs();
        self.total_swaps += perform_swaps(rng, graphs, &hameqs);
        self.graph_ham_eq_a = Some(hameqs);
    }

    fn tempering_b<R1: Rng>(&mut self, rng: R1) {
        let hameqs = self.graph_ham_eq_b.take().unwrap();
        let graphs = self.make_second_subgraphs();
        self.total_swaps += perform_swaps(rng, graphs, &hameqs);
        self.graph_ham_eq_b = Some(hameqs);
    }

    /// Perform timesteps and sample spins. Return average energy for each graph.
    pub fn timesteps_sample(
        &mut self,
        timesteps: usize,
        replica_swap_freq: usize,
        sampling_freq: usize,
    ) -> Vec<(Vec<Vec<bool>>, f64)> {
        let mut states = (0..self.num_graphs())
            .map(|_| Vec::<Vec<bool>>::with_capacity(timesteps / sampling_freq))
            .collect::<Vec<_>>();
        let mut energy_acc = vec![0.0; self.num_graphs()];

        let mut remaining_timesteps = timesteps;
        let mut time_to_swap = replica_swap_freq;
        let mut time_to_sample = sampling_freq;

        while remaining_timesteps > 0 {
            let t = min(min(time_to_sample, time_to_swap), remaining_timesteps);
            self.graphs
                .iter_mut()
                .map(|(g, beta)| g.timesteps(t, *beta))
                .zip(energy_acc.iter_mut())
                .for_each(|(te, e)| {
                    *e += te * t as f64;
                });
            time_to_sample -= t;
            time_to_swap -= t;
            remaining_timesteps -= t;

            if time_to_swap == 0 {
                self.tempering_step();
                time_to_swap = replica_swap_freq;
            }
            if time_to_sample == 0 {
                let graphs = self.graphs.iter().map(|(g, _)| g);
                states
                    .iter_mut()
                    .zip(graphs)
                    .for_each(|(s, g)| s.push(g.state_ref().to_vec()));
                time_to_sample = sampling_freq;
            }
        }
        states.into_iter().zip(energy_acc.into_iter()).collect()
    }

    /// Apply f to each graph's state.
    pub fn iter_over_states<F>(&self, f: F)
    where
        F: Fn(&[bool]),
    {
        self.graphs.iter().for_each(|(g, _)| f(g.state_ref()))
    }

    /// Return a reference to the list of graphs and their temperatures.
    pub fn graph_ref(&self) -> &[(Q, f64)] {
        &self.graphs
    }
    /// Return a mutable reference to the list of graphs and their temperatures.
    pub fn graph_mut(&mut self) -> &mut [(Q, f64)] {
        &mut self.graphs
    }
    /// Get the number of graphs in the container.
    pub fn num_graphs(&self) -> usize {
        self.graphs.len()
    }
    /// Get the total number of successful tempering swaps which have occurred.
    pub fn get_total_swaps(&self) -> u64 {
        self.total_swaps
    }
}

fn perform_swaps<R, Q>(mut rng: R, graphs: &mut [(Q, f64)], hameqs: &[bool]) -> u64
where
    R: Rng,
    Q: QMCStepper + GraphWeights + SwapManagers,
{
    assert_eq!(graphs.len() % 2, 0);
    if graphs.is_empty() {
        0
    } else {
        graphs
            .iter_mut()
            .chunks(2)
            .into_iter()
            .map(unwrap_chunk)
            .map(|x| (x, rng.gen_range(0.0, 1.0)))
            .zip(hameqs.iter())
            .map(|(((ga, gb), p), eq)| if swap_on_chunks(ga, gb, p, !eq) { 1 } else { 0 })
            .sum()
    }
}

fn unwrap_chunk<T, It: Iterator<Item = T>>(it: It) -> (T, T) {
    let mut graphs: SmallVec<[T; 2]> = it.collect();
    assert_eq!(graphs.len(), 2);
    let gb: T = graphs.pop().unwrap();
    let ga: T = graphs.pop().unwrap();
    (ga, gb)
}

/// Returns true if a swap occurs.
fn swap_on_chunks<'a, Q>(
    graph_beta_a: &'a mut (Q, f64),
    graph_beta_b: &'a mut (Q, f64),
    p: f64,
    evaluate_hamiltonians: bool,
) -> bool
where
    Q: QMCStepper + GraphWeights + SwapManagers,
{
    let (ga, ba) = graph_beta_a;
    let (gb, bb) = graph_beta_b;

    let rel_h_weight = if !evaluate_hamiltonians {
        1.0
    } else {
        let rel_bstate = ga.relative_weight(gb);
        let rel_astate = gb.relative_weight(ga);
        rel_bstate * rel_astate
    };

    let temp_swap = (*ba / *bb).powi(gb.get_n() as i32 - ga.get_n() as i32);
    let p_swap = temp_swap * rel_h_weight;
    if p_swap > p {
        ga.swap_graphs(gb);
        true
    } else {
        false
    }
}

impl<R, Q> Verify for TemperingContainer<R, Q>
where
    R: Rng,
    Q: QMCStepper + GraphWeights + SwapManagers + Verify,
{
    fn verify(&self) -> bool {
        self.graphs.iter().all(|(q, _)| q.verify())
    }
}

/// Tempering using parallelization and threads.
#[cfg(feature = "parallel-tempering")]
pub mod rayon_tempering {
    use super::*;
    use rayon::prelude::*;

    /// Parallel tempering steps.
    pub trait ParallelQMCTimeSteps {
        /// Perform qmc steps.
        fn parallel_timesteps(&mut self, t: usize);

        /// Perform a tempering step in parallel.
        fn parallel_tempering_step(&mut self);

        /// Perform qmc steps and apply f to the states.
        fn parallel_iter_over_states<F>(&self, f: F)
        where
            F: Fn(&[bool]) + Sync;

        /// Perform qmc steps and return states and energies.
        fn parallel_timesteps_sample(
            &mut self,
            timesteps: usize,
            replica_swap_freq: usize,
            sampling_freq: usize,
        ) -> Vec<(Vec<Vec<bool>>, f64)>;
    }

    fn parallel_tempering_a<R: Rng, Q, R1: Rng>(tc: &mut TemperingContainer<R, Q>, rng: R1)
    where
        Q: QMCStepper + GraphWeights + SwapManagers + Send + Sync,
    {
        let hameqs = tc.graph_ham_eq_a.take().unwrap();
        let graphs = tc.make_first_subgraphs();
        tc.total_swaps += parallel_perform_swaps(rng, graphs, &hameqs);
        tc.graph_ham_eq_a = Some(hameqs);
    }

    fn parallel_tempering_b<R: Rng, Q, R1: Rng>(tc: &mut TemperingContainer<R, Q>, rng: R1)
    where
        Q: QMCStepper + GraphWeights + SwapManagers + Send + Sync,
    {
        let hameqs = tc.graph_ham_eq_b.take().unwrap();
        let graphs = tc.make_second_subgraphs();
        tc.total_swaps += perform_swaps(rng, graphs, &hameqs);
        tc.graph_ham_eq_b = Some(hameqs);
    }

    impl<R, Q> ParallelQMCTimeSteps for TemperingContainer<R, Q>
    where
        R: Rng,
        Q: QMCStepper + GraphWeights + SwapManagers + Send + Sync,
    {
        fn parallel_timesteps(&mut self, t: usize) {
            self.graphs.par_iter_mut().for_each(|(g, beta)| {
                g.timesteps(t, *beta);
            });
        }

        fn parallel_tempering_step(&mut self) {
            if self.graphs.is_empty() {
                return;
            }
            if self.graph_ham_eq_a.is_none() || self.graph_ham_eq_b.is_none() {
                self.make_ham_equalities()
            }

            let max_cutoff = self
                .graphs
                .par_iter()
                .map(|(g, _)| g.get_op_cutoff())
                .max()
                .unwrap();
            self.graphs
                .par_iter_mut()
                .for_each(|(g, _)| g.set_op_cutoff(max_cutoff));

            let mut rng = self.rng.take().unwrap();

            if rng.gen_bool(0.5) {
                parallel_tempering_a(self, &mut rng);
                parallel_tempering_b(self, &mut rng);
            } else {
                parallel_tempering_b(self, &mut rng);
                parallel_tempering_a(self, &mut rng);
            }

            self.rng = Some(rng);
        }

        fn parallel_iter_over_states<F>(&self, f: F)
        where
            F: Fn(&[bool]) + Sync,
        {
            self.graphs.par_iter().for_each(|(g, _)| f(g.state_ref()))
        }

        fn parallel_timesteps_sample(
            &mut self,
            timesteps: usize,
            replica_swap_freq: usize,
            sampling_freq: usize,
        ) -> Vec<(Vec<Vec<bool>>, f64)> {
            let mut states = (0..self.num_graphs())
                .map(|_| Vec::<Vec<bool>>::with_capacity(timesteps / sampling_freq))
                .collect::<Vec<_>>();
            let mut energy_acc = vec![0.0; self.num_graphs()];

            let mut remaining_timesteps = timesteps;
            let mut time_to_swap = replica_swap_freq;
            let mut time_to_sample = sampling_freq;

            while remaining_timesteps > 0 {
                let t = min(min(time_to_sample, time_to_swap), remaining_timesteps);
                self.graphs
                    .par_iter_mut()
                    .map(|(g, beta)| g.timesteps(t, *beta))
                    .zip(energy_acc.par_iter_mut())
                    .for_each(|(te, e)| {
                        *e += te * t as f64;
                    });
                time_to_sample -= t;
                time_to_swap -= t;
                remaining_timesteps -= t;

                if time_to_swap == 0 {
                    self.parallel_tempering_step();
                    time_to_swap = replica_swap_freq;
                }
                if time_to_sample == 0 {
                    let graphs = self.graphs.par_iter().map(|(g, _)| g);
                    states
                        .par_iter_mut()
                        .zip(graphs)
                        .for_each(|(s, g)| s.push(g.state_ref().to_vec()));
                    time_to_sample = sampling_freq;
                }
            }
            states.into_iter().zip(energy_acc.into_iter()).collect()
        }
    }

    fn parallel_perform_swaps<R, Q>(mut rng: R, graphs: &mut [(Q, f64)], hameqs: &[bool]) -> u64
    where
        R: Rng,
        Q: QMCStepper + GraphWeights + SwapManagers + Send + Sync,
    {
        assert_eq!(graphs.len() % 2, 0);
        if graphs.is_empty() {
            0
        } else {
            // Generate probs for bools ahead of time, this way we can parallelize.
            let probs = (0..graphs.len() / 2)
                .map(|_| rng.gen_range(0.0, 1.0))
                .collect::<Vec<_>>();
            graphs
                .par_iter_mut()
                .chunks(2)
                .map(|g| unwrap_chunk(g.into_iter()))
                .zip(probs.into_par_iter())
                .zip(hameqs.into_par_iter())
                .map(|(((ga, gb), p), eq)| if swap_on_chunks(ga, gb, p, !eq) { 1 } else { 0 })
                .sum()
        }
    }

    /// Autocorrelation calculations for states.
    #[cfg(feature = "autocorrelations")]
    pub mod autocorrelations {
        use super::*;
        use crate::sse::autocorrelations::{fft_autocorrelation, naive_autocorrelation};
        use crate::sse::QMCBondAutoCorrelations;

        /// A collection of functions to calculate autocorrelations.
        pub trait ParallelTemperingAutocorrelations<Q> {
            /// Calculate autocorrelations on spin variables.
            fn calculate_variable_autocorrelation(
                &mut self,
                timesteps: usize,
                replica_swap_freq: Option<usize>,
                sampling_freq: Option<usize>,
                use_fft: Option<bool>,
            ) -> Vec<Vec<f64>>;

            /// Calculate autocorrelations on the output of f applied to states.
            fn calculate_autocorrelation<F>(
                &mut self,
                timesteps: usize,
                replica_swap_freq: Option<usize>,
                sampling_freq: Option<usize>,
                use_fft: Option<bool>,
                sample_mapper: F,
            ) -> Vec<Vec<f64>>
            where
                F: Fn(&[bool], &Q) -> Vec<f64> + Copy + Send + Sync;

            /// Take autocorrelations of products of spins.
            fn calculate_spin_product_autocorrelation(
                &mut self,
                timesteps: usize,
                replica_swap_freq: Option<usize>,
                var_products: &[&[usize]],
                sampling_freq: Option<usize>,
                use_fft: Option<bool>,
            ) -> Vec<Vec<f64>>;
        }

        /// Allows parallel computation of bond variables.
        pub trait ParallelTemperingBondAutoCorrelations<Q>:
            ParallelTemperingAutocorrelations<Q>
        {
            /// Calculate autocorrelations on bonds.
            fn calculate_bond_autocorrelation(
                &mut self,
                timesteps: usize,
                replica_swap_freq: Option<usize>,
                sampling_freq: Option<usize>,
                use_fft: Option<bool>,
            ) -> Vec<Vec<f64>>;
        }

        impl<R, Q> ParallelTemperingAutocorrelations<Q> for TemperingContainer<R, Q>
        where
            R: Rng,
            Q: QMCStepper + GraphWeights + SwapManagers + Send + Sync,
        {
            fn calculate_variable_autocorrelation(
                &mut self,
                timesteps: usize,
                replica_swap_freq: Option<usize>,
                sampling_freq: Option<usize>,
                use_fft: Option<bool>,
            ) -> Vec<Vec<f64>> {
                self.calculate_autocorrelation(
                    timesteps,
                    replica_swap_freq,
                    sampling_freq,
                    use_fft,
                    |sample, _| {
                        sample
                            .iter()
                            .cloned()
                            .map(|b| if b { 1.0 } else { -1.0 })
                            .collect()
                    },
                )
            }

            fn calculate_spin_product_autocorrelation(
                &mut self,
                timesteps: usize,
                replica_swap_freq: Option<usize>,
                var_products: &[&[usize]],
                sampling_freq: Option<usize>,
                use_fft: Option<bool>,
            ) -> Vec<Vec<f64>> {
                self.calculate_autocorrelation(
                    timesteps,
                    replica_swap_freq,
                    sampling_freq,
                    use_fft,
                    |sample, _| {
                        var_products
                            .iter()
                            .map(|vs| {
                                vs.iter()
                                    .map(|v| if sample[*v] { 1.0 } else { -1.0 })
                                    .product()
                            })
                            .collect()
                    },
                )
            }

            fn calculate_autocorrelation<F>(
                &mut self,
                timesteps: usize,
                replica_swap_freq: Option<usize>,
                sampling_freq: Option<usize>,
                use_fft: Option<bool>,
                sample_mapper: F,
            ) -> Vec<Vec<f64>>
            where
                F: Fn(&[bool], &Q) -> Vec<f64> + Copy + Send + Sync,
            {
                let replica_swap_freq = replica_swap_freq.unwrap_or(1);
                let sampling_freq = sampling_freq.unwrap_or(1);
                let states_and_energies =
                    self.parallel_timesteps_sample(timesteps, replica_swap_freq, sampling_freq);

                states_and_energies
                    .iter()
                    .zip(self.graphs.iter())
                    .map(|((samples, _), (q, _))| {
                        let samples = samples
                            .iter()
                            .map(|s| sample_mapper(s, q))
                            .collect::<Vec<_>>();

                        if use_fft.unwrap_or(true) {
                            fft_autocorrelation(&samples)
                        } else {
                            naive_autocorrelation(&samples)
                        }
                    })
                    .collect::<Vec<_>>()
            }
        }

        impl<R, Q> ParallelTemperingBondAutoCorrelations<Q> for TemperingContainer<R, Q>
        where
            R: Rng,
            Q: QMCStepper + GraphWeights + QMCBondAutoCorrelations + SwapManagers + Send + Sync,
        {
            fn calculate_bond_autocorrelation(
                &mut self,
                timesteps: usize,
                replica_swap_freq: Option<usize>,
                sampling_freq: Option<usize>,
                use_fft: Option<bool>,
            ) -> Vec<Vec<f64>> {
                self.calculate_autocorrelation(
                    timesteps,
                    replica_swap_freq,
                    sampling_freq,
                    use_fft,
                    |sample, q| {
                        let nbonds = q.n_bonds();
                        (0..nbonds)
                            .map(|bond| q.value_for_bond(bond, &sample))
                            .collect()
                    },
                )
            }
        }
    }

    #[cfg(test)]
    mod parallel_swap_test {
        use super::*;
        use crate::sse::*;
        use rand::prelude::SmallRng;
        use rand::SeedableRng;

        #[test]
        fn test_basic() -> Result<(), ()> {
            let rng1 = SmallRng::seed_from_u64(0u64);

            let edges = vec![((0, 1), 1.0), ((1, 2), 1.0), ((2, 3), 1.0), ((3, 4), 1.0)];
            let mut temper = new_with_rng::<SmallRng, _>(rng1);
            for _ in 0..2 {
                let rng = SmallRng::seed_from_u64(0u64);
                let qmc = DefaultQMCIsingGraph::<SmallRng>::new_with_rng(
                    edges.clone(),
                    0.1,
                    10,
                    rng,
                    None,
                );
                temper.add_qmc_stepper(qmc, 10.0)?;
            }
            temper.timesteps(100);
            assert!(temper.verify());

            temper.parallel_tempering_step();
            assert!(temper.verify());
            Ok(())
        }
    }
}

/// Add serialization helpers which drop rng to only store graph states.
#[cfg(feature = "serialize")]
pub mod serialization {
    use super::*;
    use crate::sse::qmc_ising::serialization::*;
    use crate::sse::qmc_ising::*;

    /// Default serializable tempering container.
    pub type DefaultSerializeTemperingContainer = SerializeTemperingContainer<FastOps>;
    type SerializeGraphBeta<M> = (SerializeQMCGraph<M>, f64);

    /// A tempering container with no rng. Just for serialization.
    #[derive(Debug, Serialize, Deserialize)]
    pub struct SerializeTemperingContainer<M: IsingManager> {
        graphs: Vec<SerializeGraphBeta<M>>,
        total_swaps: u64,
    }

    impl<R1, R2, M> Into<SerializeTemperingContainer<M>>
        for TemperingContainer<R1, QMCIsingGraph<R2, M>>
    where
        R1: Rng,
        R2: Rng,
        M: IsingManager + GraphWeights,
    {
        fn into(self) -> SerializeTemperingContainer<M> {
            SerializeTemperingContainer {
                graphs: self
                    .graphs
                    .into_iter()
                    .map(|(g, beta)| (g.into(), beta))
                    .collect(),
                total_swaps: self.total_swaps,
            }
        }
    }

    impl<M> SerializeTemperingContainer<M>
    where
        M: IsingManager,
    {
        /// Convert into a tempering container using the set of rngs.
        pub fn into_tempering_container_from_vec<R1: Rng, R2: Rng>(
            self,
            container_rng: R1,
            graph_rngs: Vec<R2>,
        ) -> TemperingContainer<R1, QMCIsingGraph<R2, M>> {
            assert_eq!(self.graphs.len(), graph_rngs.len());
            self.into_tempering_container(container_rng, graph_rngs.into_iter())
        }

        /// Convert into a tempering container using the iterator of rngs.
        pub fn into_tempering_container<R1: Rng, R2: Rng, It: Iterator<Item = R2>>(
            self,
            container_rng: R1,
            graph_rngs: It,
        ) -> TemperingContainer<R1, QMCIsingGraph<R2, M>> {
            TemperingContainer {
                graphs: self
                    .graphs
                    .into_iter()
                    .zip(graph_rngs)
                    .map(|((g, beta), rng)| (g.into_qmc(rng), beta))
                    .collect(),
                rng: Some(container_rng),
                graph_ham_eq_a: None,
                graph_ham_eq_b: None,
                total_swaps: self.total_swaps,
            }
        }
    }
}

#[cfg(test)]
mod swap_test {
    use super::*;
    use crate::sse::*;
    use rand::prelude::SmallRng;
    use rand::SeedableRng;

    #[test]
    fn test_basic() -> Result<(), ()> {
        let rng1 = SmallRng::seed_from_u64(0u64);

        let edges = vec![((0, 1), 1.0), ((1, 2), 1.0), ((2, 3), 1.0), ((3, 4), 1.0)];

        let mut temper = new_with_rng::<SmallRng, SmallRng>(rng1);
        for _ in 0..2 {
            let rng = SmallRng::seed_from_u64(0u64);
            let qmc =
                DefaultQMCIsingGraph::<SmallRng>::new_with_rng(edges.clone(), 0.1, 10, rng, None);
            temper.add_qmc_stepper(qmc, 10.0)?;
        }
        temper.timesteps(1);
        assert!(temper.verify());

        temper.tempering_step();
        assert!(temper.verify());
        Ok(())
    }
}
