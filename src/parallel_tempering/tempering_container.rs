use crate::graph::Edge;
use crate::parallel_tempering::tempering_traits::*;
use crate::sse::fast_ops::FastOps;
use crate::sse::qmc_graph;
use crate::sse::qmc_graph::QMCGraph;
use crate::sse::qmc_traits::*;
use itertools::Itertools;
use rand::prelude::ThreadRng;
use rand::Rng;
#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use std::cmp::{max, min};

/// A tempering container using FastOps and FastOpNodes.
pub type DefaultTemperingContainer<R1, R2> = TemperingContainer<R1, R2, FastOps, FastOps>;

type GraphBeta<R, M, L> = (QMCGraph<R, M, L>, f64);

/// A container to perform parallel tempering.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct TemperingContainer<
    R1: Rng,
    R2: Rng,
    M: OpContainerConstructor + DiagonalUpdater + Into<L> + StateSetter + OpWeights,
    L: ClusterUpdater + Into<M>,
> {
    nvars: usize,
    edges: Vec<(Edge, f64)>,
    cutoff: usize,

    // Graph and beta
    graphs: Vec<GraphBeta<R2, M, L>>,
    rng: Option<R1>,

    // Sort of a debug parameter to see how well swaps are going.
    total_swaps: u64,
}

/// Make a new parallel tempering container.
pub fn new_with_rng<R2: Rng, R: Rng>(
    rng: R,
    edges: Vec<(Edge, f64)>,
    cutoff: usize,
) -> DefaultTemperingContainer<R, R2> {
    TemperingContainer::new(rng, edges, cutoff)
}

/// Make a new parallel tempering container.
pub fn new_thread_rng(
    edges: Vec<(Edge, f64)>,
    cutoff: usize,
) -> DefaultTemperingContainer<ThreadRng, ThreadRng> {
    TemperingContainer::new(rand::thread_rng(), edges, cutoff)
}

impl<
        R1: Rng,
        R2: Rng,
        M: OpContainerConstructor + DiagonalUpdater + Into<L> + StateSetter + OpWeights,
        L: ClusterUpdater + Into<M>,
    > TemperingContainer<R1, R2, M, L>
{
    /// Make a new tempering container. All graphs will share this set of edgesd
    /// and start with this cutoff.
    pub fn new(rng: R1, edges: Vec<(Edge, f64)>, cutoff: usize) -> Self {
        let nvars = edges.iter().map(|((a, b), _)| max(*a, *b)).max().unwrap() + 1;
        Self {
            nvars,
            edges,
            cutoff,
            rng: Some(rng),
            graphs: vec![],
            total_swaps: 0,
        }
    }

    /// Add a graph which uses this rng, transverse field, and beta.
    pub fn add_graph(&mut self, rng: R2, transverse: f64, beta: f64) {
        let graph = QMCGraph::<R2, M, L>::new_with_rng(
            self.edges.clone(),
            transverse,
            self.cutoff,
            rng,
            None,
        );
        self.graphs.push((graph, beta))
    }

    /// Add a graph which uses this rng, transverse field, and beta. Starts with an initial state.
    pub fn add_graph_with_state(&mut self, rng: R2, transverse: f64, beta: f64, state: Vec<bool>) {
        assert_eq!(state.len(), self.nvars);
        let graph = QMCGraph::<R2, M, L>::new_with_rng(
            self.edges.clone(),
            transverse,
            self.cutoff,
            rng,
            Some(state),
        );
        self.graphs.push((graph, beta))
    }

    /// Perform a series of qmc timesteps on each graph.
    pub fn timesteps(&mut self, t: usize) {
        self.graphs.iter_mut().for_each(|(g, beta)| {
            g.timesteps(t, *beta);
        })
    }

    /// Perform a tempering step.
    pub fn tempering_step(&mut self) {
        if self.graphs.len() <= 1 {
            return;
        }
        let mut rng = self.rng.take().unwrap();

        let first_subgraphs = if self.graphs.len() % 2 == 0 {
            self.graphs.as_mut_slice()
        } else {
            let n = self.graphs.len();
            &mut self.graphs[0..n - 1]
        };
        self.total_swaps += perform_swaps(&mut rng, first_subgraphs);

        let second_subgraphs = if self.graphs.len() % 2 == 1 {
            &mut self.graphs[1..]
        } else {
            let n = self.graphs.len();
            &mut self.graphs[1..n - 1]
        };
        self.total_swaps += perform_swaps(&mut rng, second_subgraphs);

        self.rng = Some(rng);
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
    pub fn graph_ref(&self) -> &[GraphBeta<R2, M, L>] {
        &self.graphs
    }
    /// Return a mutable reference to the list of graphs and their temperatures.
    pub fn graph_mut(&mut self) -> &mut [GraphBeta<R2, M, L>] {
        &mut self.graphs
    }
    /// The number of variables in the graph.
    pub fn nvars(&self) -> usize {
        self.nvars
    }
    /// Get the number of graphs in the container.
    pub fn num_graphs(&self) -> usize {
        self.graphs.len()
    }
    /// Get the total number of successful tempering swaps which have occurred.
    pub fn get_total_swaps(&self) -> u64 {
        self.total_swaps
    }

    /// Verify all the graphs' integrity.
    pub fn verify(&self) -> bool {
        self.graphs.iter().map(|(g, _)| g.verify()).all(|b| b)
    }
    /// Print each graph.
    pub fn debug_print_each(&self) {
        println!("*********");
        for (g, _) in &self.graphs {
            g.print_debug();
        }
        println!("*********");
    }
}

fn perform_swaps<
    R1: Rng,
    R2: Rng,
    M: OpContainerConstructor + DiagonalUpdater + Into<L> + StateSetter + OpWeights,
    L: ClusterUpdater + Into<M>,
>(
    mut rng: R1,
    graphs: &mut [GraphBeta<R2, M, L>],
) -> u64 {
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
            .map(|((ga, gb), p)| if swap_on_chunks(ga, gb, p) { 1 } else { 0 })
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
fn swap_on_chunks<
    'a,
    R: 'a + Rng,
    M: 'a + OpContainerConstructor + DiagonalUpdater + Into<L> + StateSetter + OpWeights,
    L: 'a + ClusterUpdater + Into<M>,
>(
    graph_beta_a: &'a mut GraphBeta<R, M, L>,
    graph_beta_b: &'a mut GraphBeta<R, M, L>,
    p: f64,
) -> bool {
    let (ga, ba) = graph_beta_a;
    let (gb, bb) = graph_beta_b;

    let ha = ga.make_haminfo();
    let hb = gb.make_haminfo();
    let rel_h_weight = if ha.equal_assuming_graph(&hb) {
        1.0
    } else {
        let ha = |vars: &[usize], bond: usize, input_state: &[bool], output_state: &[bool]| {
            let haminfo = ga.make_haminfo();
            QMCGraph::<R, M, L>::hamiltonian(&haminfo, vars, bond, input_state, output_state)
        };
        let hb = |vars: &[usize], bond: usize, input_state: &[bool], output_state: &[bool]| {
            let haminfo = gb.make_haminfo();
            QMCGraph::<R, M, L>::hamiltonian(&haminfo, vars, bond, input_state, output_state)
        };

        // QMCGraph can only ever have 2 vars since it represents a TFIM.
        let rel_bstate = ga.relative_weight_for_hamiltonians(hb, ha);
        let rel_astate = gb.relative_weight_for_hamiltonians(ha, hb);
        rel_bstate * rel_astate
    };

    let temp_swap = (*ba / *bb).powi(gb.get_n() as i32 - ga.get_n() as i32);
    let p_swap = temp_swap * rel_h_weight;
    if p_swap > p {
        std::mem::swap(ga, gb);
        true
    } else {
        false
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

    impl<
            R1: Rng,
            R2: Rng + Send + Sync,
            M: OpContainerConstructor
                + DiagonalUpdater
                + Into<L>
                + StateSetter
                + OpWeights
                + Send
                + Sync,
            L: ClusterUpdater + Into<M> + Send + Sync,
        > ParallelQMCTimeSteps for TemperingContainer<R1, R2, M, L>
    {
        fn parallel_timesteps(&mut self, t: usize) {
            self.graphs.par_iter_mut().for_each(|(g, beta)| {
                g.timesteps(t, *beta);
            });
        }

        fn parallel_tempering_step(&mut self) {
            let mut rng = self.rng.take().unwrap();

            let first_subgraphs = if self.graphs.len() % 2 == 0 {
                &mut self.graphs
            } else {
                let n = self.graphs.len();
                &mut self.graphs[0..n - 1]
            };
            self.total_swaps += parallel_perform_swaps(&mut rng, first_subgraphs);

            let second_subgraphs = if self.graphs.len() % 2 == 1 {
                &mut self.graphs[1..]
            } else {
                let n = self.graphs.len();
                &mut self.graphs[1..n - 1]
            };
            self.total_swaps += parallel_perform_swaps(&mut rng, second_subgraphs);

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

    fn parallel_perform_swaps<
        R1: Rng,
        R2: Rng + Send + Sync,
        M: OpContainerConstructor + DiagonalUpdater + Into<L> + StateSetter + OpWeights + Send + Sync,
        L: ClusterUpdater + Into<M> + Send + Sync,
    >(
        mut rng: R1,
        graphs: &mut [GraphBeta<R2, M, L>],
    ) -> u64 {
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
                .map(|((ga, gb), p)| if swap_on_chunks(ga, gb, p) { 1 } else { 0 })
                .sum()
        }
    }

    /// Autocorrelation calculations for states.
    #[cfg(feature = "autocorrelations")]
    pub mod autocorrelations {
        use super::*;

        /// A collection of functions to calculate autocorrelations.
        pub trait ParallelTemperingAutocorrelations {
            /// Calculate autocorrelations on spin variables.
            fn calculate_variable_autocorrelation(
                &mut self,
                timesteps: usize,
                replica_swap_freq: Option<usize>,
                sampling_freq: Option<usize>,
                use_fft: Option<bool>,
            ) -> Vec<Vec<f64>>;
            /// Calculate autocorrelations on bonds.
            fn calculate_bond_autocorrelation(
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
                F: Fn(Vec<bool>) -> Vec<f64> + Copy + Send + Sync;
        }

        impl<
                R1: Rng,
                R2: Rng + Send + Sync,
                M: OpContainerConstructor
                    + DiagonalUpdater
                    + Into<L>
                    + StateSetter
                    + OpWeights
                    + Send
                    + Sync,
                L: ClusterUpdater + Into<M> + Send + Sync,
            > ParallelTemperingAutocorrelations for TemperingContainer<R1, R2, M, L>
        {
            #[cfg(feature = "autocorrelations")]
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
                    |sample| {
                        sample
                            .into_iter()
                            .map(|b| if b { 1.0 } else { -1.0 })
                            .collect()
                    },
                )
            }

            fn calculate_bond_autocorrelation(
                &mut self,
                timesteps: usize,
                replica_swap_freq: Option<usize>,
                sampling_freq: Option<usize>,
                use_fft: Option<bool>,
            ) -> Vec<Vec<f64>> {
                let edges = self.edges.clone();
                self.calculate_autocorrelation(
                    timesteps,
                    replica_swap_freq,
                    sampling_freq,
                    use_fft,
                    |sample| {
                        let even = |(a, b): &(usize, usize)| -> bool {
                            matches!((sample[*a], sample[*b]), (true, true) | (false, false))
                        };
                        edges
                            .iter()
                            .map(|(edge, j)| -> f64 {
                                let b = if *j < 0.0 { even(edge) } else { !even(edge) };
                                if b {
                                    1.0
                                } else {
                                    -1.0
                                }
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
                F: Fn(Vec<bool>) -> Vec<f64> + Copy + Send + Sync,
            {
                let replica_swap_freq = replica_swap_freq.unwrap_or(1);
                let sampling_freq = sampling_freq.unwrap_or(1);
                let states_and_energies =
                    self.parallel_timesteps_sample(timesteps, replica_swap_freq, sampling_freq);

                states_and_energies
                    .into_iter()
                    .map(|(samples, _)| {
                        let samples = samples.into_iter().map(sample_mapper).collect::<Vec<_>>();

                        if use_fft.unwrap_or(true) {
                            qmc_graph::autocorrelations::fft_autocorrelation(&samples)
                        } else {
                            qmc_graph::autocorrelations::naive_autocorrelation(&samples)
                        }
                    })
                    .collect::<Vec<_>>()
            }
        }
    }

    #[cfg(test)]
    mod parallel_swap_test {
        use super::*;
        use rand::prelude::SmallRng;
        use rand::SeedableRng;

        #[test]
        fn test_basic() {
            let rng1 = SmallRng::seed_from_u64(0u64);

            let edges = vec![((0, 1), 1.0), ((1, 2), 1.0), ((2, 3), 1.0), ((3, 4), 1.0)];
            let n = 5;

            let mut temper = new_with_rng::<SmallRng, _>(rng1, edges, 2 * n);
            for _ in 0..2 {
                let rng = SmallRng::seed_from_u64(0u64);
                temper.add_graph(rng, 0.1, 10.0);
            }
            temper.timesteps(100);
            temper.debug_print_each();
            assert!(temper.verify());

            temper.parallel_tempering_step();
            temper.debug_print_each();

            assert!(temper.verify());
        }
    }
}

/// Add serialization helpers which drop rng to only store graph states.
#[cfg(feature = "serialize")]
pub mod serialization {
    use super::*;
    use crate::sse::qmc_graph::serialization::*;

    /// Default serializable tempering container.
    pub type DefaultSerializeTemperingContainer = SerializeTemperingContainer<FastOps, FastOps>;
    type SerializeGraphBeta<M, L> = (SerializeQMCGraph<M, L>, f64);

    /// A tempering container with no rng. Just for serialization.
    #[derive(Debug, Serialize, Deserialize)]
    pub struct SerializeTemperingContainer<
        M: OpContainerConstructor + DiagonalUpdater + Into<L>,
        L: ClusterUpdater + Into<M>,
    > {
        nvars: usize,
        edges: Vec<(Edge, f64)>,
        cutoff: usize,
        graphs: Vec<SerializeGraphBeta<M, L>>,
        total_swaps: u64,
    }

    impl<
            R1: Rng,
            R2: Rng,
            M: OpContainerConstructor + DiagonalUpdater + Into<L> + StateSetter + OpWeights,
            L: ClusterUpdater + Into<M>,
        > Into<SerializeTemperingContainer<M, L>> for TemperingContainer<R1, R2, M, L>
    {
        fn into(self) -> SerializeTemperingContainer<M, L> {
            SerializeTemperingContainer {
                nvars: self.nvars,
                edges: self.edges,
                cutoff: self.cutoff,
                graphs: self
                    .graphs
                    .into_iter()
                    .map(|(g, beta)| (g.into(), beta))
                    .collect(),
                total_swaps: self.total_swaps,
            }
        }
    }

    impl<
            M: OpContainerConstructor + DiagonalUpdater + Into<L> + StateSetter + OpWeights,
            L: ClusterUpdater + Into<M>,
        > SerializeTemperingContainer<M, L>
    {
        /// Convert into a tempering container using the set of rngs.
        pub fn into_tempering_container_from_vec<R1: Rng, R2: Rng>(
            self,
            container_rng: R1,
            graph_rngs: Vec<R2>,
        ) -> TemperingContainer<R1, R2, M, L> {
            assert_eq!(self.graphs.len(), graph_rngs.len());
            self.into_tempering_container(container_rng, graph_rngs.into_iter())
        }

        /// Convert into a tempering container using the iterator of rngs.
        pub fn into_tempering_container<R1: Rng, R2: Rng, It: Iterator<Item = R2>>(
            self,
            container_rng: R1,
            graph_rngs: It,
        ) -> TemperingContainer<R1, R2, M, L> {
            TemperingContainer {
                nvars: self.nvars,
                edges: self.edges,
                cutoff: self.cutoff,
                graphs: self
                    .graphs
                    .into_iter()
                    .zip(graph_rngs)
                    .map(|((g, beta), rng)| (g.into_qmc(rng), beta))
                    .collect(),
                rng: Some(container_rng),
                total_swaps: self.total_swaps,
            }
        }
    }
}

#[cfg(test)]
mod swap_test {
    use super::*;
    use rand::prelude::SmallRng;
    use rand::SeedableRng;

    #[test]
    fn test_basic() {
        let rng1 = SmallRng::seed_from_u64(0u64);

        let edges = vec![((0, 1), 1.0), ((1, 2), 1.0), ((2, 3), 1.0), ((3, 4), 1.0)];
        let n = 5;

        let mut temper = new_with_rng::<SmallRng, _>(rng1, edges, 2 * n);
        for _ in 0..2 {
            let rng = SmallRng::seed_from_u64(0u64);
            temper.add_graph(rng, 0.1, 10.0);
        }
        temper.timesteps(1);
        temper.debug_print_each();
        assert!(temper.verify());

        temper.tempering_step();
        temper.debug_print_each();

        assert!(temper.verify());
    }
}
