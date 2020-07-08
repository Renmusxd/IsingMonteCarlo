use crate::graph::Edge;
use crate::parallel_tempering::tempering_traits::*;
use crate::sse::fast_ops::{FastOpNode, FastOps};
use crate::sse::qmc_graph;
use crate::sse::qmc_graph::QMCGraph;
use crate::sse::qmc_traits::*;
use itertools::Itertools;
use rand::prelude::ThreadRng;
use rand::Rng;
use smallvec::SmallVec;
use std::cmp::{max, min};

pub type DefaultTemperingContainer<R1, R2> =
    TemperingContainer<R1, R2, FastOpNode, FastOps, FastOps>;

pub struct TemperingContainer<
    R1: Rng,
    R2: Rng,
    N: OpNode,
    M: OpContainerConstructor + DiagonalUpdater + ConvertsToLooper<N, L> + StateSetter + OpWeights,
    L: LoopUpdater<N> + ClusterUpdater<N> + ConvertsToDiagonal<M>,
> {
    nvars: usize,
    edges: Vec<(Edge, f64)>,
    cutoff: usize,
    use_loop_update: bool,
    use_heatbath_diagonal_update: bool,

    // Graph and beta
    graphs: Vec<(QMCGraph<R2, N, M, L>, f64)>,
    rng: Option<R1>,
}

pub fn new_with_rng<R2: Rng, R: Rng>(
    rng: R,
    edges: Vec<(Edge, f64)>,
    cutoff: usize,
    use_loop_update: bool,
    use_heatbath_diagonal_update: bool,
) -> DefaultTemperingContainer<R, R2> {
    TemperingContainer::new(
        rng,
        edges,
        cutoff,
        use_loop_update,
        use_heatbath_diagonal_update,
    )
}

pub fn new_thread_rng(
    edges: Vec<(Edge, f64)>,
    cutoff: usize,
    use_loop_update: bool,
    use_heatbath_diagonal_update: bool,
) -> DefaultTemperingContainer<ThreadRng, ThreadRng> {
    TemperingContainer::new(
        rand::thread_rng(),
        edges,
        cutoff,
        use_loop_update,
        use_heatbath_diagonal_update,
    )
}

impl<
        R1: Rng,
        R2: Rng,
        N: OpNode,
        M: OpContainerConstructor + DiagonalUpdater + ConvertsToLooper<N, L> + StateSetter + OpWeights,
        L: LoopUpdater<N> + ClusterUpdater<N> + ConvertsToDiagonal<M>,
    > TemperingContainer<R1, R2, N, M, L>
{
    pub fn new(
        rng: R1,
        edges: Vec<(Edge, f64)>,
        cutoff: usize,
        use_loop_update: bool,
        use_heatbath_diagonal_update: bool,
    ) -> Self {
        let nvars = edges.iter().map(|((a, b), _)| max(*a, *b)).max().unwrap() + 1;
        Self {
            nvars,
            edges,
            cutoff,
            use_loop_update,
            use_heatbath_diagonal_update,
            rng: Some(rng),
            graphs: vec![],
        }
    }

    pub fn add_graph(&mut self, rng: R2, transverse: f64, beta: f64) {
        let graph = QMCGraph::<R2, N, M, L>::new_with_rng(
            self.edges.clone(),
            transverse,
            self.cutoff,
            self.use_loop_update,
            self.use_heatbath_diagonal_update,
            rng,
            None,
        );
        self.graphs.push((graph, beta))
    }

    pub fn add_graph_with_state(&mut self, rng: R2, transverse: f64, beta: f64, state: Vec<bool>) {
        assert_eq!(state.len(), self.nvars);
        let graph = QMCGraph::<R2, N, M, L>::new_with_rng(
            self.edges.clone(),
            transverse,
            self.cutoff,
            self.use_loop_update,
            self.use_heatbath_diagonal_update,
            rng,
            Some(state),
        );
        self.graphs.push((graph, beta))
    }

    pub fn timesteps(&mut self, t: usize) {
        self.graphs.iter_mut().for_each(|(g, beta)| {
            g.timesteps(t, *beta);
        })
    }

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
        perform_swaps(&mut rng, first_subgraphs);

        let second_subgraphs = if self.graphs.len() % 2 == 1 {
            &mut self.graphs[1..]
        } else {
            let n = self.graphs.len();
            &mut self.graphs[1..n - 1]
        };
        perform_swaps(&mut rng, second_subgraphs);

        self.rng = Some(rng);
    }

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

    pub fn iter_over_states<F>(&self, f: F)
    where
        F: Fn(&[bool]),
    {
        self.graphs.iter().for_each(|(g, _)| f(g.state_ref()))
    }

    pub fn graph_ref(&self) -> &[(QMCGraph<R2, N, M, L>, f64)] {
        &self.graphs
    }
    pub fn graph_mut(&mut self) -> &mut [(QMCGraph<R2, N, M, L>, f64)] {
        &mut self.graphs
    }

    pub fn nvars(&self) -> usize {
        self.nvars
    }

    pub fn num_graphs(&self) -> usize {
        self.graphs.len()
    }

    pub fn verify(&self) -> bool {
        self.graphs.iter().map(|(g, _)| g.verify()).all(|b| b)
    }

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
    N: OpNode,
    M: OpContainerConstructor + DiagonalUpdater + ConvertsToLooper<N, L> + StateSetter + OpWeights,
    L: LoopUpdater<N> + ClusterUpdater<N> + ConvertsToDiagonal<M>,
>(
    mut rng: R1,
    graphs: &mut [(QMCGraph<R2, N, M, L>, f64)],
) {
    assert!(graphs.len() % 2 == 0);
    if graphs.is_empty() {
        return;
    }
    graphs
        .iter_mut()
        .map(|(g, _)| g)
        .chunks(2)
        .into_iter()
        .map(unwrap_chunk)
        .map(|x| (x, rng.gen_range(0.0, 1.0)))
        .for_each(|((ga, gb), p)| swap_on_chunks(ga, gb, p));
}

fn unwrap_chunk<T, It: Iterator<Item = T>>(it: It) -> (T, T) {
    let mut graphs: SmallVec<[T; 2]> = it.collect();
    assert_eq!(graphs.len(), 2);
    let gb: T = graphs.pop().unwrap();
    let ga: T = graphs.pop().unwrap();
    (ga, gb)
}

fn swap_on_chunks<
    'a,
    R: 'a + Rng,
    N: 'a + OpNode,
    M: 'a
        + OpContainerConstructor
        + DiagonalUpdater
        + ConvertsToLooper<N, L>
        + StateSetter
        + OpWeights,
    L: 'a + LoopUpdater<N> + ClusterUpdater<N> + ConvertsToDiagonal<M>,
>(
    ga: &'a mut QMCGraph<R, N, M, L>,
    gb: &'a mut QMCGraph<R, N, M, L>,
    p: f64,
) {
    let mut a_state = ga.get_state_ref().to_vec();
    let mut b_state = gb.get_state_ref().to_vec();

    let ha = |vars: &[usize], bond: usize, input_state: &[bool], output_state: &[bool]| {
        let haminfo = ga.make_haminfo();
        QMCGraph::<R, N, M, L>::hamiltonian(&haminfo, vars, bond, input_state, output_state)
    };
    let hb = |vars: &[usize], bond: usize, input_state: &[bool], output_state: &[bool]| {
        let haminfo = gb.make_haminfo();
        QMCGraph::<R, N, M, L>::hamiltonian(&haminfo, vars, bond, input_state, output_state)
    };

    let rel_bstate = ga.relative_weight_for_state(ha, &mut b_state);
    let rel_astate = gb.relative_weight_for_state(hb, &mut a_state);
    let p_swap = rel_bstate * rel_astate;
    if p_swap > p {
        ga.set_state(b_state);
        gb.set_state(a_state);
    }
}

#[cfg(feature = "parallel-tempering")]
pub mod rayon_tempering {
    use super::*;
    use rayon::prelude::*;

    pub trait ParallelQMCTimeSteps {
        fn parallel_timesteps(&mut self, t: usize);
        fn parallel_tempering_step(&mut self);
        fn parallel_iter_over_states<F>(&self, f: F)
        where
            F: Fn(&[bool]) + Sync;
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
            N: OpNode + Send + Sync,
            M: OpContainerConstructor
                + DiagonalUpdater
                + ConvertsToLooper<N, L>
                + StateSetter
                + OpWeights
                + Send
                + Sync,
            L: LoopUpdater<N> + ClusterUpdater<N> + ConvertsToDiagonal<M> + Send + Sync,
        > ParallelQMCTimeSteps for TemperingContainer<R1, R2, N, M, L>
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
            parallel_perform_swaps(&mut rng, first_subgraphs);

            let second_subgraphs = if self.graphs.len() % 2 == 1 {
                &mut self.graphs[1..]
            } else {
                let n = self.graphs.len();
                &mut self.graphs[1..n - 1]
            };
            parallel_perform_swaps(&mut rng, second_subgraphs);

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
        N: OpNode + Send + Sync,
        M: OpContainerConstructor
            + DiagonalUpdater
            + ConvertsToLooper<N, L>
            + StateSetter
            + OpWeights
            + Send
            + Sync,
        L: LoopUpdater<N> + ClusterUpdater<N> + ConvertsToDiagonal<M> + Send + Sync,
    >(
        mut rng: R1,
        graphs: &mut [(QMCGraph<R2, N, M, L>, f64)],
    ) {
        assert_eq!(graphs.len() % 2, 0);
        if graphs.is_empty() {
            return;
        }
        // Generate probs for bools ahead of time, this way we can parallelize.
        let probs = (0..graphs.len() / 2)
            .map(|_| rng.gen_range(0.0, 1.0))
            .collect::<Vec<_>>();
        graphs
            .par_iter_mut()
            .map(|(g, _)| g)
            .chunks(2)
            .map(|g| unwrap_chunk(g.into_iter()))
            .zip(probs.into_par_iter())
            .for_each(|((ga, gb), p)| swap_on_chunks(ga, gb, p));
    }

    #[cfg(feature = "autocorrelations")]
    pub mod autocorrelations {
        use super::*;

        pub trait ParallelTemperingAutocorrelations {
            fn calculate_variable_autocorrelation(
                &mut self,
                timesteps: usize,
                replica_swap_freq: Option<usize>,
                sampling_freq: Option<usize>,
                use_fft: Option<bool>,
            ) -> Vec<Vec<f64>>;
            fn calculate_bond_autocorrelation(
                &mut self,
                timesteps: usize,
                replica_swap_freq: Option<usize>,
                sampling_freq: Option<usize>,
                use_fft: Option<bool>,
            ) -> Vec<Vec<f64>>;
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
                N: OpNode + Send + Sync,
                M: OpContainerConstructor
                    + DiagonalUpdater
                    + ConvertsToLooper<N, L>
                    + StateSetter
                    + OpWeights
                    + Send
                    + Sync,
                L: LoopUpdater<N> + ClusterUpdater<N> + ConvertsToDiagonal<M> + Send + Sync,
            > ParallelTemperingAutocorrelations for TemperingContainer<R1, R2, N, M, L>
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
                })
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
                            match (sample[*a], sample[*b]) {
                                (true, true) | (false, false) => true,
                                _ => false,
                            }
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

            let mut temper = new_with_rng::<SmallRng, _>(rng1, edges, 2 * n, false, false);
            for i in 0..2 {
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

        let mut temper = new_with_rng::<SmallRng, _>(rng1, edges, 2 * n, false, false);
        for i in 0..2 {
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
