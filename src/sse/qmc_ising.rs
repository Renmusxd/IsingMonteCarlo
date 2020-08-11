use crate::graph::{Edge, GraphState};
use crate::sse::fast_ops::FastOps;
use crate::sse::qmc_traits::*;
use rand::rngs::ThreadRng;
use rand::Rng;
#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};
use std::cmp::max;
use std::marker::PhantomData;

/// Default QMC graph implementation.
pub type DefaultQMCIsingGraph<R> = QMCIsingGraph<R, FastOps, FastOps>;

type VecEdge = Vec<usize>;

/// A container to run QMC simulations.
#[derive(Debug)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct QMCIsingGraph<
    R: Rng,
    M: OpContainerConstructor + DiagonalUpdater + Into<L>,
    L: ClusterUpdater + Into<M>,
> {
    edges: Vec<(VecEdge, f64)>,
    transverse: f64,
    state: Option<Vec<bool>>,
    cutoff: usize,
    op_manager: Option<M>,
    twosite_energy_offset: f64,
    singlesite_energy_offset: f64,
    rng: Option<R>,
    phantom: PhantomData<L>,
    // This is just an array of the variables 0..nvars
    vars: Vec<usize>,
    // An alloc to reuse in cluster updates
    state_updates: Vec<(usize, bool)>,
}

/// Build a new qmc graph with thread rng.
pub fn new_qmc(
    edges: Vec<(Edge, f64)>,
    transverse: f64,
    cutoff: usize,
    state: Option<Vec<bool>>,
) -> DefaultQMCIsingGraph<ThreadRng> {
    let rng = rand::thread_rng();
    DefaultQMCIsingGraph::<ThreadRng>::new_with_rng(edges, transverse, cutoff, rng, state)
}

/// Build a new qmc graph with thread rng from a classical graph.
pub fn new_qmc_from_graph(
    graph: GraphState,
    transverse: f64,
    cutoff: usize,
) -> DefaultQMCIsingGraph<ThreadRng> {
    let rng = rand::thread_rng();
    DefaultQMCIsingGraph::<ThreadRng>::new_from_graph(graph, transverse, cutoff, rng)
}

impl<
        R: Rng,
        M: OpContainerConstructor + DiagonalUpdater + Into<L>,
        L: ClusterUpdater + Into<M>,
    > QMCIsingGraph<R, M, L>
{
    /// Make a new QMC graph with an rng instance.
    pub fn new_with_rng<Rg: Rng>(
        edges: Vec<(Edge, f64)>,
        transverse: f64,
        cutoff: usize,
        rng: Rg,
        state: Option<Vec<bool>>,
    ) -> QMCIsingGraph<Rg, M, L> {
        let nvars = edges.iter().map(|((a, b), _)| max(*a, *b)).max().unwrap() + 1;
        let edges = edges
            .into_iter()
            .map(|((a, b), j)| (vec![a, b], j))
            .collect::<Vec<_>>();
        let twosite_energy_offset = edges
            .iter()
            .fold(None, |acc, (_, j)| match acc {
                None => Some(*j),
                Some(acc) => Some(if *j < acc { *j } else { acc }),
            })
            .unwrap_or(0.0)
            .abs();
        let singlesite_energy_offset = transverse;
        let mut ops = M::new(nvars);
        ops.set_cutoff(cutoff);

        let state = match state {
            Some(state) => state,
            None => GraphState::make_random_spin_state(nvars),
        };
        let state = Some(state);

        QMCIsingGraph::<Rg, M, L> {
            edges,
            transverse,
            state,
            op_manager: Some(ops),
            cutoff,
            twosite_energy_offset,
            singlesite_energy_offset,
            rng: Some(rng),
            phantom: PhantomData,
            vars: (0..nvars).collect(),
            state_updates: vec![],
        }
    }

    /// Make a new QMC graph with an rng instance.
    pub fn new_from_graph<Rg: Rng>(
        graph: GraphState,
        transverse: f64,
        cutoff: usize,
        rng: Rg,
    ) -> QMCIsingGraph<Rg, M, L> {
        assert!(graph.biases.into_iter().all(|v| v == 0.0));
        Self::new_with_rng(graph.edges, transverse, cutoff, rng, graph.state)
    }

    /// Take t qmc timesteps at beta.
    pub fn timesteps(&mut self, t: usize, beta: f64) -> f64 {
        let (_, average_energy) = self.timesteps_measure(t, beta, (), |_acc, _state| (), None);
        average_energy
    }

    /// Take t qmc timesteps at beta and sample states.
    pub fn timesteps_sample(
        &mut self,
        t: usize,
        beta: f64,
        sampling_freq: Option<usize>,
    ) -> (Vec<Vec<bool>>, f64) {
        let acc = Vec::with_capacity(t / sampling_freq.unwrap_or(1) + 1);
        self.timesteps_measure(
            t,
            beta,
            acc,
            |mut acc, state| {
                acc.push(state.to_vec());
                acc
            },
            sampling_freq,
        )
    }

    /// Take t qmc timesteps at beta and sample states, apply f to each.
    pub fn timesteps_sample_iter<F>(
        &mut self,
        t: usize,
        beta: f64,
        sampling_freq: Option<usize>,
        iter_fn: F,
    ) -> f64
    where
        F: Fn(&[bool]),
    {
        let (_, e) = self.timesteps_measure(t, beta, (), |_, state| iter_fn(state), sampling_freq);
        e
    }

    /// Take t qmc timesteps at beta and sample states, apply f to each and the zipped iterator.
    pub fn timesteps_sample_iter_zip<F, I, T>(
        &mut self,
        t: usize,
        beta: f64,
        sampling_freq: Option<usize>,
        zip_with: I,
        iter_fn: F,
    ) -> f64
    where
        F: Fn(T, &[bool]),
        I: Iterator<Item = T>,
    {
        let (_, e) = self.timesteps_measure(
            t,
            beta,
            Some(zip_with),
            |zip_iter, state| {
                if let Some(mut zip_iter) = zip_iter {
                    let next = zip_iter.next();
                    if let Some(next) = next {
                        iter_fn(next, state);
                        Some(zip_iter)
                    } else {
                        None
                    }
                } else {
                    None
                }
            },
            sampling_freq,
        );
        e
    }

    /// Take t qmc timesteps at beta and sample states, fold across states and output results.
    pub fn timesteps_measure<F, T>(
        &mut self,
        timesteps: usize,
        beta: f64,
        init_t: T,
        state_fold: F,
        sampling_freq: Option<usize>,
    ) -> (T, f64)
    where
        F: Fn(T, &[bool]) -> T,
    {
        let mut acc = init_t;
        let mut steps_measured = 0;
        let mut total_n = 0;
        let sampling_freq = sampling_freq.unwrap_or(1);

        for t in 0..timesteps {
            self.timestep(beta);

            // Sample every `sampling_freq`
            // Ignore first one.
            if (t + 1) % sampling_freq == 0 {
                acc = state_fold(acc, self.state.as_ref().unwrap());
                steps_measured += 1;
                total_n += self.op_manager.as_ref().unwrap().get_n();
            }
        }
        let average_energy = -(total_n as f64 / (steps_measured as f64 * beta));
        // Get total energy offset (num_vars * singlesite + num_edges * twosite)
        let twosite_energy_offset = self.twosite_energy_offset;
        let singlesite_energy_offset = self.singlesite_energy_offset;
        let nvars = self.state.as_ref().unwrap().len();

        let offset = twosite_energy_offset * self.edges.len() as f64
            + singlesite_energy_offset * nvars as f64;
        (acc, average_energy + offset)
    }

    /// Make the hamiltonian struct.
    pub fn make_haminfo(&self) -> HamInfo {
        HamInfo {
            edges: &self.edges,
            transverse: self.transverse,
            singlesite_energy_offset: self.singlesite_energy_offset,
            twosite_energy_offset: self.twosite_energy_offset,
        }
    }

    /// Evaluate the hamiltonian using the HamInfo for the graph.
    pub fn hamiltonian(
        info: &HamInfo,
        vars: &[usize],
        bond: usize,
        input_state: &[bool],
        output_state: &[bool],
    ) -> f64 {
        match vars.len() {
            1 => single_site_hamiltonian(
                input_state[0],
                output_state[0],
                info.transverse,
                info.singlesite_energy_offset,
            ),
            2 => two_site_hamiltonian(
                (input_state[0], input_state[1]),
                (output_state[0], output_state[1]),
                info.edges[bond].1,
                info.twosite_energy_offset,
            ),
            _ => unreachable!(),
        }
    }

    /// Perform a single step of qmc.
    pub fn timestep(&mut self, beta: f64) {
        let mut state = self.state.take().unwrap();
        let mut manager = self.op_manager.take().unwrap();
        let rng = self.rng.as_mut().unwrap();

        let nvars = state.len();
        let edges = &self.edges;
        let vars = &self.vars;
        let transverse = self.transverse;
        let twosite_energy_offset = self.twosite_energy_offset;
        let singlesite_energy_offset = self.singlesite_energy_offset;
        let hinfo = HamInfo {
            edges,
            transverse,
            singlesite_energy_offset,
            twosite_energy_offset,
        };
        let h = |vars: &[usize], bond: usize, input_state: &[bool], output_state: &[bool]| {
            Self::hamiltonian(&hinfo, vars, bond, input_state, output_state)
        };

        let num_bonds = edges.len() + nvars;
        let bonds_fn = |b: usize| -> &[usize] {
            if b < edges.len() {
                &edges[b].0
            } else {
                let b = b - edges.len();
                &vars[b..b + 1]
            }
        };

        // Start by editing the ops list
        let ham = Hamiltonian::new(h, bonds_fn, num_bonds);
        manager.make_diagonal_update_with_rng_and_state_ref(
            self.cutoff,
            beta,
            &mut state,
            &ham,
            rng,
        );
        self.cutoff = max(self.cutoff, manager.get_n() + manager.get_n() / 2);

        let mut manager = manager.into();
        manager.flip_each_cluster_rng(0.5, rng, &mut state);

        state.iter_mut().enumerate().for_each(|(var, state)| {
            if !manager.does_var_have_ops(var) {
                *state = rng.gen_bool(0.5);
            }
        });

        self.op_manager = Some(manager.into());
        self.state = Some(state);
    }

    /// Take a single diagonal step.
    pub fn single_diagonal_step(&mut self, beta: f64) {
        let mut state = self.state.take().unwrap();
        let mut manager = self.op_manager.take().unwrap();
        let rng = self.rng.as_mut().unwrap();

        let nvars = state.len();
        let edges = &self.edges;
        let vars = &self.vars;
        let transverse = self.transverse;
        let twosite_energy_offset = self.twosite_energy_offset;
        let singlesite_energy_offset = self.singlesite_energy_offset;
        let hinfo = HamInfo {
            edges,
            transverse,
            singlesite_energy_offset,
            twosite_energy_offset,
        };
        let h = |vars: &[usize], bond: usize, input_state: &[bool], output_state: &[bool]| {
            Self::hamiltonian(&hinfo, vars, bond, input_state, output_state)
        };

        let num_bonds = edges.len() + nvars;
        let bonds_fn = |b: usize| -> &[usize] {
            if b < edges.len() {
                &edges[b].0
            } else {
                let b = b - edges.len();
                &vars[b..b + 1]
            }
        };

        // Start by editing the ops list
        let ham = Hamiltonian::new(h, bonds_fn, num_bonds);
        manager.make_diagonal_update_with_rng_and_state_ref(
            self.cutoff,
            beta,
            &mut state,
            &ham,
            rng,
        );
        self.cutoff = max(self.cutoff, manager.get_n() + manager.get_n() / 2);
        self.op_manager = Some(manager);
        self.state = Some(state);
    }

    /// Take a single offdiagonal step.
    pub fn single_offdiagonal_step(&mut self) {
        let mut state = self.state.take().unwrap();
        let manager = self.op_manager.take().unwrap();
        let rng = self.rng.as_mut().unwrap();

        // Start by editing the ops list
        let mut manager = manager.into();
        manager.flip_each_cluster_rng(0.5, rng, &mut state);

        state.iter_mut().enumerate().for_each(|(var, state)| {
            if !manager.does_var_have_ops(var) {
                *state = rng.gen_bool(0.5);
            }
        });

        self.op_manager = Some(manager.into());
        self.state = Some(state);
    }

    /// Calculate the autcorrelation calculations for variables.
    #[cfg(feature = "autocorrelations")]
    pub fn calculate_variable_autocorrelation(
        &mut self,
        timesteps: usize,
        beta: f64,
        sampling_freq: Option<usize>,
        use_fft: Option<bool>,
    ) -> Vec<f64> {
        self.calculate_autocorrelation(timesteps, beta, sampling_freq, use_fft, |sample| {
            sample
                .into_iter()
                .map(|b| if b { 1.0 } else { -1.0 })
                .collect()
        })
    }

    /// Calculate the autcorrelation calculations for bonds.
    #[cfg(feature = "autocorrelations")]
    pub fn calculate_bond_autocorrelation(
        &mut self,
        timesteps: usize,
        beta: f64,
        sampling_freq: Option<usize>,
        use_fft: Option<bool>,
    ) -> Vec<f64> {
        let edges = self.edges.clone();
        self.calculate_autocorrelation(timesteps, beta, sampling_freq, use_fft, |sample| {
            edges
                .iter()
                .map(|(edge, j)| -> f64 {
                    let even = edge.iter().cloned().filter(|i| sample[*i]).count() % 2 == 0;
                    let b = if *j < 0.0 { even } else { !even };
                    if b {
                        1.0
                    } else {
                        -1.0
                    }
                })
                .collect()
        })
    }

    /// Calculate the autcorrelation calculations for the results of f(state).
    #[cfg(feature = "autocorrelations")]
    pub fn calculate_autocorrelation<F>(
        &mut self,
        timesteps: usize,
        beta: f64,
        sampling_freq: Option<usize>,
        use_fft: Option<bool>,
        sample_mapper: F,
    ) -> Vec<f64>
    where
        F: Fn(Vec<bool>) -> Vec<f64>,
    {
        let acc = Vec::with_capacity(timesteps / sampling_freq.unwrap_or(1) + 1);
        let (samples, _) = self.timesteps_measure(
            timesteps,
            beta,
            acc,
            |mut acc, state| {
                acc.push(state.to_vec());
                acc
            },
            sampling_freq,
        );
        let samples = samples
            .into_iter()
            .map(sample_mapper)
            .collect::<Vec<Vec<f64>>>();

        if use_fft.unwrap_or(true) {
            autocorrelations::fft_autocorrelation(&samples)
        } else {
            autocorrelations::naive_autocorrelation(&samples)
        }
    }

    /// Prinbt debug output.
    pub fn print_debug(&self) {
        debug_print_diagonal(
            self.op_manager.as_ref().unwrap(),
            self.state.as_ref().unwrap(),
        )
    }

    /// Verify the integrity of the graph.
    pub fn verify(&self) -> bool {
        self.op_manager
            .as_ref()
            .and_then(|op| self.state.as_ref().map(|state| (op, state)))
            .map(|(op, state)| op.verify(state))
            .unwrap_or(false)
    }

    /// Get a reference to the state at p=0.
    pub fn state_ref(&self) -> &Vec<bool> {
        self.state.as_ref().unwrap()
    }

    /// Get a mutable reference to the state at p=0 (can break integrity)
    pub fn state_mut(&mut self) -> &mut Vec<bool> {
        self.state.as_mut().unwrap()
    }

    /// Clone the state at p=0.
    pub fn clone_state(&self) -> Vec<bool> {
        self.state.as_ref().unwrap().clone()
    }

    /// Convert the state to a vector.
    pub fn into_vec(self) -> Vec<bool> {
        self.state.unwrap()
    }

    /// Get the number of variables in the graph.
    pub fn get_nvars(&self) -> usize {
        self.vars.len()
    }

    /// Get the cutoff used for qmc calculations (pmax).
    pub fn get_cutoff(&self) -> usize {
        self.cutoff
    }

    /// Get the number of ops in graph.
    pub fn get_n(&self) -> usize {
        self.op_manager.as_ref().unwrap().get_n()
    }

    /// Get a reference to the op manager.
    pub fn get_manager_ref(&self) -> &M {
        self.op_manager.as_ref().unwrap()
    }

    /// Get a mutable reference to the op manager.
    pub fn get_manager_mut(&mut self) -> &mut M {
        self.op_manager.as_mut().unwrap()
    }
}

fn two_site_hamiltonian(
    inputs: (bool, bool),
    outputs: (bool, bool),
    bond: f64,
    energy_offset: f64,
) -> f64 {
    let matentry = if inputs == outputs {
        energy_offset
            + match inputs {
                (false, false) => -bond,
                (false, true) => bond,
                (true, false) => bond,
                (true, true) => -bond,
            }
    } else {
        0.0
    };
    assert!(matentry >= 0.0);
    matentry
}

fn single_site_hamiltonian(
    input_state: bool,
    output_state: bool,
    transverse: f64,
    energy_offset: f64,
) -> f64 {
    match (input_state, output_state) {
        (false, false) | (true, true) => energy_offset,
        (false, true) | (true, false) => transverse,
    }
}

/// Data required to evaluate the hamiltonian.
#[derive(Debug)]
pub struct HamInfo<'a> {
    edges: &'a [(VecEdge, f64)],
    transverse: f64,
    singlesite_energy_offset: f64,
    twosite_energy_offset: f64,
}

impl<'a> HamInfo<'a> {
    /// Check if the hamiltonians (Assumed to be on the same graph) are equal.
    pub fn equal_assuming_graph(&self, other: &HamInfo) -> bool {
        (self.transverse - other.transverse).abs() < f64::EPSILON
    }
}

// Implement clone where available.
impl<
        R: Rng + Clone,
        M: OpContainerConstructor + DiagonalUpdater + Into<L> + Clone,
        L: ClusterUpdater + Into<M> + Clone,
    > Clone for QMCIsingGraph<R, M, L>
{
    fn clone(&self) -> Self {
        Self {
            edges: self.edges.clone(),
            transverse: self.transverse,
            state: self.state.clone(),
            cutoff: self.cutoff,
            op_manager: self.op_manager.clone(),
            twosite_energy_offset: self.twosite_energy_offset,
            singlesite_energy_offset: self.singlesite_energy_offset,
            rng: self.rng.clone(),
            phantom: self.phantom,
            vars: self.vars.clone(),
            state_updates: self.state_updates.clone(),
        }
    }
}

#[cfg(feature = "autocorrelations")]
pub(crate) mod autocorrelations {
    use rayon::prelude::*;
    use rustfft::num_complex::Complex;
    use rustfft::num_traits::Zero;
    use rustfft::FFTplanner;
    use std::ops::DivAssign;

    pub(crate) fn fft_autocorrelation(samples: &[Vec<f64>]) -> Vec<f64> {
        let tmax = samples.len();
        let n = samples[0].len();

        let means = (0..n)
            .map(|i| (0..tmax).map(|t| samples[t][i]).sum::<f64>() / tmax as f64)
            .collect::<Vec<_>>();

        let mut input = (0..n)
            .map(|i| {
                let mut v = (0..tmax)
                    .map(|t| Complex::<f64>::new(samples[t][i] - means[i], 0.0))
                    .collect::<Vec<Complex<f64>>>();
                let norm = v.iter().map(|v| v.powi(2).re).sum::<f64>().sqrt();
                v.iter_mut().for_each(|c| c.div_assign(norm));
                v
            })
            .collect::<Vec<_>>();
        let mut planner = FFTplanner::new(false);
        let fft = planner.plan_fft(tmax);
        let mut iplanner = FFTplanner::new(true);
        let ifft = iplanner.plan_fft(tmax);

        let mut output = vec![Complex::zero(); tmax];
        input.iter_mut().for_each(|input| {
            fft.process(input, &mut output);
            output
                .iter_mut()
                .for_each(|c| *c = Complex::new(c.norm_sqr(), 0.0));
            ifft.process(&mut output, input);
        });

        (0..tmax)
            .map(|t| (0..n).map(|i| input[i][t].re).sum::<f64>() / ((n * tmax) as f64))
            .collect()
    }

    pub(crate) fn naive_autocorrelation(samples: &[Vec<f64>]) -> Vec<f64> {
        let tmax = samples.len();
        let n: usize = samples[0].len();
        let mu = (0..n)
            .map(|i| -> f64 {
                let total = samples.iter().map(|sample| sample[i]).sum::<f64>();
                total / samples.len() as f64
            })
            .collect::<Vec<_>>();

        (0..tmax)
            .into_par_iter()
            .map(|tau| {
                (0..tmax)
                    .map(|t| (t, (t + tau) % tmax))
                    .map(|(ta, tb)| {
                        let sample_a = &samples[ta];
                        let sample_b = &samples[tb];
                        let (d, ma, mb) = sample_a
                            .iter()
                            .enumerate()
                            .zip(sample_b.iter().enumerate())
                            .fold(
                                (0.0, 0.0, 0.0),
                                |(mut dot_acc, mut a_acc, mut b_acc), ((i, a), (j, b))| {
                                    let da = a - mu[i];
                                    let db = b - mu[j];
                                    dot_acc += da * db;
                                    a_acc += da.powi(2);
                                    b_acc += db.powi(2);
                                    (dot_acc, a_acc, b_acc)
                                },
                            );
                        d / (ma * mb).sqrt()
                    })
                    .sum::<f64>()
                    / (tmax as f64)
            })
            .collect::<Vec<_>>()
    }
}

/// Structs for easy serialization.
#[cfg(feature = "serialize")]
pub mod serialization {
    use super::*;

    /// The serializable version of the default QMC graph.
    pub type DefaultSerializeQMCGraph = SerializeQMCGraph<FastOps, FastOps>;

    /// A QMC graph without rng for easy serialization.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct SerializeQMCGraph<
        M: OpContainerConstructor + DiagonalUpdater + Into<L>,
        L: ClusterUpdater + Into<M>,
    > {
        edges: Vec<(VecEdge, f64)>,
        transverse: f64,
        state: Option<Vec<bool>>,
        cutoff: usize,
        op_manager: Option<M>,
        twosite_energy_offset: f64,
        singlesite_energy_offset: f64,
        phantom: PhantomData<L>,
        // Can be easily reconstructed
        nvars: usize,
        nstate_updates: usize,
    }

    impl<M: OpContainerConstructor + DiagonalUpdater + Into<L>, L: ClusterUpdater + Into<M>>
        SerializeQMCGraph<M, L>
    {
        /// Convert into a proper QMC graph using a new rng instance.
        pub fn into_qmc<R: Rng>(self, rng: R) -> QMCIsingGraph<R, M, L> {
            QMCIsingGraph {
                edges: self.edges,
                transverse: self.transverse,
                state: self.state,
                cutoff: self.cutoff,
                op_manager: self.op_manager,
                twosite_energy_offset: self.twosite_energy_offset,
                singlesite_energy_offset: self.singlesite_energy_offset,
                rng: Some(rng),
                phantom: self.phantom,
                vars: (0..self.nvars).collect(),
                state_updates: Vec::with_capacity(self.nstate_updates),
            }
        }
    }

    impl<
            R: Rng,
            M: OpContainerConstructor + DiagonalUpdater + Into<L>,
            L: ClusterUpdater + Into<M>,
        > Into<SerializeQMCGraph<M, L>> for QMCIsingGraph<R, M, L>
    {
        fn into(self) -> SerializeQMCGraph<M, L> {
            SerializeQMCGraph {
                edges: self.edges,
                transverse: self.transverse,
                state: self.state,
                cutoff: self.cutoff,
                op_manager: self.op_manager,
                twosite_energy_offset: self.twosite_energy_offset,
                singlesite_energy_offset: self.singlesite_energy_offset,
                phantom: self.phantom,
                nvars: self.vars.len(),
                nstate_updates: self.state_updates.len(),
            }
        }
    }

    #[cfg(test)]
    mod serialize_test {
        use super::*;
        use rand::prelude::SmallRng;
        use rand::SeedableRng;
        use rand_isaac::IsaacRng;

        #[test]
        fn test_serialize() {
            let rng = IsaacRng::seed_from_u64(1234);
            let mut g = DefaultQMCIsingGraph::<IsaacRng>::new_with_rng(
                vec![((0, 1), 1.0)],
                1.0,
                1,
                rng,
                None,
            );
            g.timesteps(100, 1.0);
            let mut v: Vec<u8> = Vec::default();
            serde_json::to_writer_pretty(&mut v, &g).unwrap();
            let _: DefaultQMCIsingGraph<IsaacRng> = serde_json::from_slice(&v).unwrap();
        }

        #[test]
        fn test_serialize_no_rng() {
            let rng = SmallRng::seed_from_u64(1234);
            let mut g = DefaultQMCIsingGraph::<SmallRng>::new_with_rng(
                vec![((0, 1), 1.0)],
                1.0,
                1,
                rng,
                None,
            );
            g.timesteps(100, 1.0);
            let mut v: Vec<u8> = Vec::default();
            let sg: DefaultSerializeQMCGraph = g.into();
            serde_json::to_writer_pretty(&mut v, &sg).unwrap();

            let rng = SmallRng::seed_from_u64(1234);
            let sg: DefaultSerializeQMCGraph = serde_json::from_slice(&v).unwrap();
            let _g: DefaultQMCIsingGraph<SmallRng> = sg.into_qmc(rng);
        }
    }
}
