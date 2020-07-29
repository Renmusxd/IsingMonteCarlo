use crate::graph::{Edge, GraphState};
use crate::sse::fast_ops::{FastOpNode, FastOps};
use crate::sse::qmc_traits::*;
use rand::rngs::ThreadRng;
use rand::Rng;
use std::cmp::max;
use std::marker::PhantomData;

pub type DefaultQMCGraph<R> = QMCGraph<R, FastOpNode, FastOps, FastOps>;

type VecEdge = Vec<usize>;
pub struct QMCGraph<
    R: Rng,
    N: OpNode,
    M: OpContainerConstructor + DiagonalUpdater + Into<L>,
    L: LoopUpdater<N> + ClusterUpdater<N> + Into<M>,
> {
    edges: Vec<(VecEdge, f64)>,
    transverse: f64,
    state: Option<Vec<bool>>,
    cutoff: usize,
    op_manager: Option<M>,
    twosite_energy_offset: f64,
    singlesite_energy_offset: f64,
    rng: Option<R>,
    phantom: PhantomData<(L, N)>,
    // This is just an array of the variables 0..nvars
    vars: Vec<usize>,
    // An alloc to reuse in cluster updates
    state_updates: Vec<(usize, bool)>,
}

pub fn new_qmc(
    edges: Vec<(Edge, f64)>,
    transverse: f64,
    cutoff: usize,
    state: Option<Vec<bool>>,
) -> DefaultQMCGraph<ThreadRng> {
    let rng = rand::thread_rng();
    DefaultQMCGraph::<ThreadRng>::new_with_rng(edges, transverse, cutoff, rng, state)
}

pub fn new_qmc_from_graph(
    graph: GraphState,
    transverse: f64,
    cutoff: usize,
) -> DefaultQMCGraph<ThreadRng> {
    let rng = rand::thread_rng();
    DefaultQMCGraph::<ThreadRng>::new_from_graph(graph, transverse, cutoff, rng)
}

impl<
        R: Rng,
        N: OpNode,
        M: OpContainerConstructor + DiagonalUpdater + Into<L>,
        L: LoopUpdater<N> + ClusterUpdater<N> + Into<M>,
    > QMCGraph<R, N, M, L>
{
    pub fn new_with_rng<Rg: Rng>(
        edges: Vec<(Edge, f64)>,
        transverse: f64,
        cutoff: usize,
        rng: Rg,
        state: Option<Vec<bool>>,
    ) -> QMCGraph<Rg, N, M, L> {
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

        QMCGraph::<Rg, N, M, L> {
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

    pub fn new_from_graph<Rg: Rng>(
        graph: GraphState,
        transverse: f64,
        cutoff: usize,
        rng: Rg,
    ) -> QMCGraph<Rg, N, M, L> {
        assert!(graph.biases.into_iter().all(|v| v == 0.0));
        Self::new_with_rng(graph.edges, transverse, cutoff, rng, graph.state)
    }

    pub fn timesteps(&mut self, t: usize, beta: f64) -> f64 {
        let (_, average_energy) = self.timesteps_measure(t, beta, (), |_acc, _state| (), None);
        average_energy
    }

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

    pub fn make_haminfo(&self) -> HamInfo {
        HamInfo {
            edges: &self.edges,
            transverse: self.transverse,
            singlesite_energy_offset: self.singlesite_energy_offset,
            twosite_energy_offset: self.twosite_energy_offset,
        }
    }

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
        let state_changes = &mut self.state_updates;
        state_changes.clear();
        manager.flip_each_cluster_rng_to_acc(0.5, rng, state_changes);
        state_changes.iter().cloned().for_each(|(i, v)| {
            state[i] = v;
        });

        state.iter_mut().enumerate().for_each(|(var, state)| {
            if !manager.does_var_have_ops(var) {
                *state = rng.gen_bool(0.5);
            }
        });

        self.op_manager = Some(manager.into());
        self.state = Some(state);
    }

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

    pub fn print_debug(&self) {
        debug_print_diagonal(
            self.op_manager.as_ref().unwrap(),
            self.state.as_ref().unwrap(),
        )
    }

    pub fn verify(&self) -> bool {
        self.op_manager
            .as_ref()
            .and_then(|op| self.state.as_ref().map(|state| (op, state)))
            .map(|(op, state)| op.verify(state))
            .unwrap_or(false)
    }

    pub fn state_ref(&self) -> &Vec<bool> {
        self.state.as_ref().unwrap()
    }

    pub fn state_mut(&mut self) -> &mut Vec<bool> {
        self.state.as_mut().unwrap()
    }

    pub fn clone_state(&self) -> Vec<bool> {
        self.state.as_ref().unwrap().clone()
    }

    pub fn into_vec(self) -> Vec<bool> {
        self.state.unwrap()
    }

    pub fn get_nvars(&self) -> usize {
        self.vars.len()
    }

    pub fn get_cutoff(&self) -> usize {
        self.cutoff
    }

    pub fn get_manager_ref(&self) -> &M {
        self.op_manager.as_ref().unwrap()
    }

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

pub struct HamInfo<'a> {
    edges: &'a [(VecEdge, f64)],
    transverse: f64,
    singlesite_energy_offset: f64,
    twosite_energy_offset: f64,
}

// Implement clone where available.
impl<
        R: Rng + Clone,
        N: OpNode + Clone,
        M: OpContainerConstructor + DiagonalUpdater + Into<L> + Clone,
        L: LoopUpdater<N> + ClusterUpdater<N> + Into<M> + Clone,
    > Clone for QMCGraph<R, N, M, L>
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

    pub fn fft_autocorrelation(samples: &[Vec<f64>]) -> Vec<f64> {
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

    pub fn naive_autocorrelation(samples: &[Vec<f64>]) -> Vec<f64> {
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
