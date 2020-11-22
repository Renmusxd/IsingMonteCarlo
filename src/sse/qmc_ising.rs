use crate::classical::graph::{Edge, GraphState};
#[cfg(feature = "autocorrelations")]
pub use crate::sse::autocorrelations::*;
use crate::sse::fast_ops::FastOps;
use crate::sse::ham::Ham;
use crate::sse::qmc_runner::{QMCManager, QMC};
pub use crate::sse::qmc_traits::*;
use rand::rngs::ThreadRng;
use rand::Rng;
#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};
use std::cmp::max;

/// Default QMC graph implementation.
pub type DefaultQMCIsingGraph<R> = QMCIsingGraph<R, FastOps>;

type VecEdge = Vec<usize>;

/// Trait encompassing all requirements for op managers in QMCIsingGraph.
pub trait IsingManager:
    OpContainerConstructor + HeatBathDiagonalUpdater + RVBUpdater + ClusterUpdater
{
}

/// A container to run QMC simulations.
#[derive(Debug)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct QMCIsingGraph<R: Rng, M: IsingManager> {
    edges: Vec<(VecEdge, f64)>,
    transverse: f64,
    state: Option<Vec<bool>>,
    cutoff: usize,
    op_manager: Option<M>,
    total_energy_offset: f64,
    rng: Option<R>,
    // This is just an array of the variables 0..nvars
    vars: Vec<usize>,
    run_rvb_steps: bool,
    // List of bonds.
    classical_bonds: Option<Vec<Vec<usize>>>,
    total_rvb_successes: usize,
    rvb_clusters_counted: usize,
    // Heatbath bond weights
    bond_weights: Option<BondWeights>,
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

impl<R: Rng, M: IsingManager> QMCIsingGraph<R, M> {
    /// Make a new QMC graph with an rng instance.
    pub fn new_with_rng<Rg: Rng>(
        edges: Vec<(Edge, f64)>,
        transverse: f64,
        cutoff: usize,
        rng: Rg,
        state: Option<Vec<bool>>,
    ) -> QMCIsingGraph<Rg, M> {
        let nvars = edges.iter().map(|((a, b), _)| max(*a, *b)).max().unwrap() + 1;
        let edges = edges
            .into_iter()
            .map(|((a, b), j)| (vec![a, b], j))
            .collect::<Vec<_>>();
        let edge_offset = edges.iter().map(|(_, j)| j.abs()).sum::<f64>();
        let field_offset = nvars as f64 * transverse;
        let total_energy_offset = edge_offset + field_offset;

        let mut ops = M::new_with_bonds(nvars, edges.len() + nvars);
        ops.set_cutoff(cutoff);

        let state = match state {
            Some(state) => state,
            None => GraphState::make_random_spin_state(nvars),
        };
        let state = Some(state);

        QMCIsingGraph::<Rg, M> {
            edges,
            transverse,
            state,
            op_manager: Some(ops),
            cutoff,
            total_energy_offset,
            rng: Some(rng),
            vars: (0..nvars).collect(),
            run_rvb_steps: false,
            classical_bonds: None,
            total_rvb_successes: 0,
            rvb_clusters_counted: 0,
            bond_weights: None,
        }
    }

    /// Make a new QMC graph with an rng instance.
    pub fn new_from_graph<Rg: Rng>(
        graph: GraphState,
        transverse: f64,
        cutoff: usize,
        rng: Rg,
    ) -> QMCIsingGraph<Rg, M> {
        assert!(graph.biases.into_iter().all(|v| v == 0.0));
        Self::new_with_rng(graph.edges, transverse, cutoff, rng, graph.state)
    }

    /// Make the hamiltonian struct.
    pub fn make_haminfo(&self) -> HamInfo {
        HamInfo {
            edges: &self.edges,
            transverse: self.transverse,
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
            1 => single_site_hamiltonian(input_state[0], output_state[0], info.transverse),
            2 => two_site_hamiltonian(
                (input_state[0], input_state[1]),
                (output_state[0], output_state[1]),
                info.edges[bond].1,
            ),
            _ => unreachable!(),
        }
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
        let hinfo = HamInfo { edges, transverse };
        let h = |vars: &[usize], bond: usize, input_state: &[bool], output_state: &[bool]| {
            Self::hamiltonian(&hinfo, vars, bond, input_state, output_state)
        };

        let num_bonds = edges.len() + nvars;
        let bonds_fn = |b: usize| -> (&[usize], bool) {
            if b < edges.len() {
                (&edges[b].0, false)
            } else {
                let b = b - edges.len();
                (&vars[b..b + 1], true)
            }
        };

        // Start by editing the ops list
        let ham = Ham::new(h, bonds_fn, num_bonds);

        if let Some(bond_weights) = &self.bond_weights {
            manager.make_heatbath_diagonal_update_with_rng_and_state_ref(
                self.cutoff,
                beta,
                &mut state,
                &ham,
                bond_weights,
                rng,
            );
        } else {
            manager.make_diagonal_update_with_rng_and_state_ref(
                self.cutoff,
                beta,
                &mut state,
                &ham,
                rng,
            );
        }

        self.cutoff = max(self.cutoff, manager.get_n() + manager.get_n() / 2);
        self.op_manager = Some(manager);
        self.state = Some(state);
    }

    /// Take a single offdiagonal step.
    pub fn single_offdiagonal_step(&mut self) {
        let mut state = self.state.take().unwrap();
        let mut manager = self.op_manager.take().unwrap();
        let rng = self.rng.as_mut().unwrap();

        // Start by editing the ops list
        manager.flip_each_cluster_ising_symmetry_rng(0.5, rng, &mut state);

        state.iter_mut().enumerate().for_each(|(var, state)| {
            if !manager.does_var_have_ops(var) {
                *state = rng.gen_bool(0.5);
            }
        });

        self.op_manager = Some(manager);
        self.state = Some(state);
    }

    /// Perform a single rvb step.
    pub fn single_rvb_step(&mut self) -> Result<(), String> {
        let mut state = self.state.take().unwrap();
        if self.classical_bonds.is_none() {
            self.make_classical_bonds(state.len())?;
        }
        let mut manager = self.op_manager.take().unwrap();
        let rng = self.rng.as_mut().unwrap();

        let nvars = state.len();
        let edges = &self.edges;
        let vars = &self.vars;
        let transverse = self.transverse;
        let hinfo = HamInfo { edges, transverse };
        let h = |vars: &[usize], bond: usize, input_state: &[bool], output_state: &[bool]| {
            Self::hamiltonian(&hinfo, vars, bond, input_state, output_state)
        };

        let num_bonds = edges.len() + nvars;
        let bonds_fn = |b: usize| -> (&[usize], bool) {
            if b < edges.len() {
                (&edges[b].0, false)
            } else {
                let b = b - edges.len();
                (&vars[b..b + 1], true)
            }
        };

        // Start by editing the ops list
        let ham = Ham::new(h, bonds_fn, num_bonds);

        let edges = EdgeNav {
            var_to_bonds: self.classical_bonds.as_ref().unwrap(),
            edges: &self.edges,
        };

        manager.rvb_update(
            &edges,
            &mut state,
            1,
            |bond, sa, sb| {
                let (va, vb) = edges.vars_for_bond(bond);
                ham.hamiltonian(&[va, vb], bond, &[sa, sb], &[sa, sb])
            },
            rng,
        );

        self.op_manager = Some(manager);
        self.state = Some(state);
        Ok(())
    }

    /// Build classical bonds list.
    fn make_classical_bonds(&mut self, nvars: usize) -> Result<(), String> {
        let mut edge_lookup = vec![vec![]; nvars];
        self.edges
            .iter()
            .map(|(edge, _)| (edge[0], edge[1]))
            .enumerate()
            .for_each(|(bond, (a, b))| {
                edge_lookup[a].push(bond);
                edge_lookup[b].push(bond);
            });
        self.classical_bonds = Some(edge_lookup);
        Ok(())
    }

    /// Enable or disable automatic rvb steps. Errors if all js not equal magnitude.
    pub fn set_run_rvb(&mut self, run_rvb: bool) -> Result<(), String> {
        self.run_rvb_steps = run_rvb;
        if run_rvb && self.classical_bonds.is_none() {
            let nvars = self.state.as_ref().map(|s| s.len()).unwrap();
            self.make_classical_bonds(nvars)
        } else {
            Ok(())
        }
    }

    /// Enable heatbath diagonal updates.
    pub fn set_enable_heatbath(&mut self, enable_heatbath: bool) {
        if enable_heatbath {
            let nvars = self.get_nvars();
            let edges = &self.edges;
            let vars = &self.vars;
            let transverse = self.transverse;
            let hinfo = HamInfo { edges, transverse };
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
            let bw = M::make_bond_weights(h, num_bonds, bonds_fn);
            self.bond_weights = Some(bw);
        } else {
            self.bond_weights = None;
        }
    }

    /// Print debug output.
    pub fn print_debug(&self) {
        debug_print_diagonal(
            self.op_manager.as_ref().unwrap(),
            self.state.as_ref().unwrap(),
        )
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

    /// Get the edges on the graph
    pub fn get_edges(&self) -> &[(VecEdge, f64)] {
        &self.edges
    }

    /// Get the transverse field on the system.
    pub fn get_transverse_field(&self) -> f64 {
        self.transverse
    }

    /// Get the cutoff used for qmc calculations (pmax).
    pub fn get_cutoff(&self) -> usize {
        self.cutoff
    }

    /// Set the cutoff.
    pub fn set_cutoff(&mut self, cutoff: usize) {
        self.cutoff = cutoff;
        self.op_manager.as_mut().unwrap().set_cutoff(cutoff)
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

    /// Get internal energy offset.
    pub fn get_offset(&self) -> f64 {
        self.total_energy_offset
    }

    /// Check if two instances can safely swap managers and initial states
    pub fn can_swap_managers(&self, other: &Self) -> bool {
        self.edges == other.edges
    }

    /// Swap managers and initial states
    pub fn swap_manager_and_state(&mut self, other: &mut Self) {
        let m = self.op_manager.take().unwrap();
        let s = self.state.take().unwrap();
        let om = other.op_manager.take().unwrap();
        let os = other.state.take().unwrap();
        self.op_manager = Some(om);
        self.state = Some(os);
        other.op_manager = Some(m);
        other.state = Some(s);
    }

    /// Average rvb success rate.
    pub fn rvb_success_rate(&self) -> f64 {
        self.total_rvb_successes as f64 / self.rvb_clusters_counted as f64
    }
}

struct EdgeNav<'a, 'b> {
    var_to_bonds: &'a [Vec<usize>],
    edges: &'b [(VecEdge, f64)],
}

impl<'a, 'b> EdgeNavigator for EdgeNav<'a, 'b> {
    fn n_bonds(&self) -> usize {
        self.edges.len()
    }

    fn bonds_for_var(&self, var: usize) -> &[usize] {
        &self.var_to_bonds[var]
    }

    fn vars_for_bond(&self, bond: usize) -> (usize, usize) {
        let e = &self.edges[bond].0;
        (e[0], e[1])
    }

    fn bond_prefers_aligned(&self, bond: usize) -> bool {
        self.edges[bond].1 < 0.0
    }
}

impl<R, M> QMCStepper for QMCIsingGraph<R, M>
where
    R: Rng,
    M: IsingManager,
{
    /// Perform a single step of qmc.
    fn timestep(&mut self, beta: f64) -> &[bool] {
        let mut state = self.state.take().unwrap();
        let mut manager = self.op_manager.take().unwrap();
        let mut rng = self.rng.take().unwrap();

        let nvars = state.len();
        let edges = &self.edges;
        let vars = &self.vars;
        let transverse = self.transverse;
        let hinfo = HamInfo { edges, transverse };
        let h = |vars: &[usize], bond: usize, input_state: &[bool], output_state: &[bool]| {
            Self::hamiltonian(&hinfo, vars, bond, input_state, output_state)
        };

        let num_bonds = edges.len() + nvars;
        let bonds_fn = |b: usize| -> (&[usize], bool) {
            if b < edges.len() {
                (&edges[b].0, false)
            } else {
                let b = b - edges.len();
                (&vars[b..b + 1], true)
            }
        };

        // Start by editing the ops list
        let ham = Ham::new(h, bonds_fn, num_bonds);
        if let Some(bond_weights) = &self.bond_weights {
            manager.make_heatbath_diagonal_update_with_rng_and_state_ref(
                self.cutoff,
                beta,
                &mut state,
                &ham,
                bond_weights,
                &mut rng,
            )
        } else {
            manager.make_diagonal_update_with_rng_and_state_ref(
                self.cutoff,
                beta,
                &mut state,
                &ham,
                &mut rng,
            );
        };
        self.cutoff = max(self.cutoff, manager.get_n() + manager.get_n() / 2);

        if self.run_rvb_steps {
            let edges = EdgeNav {
                var_to_bonds: self.classical_bonds.as_ref().unwrap(),
                edges: &self.edges,
            };
            // Average cluster size is always 2.
            let steps_to_run = (state.len() + 1) / 2;

            let succs = manager.rvb_update(
                &edges,
                &mut state,
                steps_to_run,
                |bond, sa, sb| {
                    let (va, vb) = edges.vars_for_bond(bond);
                    ham.hamiltonian(&[va, vb], bond, &[sa, sb], &[sa, sb])
                },
                &mut rng,
            );
            self.total_rvb_successes += succs;
            self.rvb_clusters_counted += steps_to_run;
        }

        manager.flip_each_cluster_ising_symmetry_rng(0.5, &mut rng, &mut state);

        state.iter_mut().enumerate().for_each(|(var, state)| {
            if !manager.does_var_have_ops(var) {
                *state = rng.gen_bool(0.5);
            }
        });

        self.rng = Some(rng);
        self.op_manager = Some(manager);
        self.state = Some(state);

        debug_assert!(self.verify());

        self.state.as_ref().unwrap()
    }

    fn state_ref(&self) -> &[bool] {
        self.state.as_ref().unwrap()
    }

    fn get_n(&self) -> usize {
        self.op_manager.as_ref().unwrap().get_n()
    }

    fn get_energy_for_average_n(&self, average_n: f64, beta: f64) -> f64 {
        let average_energy = -(average_n / beta);
        let offset = self.get_offset();
        average_energy + offset
    }
}

impl<R, M> Verify for QMCIsingGraph<R, M>
where
    R: Rng,
    M: IsingManager,
{
    fn verify(&self) -> bool {
        self.op_manager
            .as_ref()
            .zip(self.state.as_ref())
            .map(|(m, state)| m.verify(state))
            .unwrap_or(false)
    }
}

fn two_site_hamiltonian(inputs: (bool, bool), outputs: (bool, bool), bond: f64) -> f64 {
    let matentry = if inputs == outputs {
        bond.abs()
            + match inputs {
                (false, false) => -bond,
                (false, true) => bond,
                (true, false) => bond,
                (true, true) => -bond,
            }
    } else {
        0.0
    };
    debug_assert!(matentry >= 0.0);
    matentry
}

fn single_site_hamiltonian(_input_state: bool, _output_state: bool, transverse: f64) -> f64 {
    transverse
}

/// Data required to evaluate the hamiltonian.
#[derive(Debug)]
pub struct HamInfo<'a> {
    edges: &'a [(VecEdge, f64)],
    transverse: f64,
}

impl<'a> PartialEq for HamInfo<'a> {
    fn eq(&self, other: &Self) -> bool {
        self.edges == other.edges && self.transverse == other.transverse
    }
}

impl<'a> Eq for HamInfo<'a> {}

// Implement clone where available.
impl<R, M> Clone for QMCIsingGraph<R, M>
where
    R: Rng + Clone,
    M: IsingManager + Clone,
{
    fn clone(&self) -> Self {
        Self {
            edges: self.edges.clone(),
            transverse: self.transverse,
            state: self.state.clone(),
            cutoff: self.cutoff,
            op_manager: self.op_manager.clone(),
            total_energy_offset: self.total_energy_offset,
            rng: self.rng.clone(),
            vars: self.vars.clone(),
            run_rvb_steps: self.run_rvb_steps,
            classical_bonds: self.classical_bonds.clone(),
            total_rvb_successes: self.total_rvb_successes,
            rvb_clusters_counted: self.rvb_clusters_counted,
            bond_weights: self.bond_weights.clone(),
        }
    }
}

/// Convertable into QMC, helps since calling .into() runs into type inference problems.
pub trait IntoQMC<R, M>
where
    R: Rng,
    M: QMCManager,
{
    /// Convert into QMC.
    fn into_qmc(self) -> QMC<R, M>;
}

impl<R, M> IntoQMC<R, M> for QMCIsingGraph<R, M>
where
    R: Rng,
    M: IsingManager + QMCManager,
{
    fn into_qmc(self) -> QMC<R, M> {
        let nvars = self.get_nvars();
        let rng = self.rng.unwrap();
        let state = self.state.as_ref().unwrap().to_vec();
        let mut qmc = QMC::<R, M>::new_with_state(nvars, rng, state, false);
        let transverse = self.transverse;
        self.edges.into_iter().for_each(|(vars, j)| {
            qmc.make_diagonal_interaction_and_offset(vec![-j, j, j, -j], vars)
                .unwrap()
        });
        (0..nvars).for_each(|var| {
            qmc.make_interaction(
                vec![transverse, transverse, transverse, transverse],
                vec![var],
            )
            .unwrap()
        });
        qmc.increase_cutoff_to(self.cutoff);
        qmc.set_manager(self.op_manager.unwrap());
        qmc
    }
}

// Allow for conversion to generic QMC type. Clears the internal state, converts edges and field
// into interactions.
impl<R, M> Into<QMC<R, M>> for QMCIsingGraph<R, M>
where
    R: Rng,
    M: IsingManager + QMCManager,
{
    fn into(self) -> QMC<R, M> {
        self.into_qmc()
    }
}

#[cfg(feature = "autocorrelations")]
impl<R, M> QMCBondAutoCorrelations for QMCIsingGraph<R, M>
where
    R: Rng,
    M: IsingManager,
{
    fn n_bonds(&self) -> usize {
        self.edges.len()
    }

    fn value_for_bond(&self, bond: usize, sample: &[bool]) -> f64 {
        let (edge, j) = &self.edges[bond];
        let even = edge.iter().cloned().filter(|i| sample[*i]).count() % 2 == 0;
        let val = if *j < 0.0 { even } else { !even };
        if val {
            1.0
        } else {
            -1.0
        }
    }
}

/// Structs for easy serialization.
#[cfg(feature = "serialize")]
pub mod serialization {
    use super::*;

    /// The serializable version of the default QMC graph.
    pub type DefaultSerializeQMCGraph = SerializeQMCGraph<FastOps>;

    /// A QMC graph without rng for easy serialization.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct SerializeQMCGraph<M: IsingManager> {
        edges: Vec<(VecEdge, f64)>,
        transverse: f64,
        state: Option<Vec<bool>>,
        cutoff: usize,
        op_manager: Option<M>,
        total_energy_offset: f64,
        // Can be easily reconstructed
        nvars: usize,
        run_rvb_steps: bool,
        classical_bonds: Option<Vec<Vec<usize>>>,
        total_rvb_successes: usize,
        rvb_clusters_counted: usize,
        // Heatbath
        bond_weights: Option<BondWeights>,
    }

    impl<M> SerializeQMCGraph<M>
    where
        M: IsingManager,
    {
        /// Convert into a proper QMC graph using a new rng instance.
        pub fn into_qmc<R: Rng>(self, rng: R) -> QMCIsingGraph<R, M> {
            QMCIsingGraph {
                edges: self.edges,
                transverse: self.transverse,
                state: self.state,
                cutoff: self.cutoff,
                op_manager: self.op_manager,
                total_energy_offset: self.total_energy_offset,
                rng: Some(rng),
                vars: (0..self.nvars).collect(),
                run_rvb_steps: self.run_rvb_steps,
                classical_bonds: self.classical_bonds,
                total_rvb_successes: self.total_rvb_successes,
                rvb_clusters_counted: self.rvb_clusters_counted,
                bond_weights: self.bond_weights,
            }
        }
    }

    impl<R, M> Into<SerializeQMCGraph<M>> for QMCIsingGraph<R, M>
    where
        R: Rng,
        M: IsingManager,
    {
        fn into(self) -> SerializeQMCGraph<M> {
            SerializeQMCGraph {
                edges: self.edges,
                transverse: self.transverse,
                state: self.state,
                cutoff: self.cutoff,
                op_manager: self.op_manager,
                total_energy_offset: self.total_energy_offset,
                nvars: self.vars.len(),
                run_rvb_steps: self.run_rvb_steps,
                classical_bonds: self.classical_bonds,
                total_rvb_successes: self.total_rvb_successes,
                rvb_clusters_counted: self.rvb_clusters_counted,
                bond_weights: self.bond_weights,
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
