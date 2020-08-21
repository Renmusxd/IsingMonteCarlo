use crate::classical::graph::{Edge, GraphState};
#[cfg(feature = "autocorrelations")]
pub use crate::sse::autocorrelations::*;
use crate::sse::fast_ops::FastOps;
use crate::sse::qmc_runner::QMC;
pub use crate::sse::qmc_traits::*;
use crate::sse::semi_classical::*;
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
    M: OpContainerConstructor + ClassicalLoopUpdater + Into<L>,
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
    // Optional semiclassical update, for each var a: [(b, ferro/antiferro, bond)...]
    run_semiclassical_steps: bool,
    semiclassical_bonds: Option<Vec<Vec<usize>>>,
    total_cluster_size: f64,
    clusters_counted: usize,
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
        M: OpContainerConstructor + ClassicalLoopUpdater + Into<L>,
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
        let mut ops = M::new_with_bonds(nvars, edges.len() + nvars);
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
            run_semiclassical_steps: false,
            semiclassical_bonds: None,
            total_cluster_size: 2.0,
            clusters_counted: 0,
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
        let bonds_fn = |b: usize| -> (&[usize], bool) {
            if b < edges.len() {
                (&edges[b].0, false)
            } else {
                let b = b - edges.len();
                (&vars[b..b + 1], true)
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

    /// Perform a single semiclassical step.
    pub fn single_semiclassical_step(&mut self) {
        let mut state = self.state.take().unwrap();
        if self.semiclassical_bonds.is_none() {
            self.make_semiclassical_bonds(state.len());
        }
        let mut manager = self.op_manager.take().unwrap();
        let rng = self.rng.as_mut().unwrap();

        let edges = EdgeNav {
            var_to_bonds: self.semiclassical_bonds.as_ref().unwrap(),
            edges: &self.edges,
        };

        manager.run_classical_loop_update(&edges, &mut state, rng);

        self.op_manager = Some(manager);
        self.state = Some(state);
    }

    fn make_semiclassical_bonds(&mut self, nvars: usize) {
        let mut edge_lookup = vec![vec![]; nvars];
        self.edges
            .iter()
            .map(|(edge, _)| (edge[0], edge[1]))
            .enumerate()
            .for_each(|(bond, (a, b))| {
                edge_lookup[a].push(bond);
                edge_lookup[b].push(bond);
            });
        self.semiclassical_bonds = Some(edge_lookup);
    }

    /// Enable or disable automatic semiclassical steps.
    pub fn set_run_semiclassical(&mut self, run_semiclassical: bool) {
        self.run_semiclassical_steps = run_semiclassical;
        if self.run_semiclassical_steps && self.semiclassical_bonds.is_none() {
            let nvars = self.state.as_ref().map(|s| s.len()).unwrap();
            self.make_semiclassical_bonds(nvars);
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
        // Get total energy offset (num_vars * singlesite + num_edges * twosite)
        let twosite_energy_offset = self.twosite_energy_offset;
        let singlesite_energy_offset = self.singlesite_energy_offset;
        let nvars = self.vars.len();
        twosite_energy_offset * self.edges.len() as f64 + singlesite_energy_offset * nvars as f64
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

    /// Convert to a generic QMC instance.
    pub fn into_qmc(self) -> QMC<R, M, L> {
        let nvars = self.get_nvars();
        let rng = self.rng.unwrap();
        let state = self.state.as_ref().unwrap().to_vec();
        let mut qmc = QMC::<R, M, L>::new_with_state(nvars, rng, state, false);
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
        qmc.set_diagonal_manager(self.op_manager.unwrap());
        qmc
    }

    /// Average semiclassical cluster size.
    pub fn average_cluster_size(&self) -> f64 {
        self.total_cluster_size / self.clusters_counted as f64
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

impl<R, M, L> QMCStepper for QMCIsingGraph<R, M, L>
where
    R: Rng,
    M: OpContainerConstructor + ClassicalLoopUpdater + Into<L>,
    L: ClusterUpdater + Into<M>,
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
        let bonds_fn = |b: usize| -> (&[usize], bool) {
            if b < edges.len() {
                (&edges[b].0, false)
            } else {
                let b = b - edges.len();
                (&vars[b..b + 1], true)
            }
        };

        // Start by editing the ops list
        let ham = Hamiltonian::new(h, bonds_fn, num_bonds);
        manager.make_diagonal_update_with_rng_and_state_ref(
            self.cutoff,
            beta,
            &mut state,
            &ham,
            &mut rng,
        );
        self.cutoff = max(self.cutoff, manager.get_n() + manager.get_n() / 2);

        // Perform semiclassical steps if requested.
        if self.run_semiclassical_steps {
            let edges = EdgeNav {
                var_to_bonds: self.semiclassical_bonds.as_ref().unwrap(),
                edges: &self.edges,
            };
            // Each semi-classic update only hits a few variables so should be repeated.
            let average_cluster_size = self.total_cluster_size / self.clusters_counted as f64;
            let average_cluster_size = if average_cluster_size < 2.0 {
                2.0
            } else {
                average_cluster_size
            };
            let steps_to_run = (state.len() as f64 / average_cluster_size).ceil();
            let steps_to_run = if steps_to_run < 1.0 {
                1
            } else {
                steps_to_run as usize
            };
            self.total_cluster_size += (0..steps_to_run)
                .map(|_| {
                    let (size, _) = manager.run_classical_loop_update(&edges, &mut state, &mut rng);
                    size
                })
                .sum::<usize>() as f64;
            self.clusters_counted += steps_to_run;
        }

        let mut manager = manager.into();
        manager.flip_each_cluster_rng(0.5, &mut rng, &mut state);

        state.iter_mut().enumerate().for_each(|(var, state)| {
            if !manager.does_var_have_ops(var) {
                *state = rng.gen_bool(0.5);
            }
        });

        self.rng = Some(rng);
        self.op_manager = Some(manager.into());
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

impl<R, M, L> Verify for QMCIsingGraph<R, M, L>
where
    R: Rng,
    M: OpContainerConstructor + ClassicalLoopUpdater + Into<L>,
    L: ClusterUpdater + Into<M>,
{
    fn verify(&self) -> bool {
        self.op_manager
            .as_ref()
            .and_then(|op| self.state.as_ref().map(|state| (op, state)))
            .map(|(op, state)| op.verify(state))
            .unwrap_or(false)
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

impl<'a> PartialEq for HamInfo<'a> {
    fn eq(&self, other: &Self) -> bool {
        self.edges == other.edges
            && self.transverse == other.transverse
            && self.singlesite_energy_offset == other.singlesite_energy_offset
            && self.twosite_energy_offset == other.twosite_energy_offset
    }
}

impl<'a> Eq for HamInfo<'a> {}

// Implement clone where available.
impl<R, M, L> Clone for QMCIsingGraph<R, M, L>
where
    R: Rng + Clone,
    M: OpContainerConstructor + ClassicalLoopUpdater + Into<L> + Clone,
    L: ClusterUpdater + Into<M> + Clone,
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
            run_semiclassical_steps: self.run_semiclassical_steps,
            semiclassical_bonds: self.semiclassical_bonds.clone(),
            total_cluster_size: self.total_cluster_size,
            clusters_counted: self.clusters_counted,
        }
    }
}

// Allow for conversion to generic QMC type. Clears the internal state, converts edges and field
// into interactions.
impl<R, M, L> Into<QMC<R, M, L>> for QMCIsingGraph<R, M, L>
where
    R: Rng,
    M: OpContainerConstructor + ClassicalLoopUpdater + Into<L>,
    L: ClusterUpdater + Into<M>,
{
    fn into(self) -> QMC<R, M, L> {
        self.into_qmc()
    }
}

#[cfg(feature = "autocorrelations")]
impl<R, M, L> QMCBondAutoCorrelations for QMCIsingGraph<R, M, L>
where
    R: Rng,
    M: OpContainerConstructor + ClassicalLoopUpdater + Into<L>,
    L: ClusterUpdater + Into<M>,
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
    pub type DefaultSerializeQMCGraph = SerializeQMCGraph<FastOps, FastOps>;

    /// A QMC graph without rng for easy serialization.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct SerializeQMCGraph<
        M: OpContainerConstructor + ClassicalLoopUpdater + Into<L>,
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
        run_semiclassical_steps: bool,
        semiclassical_bonds: Option<Vec<Vec<usize>>>,
        total_cluster_size: f64,
        clusters_counted: usize,
    }

    impl<
            M: OpContainerConstructor + ClassicalLoopUpdater + Into<L>,
            L: ClusterUpdater + Into<M>,
        > SerializeQMCGraph<M, L>
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
                run_semiclassical_steps: self.run_semiclassical_steps,
                semiclassical_bonds: self.semiclassical_bonds,
                total_cluster_size: self.total_cluster_size,
                clusters_counted: self.clusters_counted,
            }
        }
    }

    impl<
            R: Rng,
            M: OpContainerConstructor + ClassicalLoopUpdater + Into<L>,
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
                run_semiclassical_steps: self.run_semiclassical_steps,
                semiclassical_bonds: self.semiclassical_bonds,
                total_cluster_size: self.total_cluster_size,
                clusters_counted: self.clusters_counted,
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
