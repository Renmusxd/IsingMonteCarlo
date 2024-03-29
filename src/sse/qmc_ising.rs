use crate::classical::graph::{make_random_spin_state, Edge, GraphState};
#[cfg(feature = "autocorrelations")]
pub use crate::sse::autocorrelations::*;
use crate::sse::fast_ops::FastOps;
use crate::sse::ham::Ham;
use crate::sse::qmc_runner::{Qmc, QmcManager};
pub use crate::sse::qmc_traits::*;
use rand::rngs::ThreadRng;
use rand::Rng;
#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};
use std::cmp::max;

/// Default QMC graph implementation.
pub type DefaultQmcIsingGraph<R> = QmcIsingGraph<R, FastOps>;

type VecEdge = Vec<usize>;

/// Trait encompassing all requirements for op managers in QMCIsingGraph.
pub trait IsingManager:
    OpContainerConstructor + HeatBathDiagonalUpdater + RvbUpdater + ClusterUpdater
{
}

/// A container to run QMC simulations.
#[derive(Debug)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct QmcIsingGraph<R: Rng, M: IsingManager> {
    edges: Vec<(VecEdge, f64)>,
    transverse: f64,
    longitudinal: f64,
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
    longitudinal: f64,
    cutoff: usize,
    state: Option<Vec<bool>>,
) -> DefaultQmcIsingGraph<ThreadRng> {
    let rng = rand::thread_rng();
    DefaultQmcIsingGraph::<ThreadRng>::new_with_rng(
        edges,
        transverse,
        longitudinal,
        cutoff,
        rng,
        state,
    )
}

/// Build a new qmc graph with thread rng from a classical graph.
pub fn new_qmc_from_graph<R: Rng>(
    graph: GraphState<R>,
    transverse: f64,
    longitudinal: f64,
    cutoff: usize,
) -> DefaultQmcIsingGraph<R> {
    DefaultQmcIsingGraph::<R>::new_from_graph(graph, transverse, longitudinal, cutoff)
}

impl<R: Rng, M: IsingManager> QmcIsingGraph<R, M> {
    /// Make a new QMC graph with an rng instance and a function to construct the op manager from
    /// the number of variables and number of interactions / bonds.
    pub fn new_with_rng_with_manager_hook<F, Rg: Rng>(
        edges: Vec<(Edge, f64)>,
        transverse: f64,
        longitudinal: f64,
        cutoff: usize,
        mut rng: Rg,
        state: Option<Vec<bool>>,
        f: F,
    ) -> QmcIsingGraph<Rg, M>
    where
        F: Fn(usize, usize) -> M,
    {
        let nvars = edges.iter().map(|((a, b), _)| max(*a, *b)).max().unwrap() + 1;
        let edges = edges
            .into_iter()
            .map(|((a, b), j)| (vec![a, b], j))
            .collect::<Vec<_>>();
        let edge_offset = edges.iter().map(|(_, j)| j.abs()).sum::<f64>();
        let field_offset = nvars as f64 * (transverse + longitudinal.abs());
        let total_energy_offset = edge_offset + field_offset;

        // Allow for extra bonds in case this is used in tempering.
        let nbonds = edges.len() + nvars + nvars;
        let mut ops = f(nvars, nbonds);
        ops.set_cutoff(cutoff);

        let state = match state {
            Some(state) => state,
            None => make_random_spin_state(nvars, &mut rng),
        };
        let state = Some(state);

        QmcIsingGraph::<Rg, M> {
            edges,
            transverse,
            longitudinal,
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
    pub fn new_with_rng<Rg: Rng>(
        edges: Vec<(Edge, f64)>,
        transverse: f64,
        longitudinal: f64,
        cutoff: usize,
        rng: Rg,
        state: Option<Vec<bool>>,
    ) -> QmcIsingGraph<Rg, M> {
        Self::new_with_rng_with_manager_hook(
            edges,
            transverse,
            longitudinal,
            cutoff,
            rng,
            state,
            M::new_with_bonds,
        )
    }

    /// Make a new QMC graph with an rng instance.
    pub fn new_from_graph<Rg: Rng>(
        graph: GraphState<Rg>,
        transverse: f64,
        longitudinal: f64,
        cutoff: usize,
    ) -> QmcIsingGraph<Rg, M> {
        assert!(graph.biases.into_iter().all(|v| v == 0.0));
        Self::new_with_rng(
            graph.edges,
            transverse,
            longitudinal,
            cutoff,
            graph.rng,
            graph.state,
        )
    }

    /// Make the hamiltonian struct.
    pub fn make_haminfo(&self) -> HamInfo {
        HamInfo {
            edges: &self.edges,
            transverse: self.transverse,
            longitudinal: self.longitudinal,
            nvars: self.get_nvars(),
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
        match bond {
            bond if bond < info.edges.len() => {
                debug_assert_eq!(vars.len(), 2);
                two_site_hamiltonian(
                    (input_state[0], input_state[1]),
                    (output_state[0], output_state[1]),
                    info.edges[bond].1,
                )
            }
            bond if bond < info.edges.len() + info.nvars => {
                debug_assert_eq!(vars.len(), 1);
                transverse_hamiltonian(input_state[0], output_state[0], info.transverse)
            }
            bond if bond < info.edges.len() + 2 * info.nvars => {
                debug_assert_eq!(vars.len(), 1);
                longitudinal_hamiltonian(input_state[0], output_state[0], info.longitudinal)
            }
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
        let longitudinal = self.longitudinal;
        let hinfo = HamInfo {
            edges,
            transverse,
            longitudinal,
            nvars,
        };
        let h = |vars: &[usize], bond: usize, input_state: &[bool], output_state: &[bool]| {
            Self::hamiltonian(&hinfo, vars, bond, input_state, output_state)
        };

        let num_bonds = edges.len()
            + nvars
            + if longitudinal.abs() > std::f64::EPSILON {
                nvars
            } else {
                0
            };
        let bonds_fn = |b: usize| -> (&[usize], bool) {
            // 0 to edges.len() are 2-site
            if b < edges.len() {
                (&edges[b].0, false)
            } else if b < edges.len() + nvars {
                let b = b - edges.len();
                (&vars[b..b + 1], true)
            } else {
                let b = b - nvars - edges.len();
                (&vars[b..b + 1], false)
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

    /// Take a single cluster step and return the number of cluster found.
    pub fn single_cluster_step(&mut self) -> usize {
        let mut state = self.state.take().unwrap();
        let nvars = self.get_nvars();
        let mut manager = self.op_manager.take().unwrap();
        let rng = self.rng.as_mut().unwrap();

        let nedges = self.edges.len();

        let n_clusters = if self.longitudinal.abs() > std::f64::EPSILON {
            let long = self.longitudinal;
            manager.flip_each_cluster_rng(
                0.5,
                rng,
                &mut state,
                Some(|node: &M::Node| -> f64 {
                    let bond = node.get_op_ref().get_bond();
                    let is_long_field_bond = bond >= nedges + nvars;
                    if !is_long_field_bond {
                        1.0
                    } else {
                        // We can assume the longitudinal bond is not currently in the 0 weight
                        // state since it wouldn't be in the graph.
                        debug_assert_eq!(node.get_op_ref().get_vars().len(), 1);
                        debug_assert!(node.get_op_ref().is_diagonal());
                        debug_assert!({
                            let op = node.get_op_ref();
                            op.get_inputs()[0] == (long > 0.)
                        });
                        0.0
                    }
                }),
            )
        } else {
            manager.flip_each_cluster_ising_symmetry_rng(0.5, rng, &mut state)
        };

        state.iter_mut().enumerate().for_each(|(var, state)| {
            if !manager.does_var_have_ops(var) {
                *state = rng.gen_bool(0.5);
            }
        });

        self.op_manager = Some(manager);
        self.state = Some(state);
        n_clusters
    }

    /// Perform a single rvb step and return the number of successes and attempts.
    pub fn single_rvb_sweep(&mut self, updates_in_sweep: Option<usize>) -> (usize, usize) {
        let mut state = self.state.take().unwrap();
        if self.classical_bonds.is_none() {
            self.make_classical_bonds(state.len());
        }
        let mut manager = self.op_manager.take().unwrap();
        let rng = self.rng.as_mut().unwrap();

        let nvars = state.len();
        let edges = &self.edges;
        let nedges = edges.len();
        let vars = &self.vars;
        let transverse = self.transverse;
        let longitudinal = self.longitudinal;
        let hinfo = HamInfo {
            edges,
            transverse,
            longitudinal,
            nvars,
        };
        let h = |vars: &[usize], bond: usize, input_state: &[bool], output_state: &[bool]| {
            Self::hamiltonian(&hinfo, vars, bond, input_state, output_state)
        };

        let num_bonds = edges.len()
            + nvars
            + if longitudinal.abs() > std::f64::EPSILON {
                nvars
            } else {
                0
            };
        let bonds_fn = |b: usize| -> (&[usize], bool) {
            if b < edges.len() {
                (&edges[b].0, false)
            } else if b < edges.len() + nvars {
                let b = b - edges.len();
                (&vars[b..b + 1], true)
            } else {
                let b = b - nvars - edges.len();
                (&vars[b..b + 1], false)
            }
        };

        // Start by editing the ops list
        let ham = Ham::new(h, bonds_fn, num_bonds);

        let edges = EdgeNav {
            var_to_bonds: self.classical_bonds.as_ref().unwrap(),
            edges: &self.edges,
        };

        // Average cluster size is always 2.
        let steps_to_run = updates_in_sweep.unwrap_or((state.len() + 1) / 2);
        let successes = if self.longitudinal.abs() > std::f64::EPSILON {
            let long = self.longitudinal;
            manager.rvb_update_with_ising_weight(
                &edges,
                &mut state,
                steps_to_run,
                |bond, sa, sb| {
                    let (va, vb) = edges.vars_for_bond(bond);
                    ham.hamiltonian(&[va, vb], bond, &[sa, sb], &[sa, sb])
                },
                |op| {
                    let bond = op.get_bond();
                    let is_long_field_bond = bond >= nedges + nvars;
                    if !is_long_field_bond {
                        1.0
                    } else {
                        // We can assume the longitudinal bond is not currently in the 0 weight
                        // state since it wouldn't be in the graph.
                        debug_assert_eq!(op.get_vars().len(), 1);
                        debug_assert!(op.is_diagonal());
                        debug_assert!(op.get_inputs()[0] == (long > 0.));
                        0.0
                    }
                },
                rng,
            )
        } else {
            manager.rvb_update(
                &edges,
                &mut state,
                steps_to_run,
                |bond, sa, sb| {
                    let (va, vb) = edges.vars_for_bond(bond);
                    ham.hamiltonian(&[va, vb], bond, &[sa, sb], &[sa, sb])
                },
                rng,
            )
        };

        self.op_manager = Some(manager);
        self.state = Some(state);
        (successes, steps_to_run)
    }

    /// Build classical bonds list.
    fn make_classical_bonds(&mut self, nvars: usize) {
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
    }

    /// Enable or disable automatic rvb steps. Errors if all js not equal magnitude.
    pub fn set_run_rvb(&mut self, run_rvb: bool) {
        self.run_rvb_steps = run_rvb;
        if run_rvb && self.classical_bonds.is_none() {
            let nvars = self.state.as_ref().map(|s| s.len()).unwrap();
            self.make_classical_bonds(nvars)
        }
    }

    /// Enable heatbath diagonal updates.
    pub fn set_enable_heatbath(&mut self, enable_heatbath: bool) {
        if enable_heatbath {
            let nvars = self.get_nvars();
            let edges = &self.edges;
            let vars = &self.vars;
            let transverse = self.transverse;
            let longitudinal = self.longitudinal;
            let hinfo = HamInfo {
                edges,
                transverse,
                longitudinal,
                nvars,
            };
            let h = |vars: &[usize], bond: usize, input_state: &[bool], output_state: &[bool]| {
                Self::hamiltonian(&hinfo, vars, bond, input_state, output_state)
            };

            let num_bonds = edges.len()
                + nvars
                + if longitudinal.abs() > std::f64::EPSILON {
                    nvars
                } else {
                    0
                };
            let bonds_fn = |b: usize| -> &[usize] {
                if b < edges.len() {
                    &edges[b].0
                } else if b < edges.len() + nvars {
                    let b = b - edges.len();
                    &vars[b..b + 1]
                } else {
                    let b = b - nvars - edges.len();
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

    /// Get the longitudinal field on the system.
    pub fn get_longitudinal_field(&self) -> f64 {
        self.longitudinal
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
    pub fn can_swap_managers(&self, other: &Self) -> Result<(), String> {
        self.edges
            .iter()
            .zip(other.edges.iter())
            .try_for_each(|(s, o)| {
                let (sedge, sj) = s;
                let (oedge, oj) = o;

                if sedge != oedge {
                    Err(format!("Edge {:?} not equal to {:?}", sedge, oedge))
                } else if sj.signum() != oj.signum() {
                    Err(format!(
                        "For edge {:?}: bonds must be of same sign {} / {}",
                        sedge, sj, oj
                    ))
                } else {
                    Ok(())
                }
            })?;
        if self.longitudinal.signum() != other.longitudinal.signum() {
            Err(format!(
                "Longitudinal fields are not of the same sign: {} / {}",
                self.longitudinal, other.longitudinal
            ))
        } else {
            Ok(())
        }
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

    fn bond_mag(&self, b: usize) -> f64 {
        self.edges[b].1.abs()
    }
}

impl<R, M> QmcStepper for QmcIsingGraph<R, M>
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
        let longitudinal = self.longitudinal;
        let hinfo = HamInfo {
            edges,
            transverse,
            longitudinal,
            nvars,
        };
        let h = |vars: &[usize], bond: usize, input_state: &[bool], output_state: &[bool]| {
            Self::hamiltonian(&hinfo, vars, bond, input_state, output_state)
        };

        let num_bonds = edges.len()
            + nvars
            + if longitudinal.abs() > std::f64::EPSILON {
                nvars
            } else {
                0
            };
        let bonds_fn = |b: usize| -> (&[usize], bool) {
            if b < edges.len() {
                (&edges[b].0, false)
            } else if b < edges.len() + nvars {
                let b = b - edges.len();
                (&vars[b..b + 1], true)
            } else {
                let b = b - nvars - edges.len();
                (&vars[b..b + 1], false)
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

        let nedges = edges.len();
        if self.run_rvb_steps {
            let edges = EdgeNav {
                var_to_bonds: self.classical_bonds.as_ref().unwrap(),
                edges: &self.edges,
            };
            // Average cluster size is always 2.
            let steps_to_run = (state.len() + 1) / 2;

            let succs = if self.longitudinal.abs() > std::f64::EPSILON {
                manager.rvb_update_with_ising_weight(
                    &edges,
                    &mut state,
                    steps_to_run,
                    |bond, sa, sb| {
                        let (va, vb) = edges.vars_for_bond(bond);
                        ham.hamiltonian(&[va, vb], bond, &[sa, sb], &[sa, sb])
                    },
                    |op| {
                        let bond = op.get_bond();
                        let is_long_field_bond = bond >= nedges + nvars;
                        if !is_long_field_bond {
                            1.0
                        } else {
                            // We can assume the longitudinal bond is not currently in the 0 weight
                            // state since it wouldn't be in the graph.
                            debug_assert_eq!(op.get_vars().len(), 1);
                            debug_assert!(op.is_diagonal());
                            debug_assert!(op.get_inputs()[0] == (self.longitudinal > 0.));
                            0.0
                        }
                    },
                    &mut rng,
                )
            } else {
                manager.rvb_update(
                    &edges,
                    &mut state,
                    steps_to_run,
                    |bond, sa, sb| {
                        let (va, vb) = edges.vars_for_bond(bond);
                        ham.hamiltonian(&[va, vb], bond, &[sa, sb], &[sa, sb])
                    },
                    &mut rng,
                )
            };
            self.total_rvb_successes += succs;
            self.rvb_clusters_counted += steps_to_run;
        }

        if self.longitudinal.abs() > std::f64::EPSILON {
            manager.flip_each_cluster_rng(
                0.5,
                &mut rng,
                &mut state,
                Some(|node: &M::Node| -> f64 {
                    let bond = node.get_op_ref().get_bond();
                    let is_long_field_bond = bond >= nedges + nvars;
                    if !is_long_field_bond {
                        1.0
                    } else {
                        // We can assume the longitudinal bond is not currently in the 0 weight
                        // state since it wouldn't be in the graph.
                        debug_assert_eq!(node.get_op_ref().get_vars().len(), 1);
                        debug_assert!(node.get_op_ref().is_diagonal());
                        debug_assert!({
                            let op = node.get_op_ref();
                            op.get_inputs()[0] == (self.longitudinal > 0.)
                        });
                        0.0
                    }
                }),
            );
        } else {
            manager.flip_each_cluster_ising_symmetry_rng(0.5, &mut rng, &mut state);
        }
        state.iter_mut().enumerate().for_each(|(var, state)| {
            if !manager.does_var_have_ops(var) {
                *state = rng.gen_bool(0.5);
            }
        });

        self.cutoff = max(self.cutoff, manager.get_n() + manager.get_n() / 2);

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

    fn get_bond_count(&self, bond: usize) -> usize {
        self.get_manager_ref().get_count(bond)
    }

    fn imaginary_time_fold<F, T>(&self, fold_fn: F, init: T) -> T
    where
        F: Fn(T, &[bool]) -> T,
    {
        let mut state = self.clone_state();
        self.get_manager_ref().itime_fold(&mut state, fold_fn, init)
    }
}

impl<R, M> Verify for QmcIsingGraph<R, M>
where
    R: Rng,
    M: IsingManager,
{
    fn verify(&self) -> bool {
        let ham_info = self.make_haminfo();
        if let Some(m) = self.op_manager.as_ref() {
            let all_pos = m
                .try_iterate_ops(0, self.get_cutoff(), (), |_, op, _, _| {
                    let w = Self::hamiltonian(
                        &ham_info,
                        op.get_vars(),
                        op.get_bond(),
                        op.get_inputs(),
                        op.get_outputs(),
                    );
                    if w.abs() > std::f64::EPSILON {
                        Ok(())
                    } else {
                        Err(())
                    }
                })
                .is_ok();

            let m_verify = self
                .op_manager
                .as_ref()
                .zip(self.state.as_ref())
                .map(|(m, state)| m.verify(state))
                .unwrap_or(false);

            all_pos && m_verify
        } else {
            false
        }
    }
}

fn two_site_hamiltonian(inputs: (bool, bool), outputs: (bool, bool), bond: f64) -> f64 {
    if inputs == outputs {
        bond.abs()
            + match inputs {
                (false, false) => -bond,
                (false, true) => bond,
                (true, false) => bond,
                (true, true) => -bond,
            }
    } else {
        0.0
    }
}

fn transverse_hamiltonian(_input_state: bool, _output_state: bool, transverse: f64) -> f64 {
    transverse
}

fn longitudinal_hamiltonian(input_state: bool, output_state: bool, longitudinal: f64) -> f64 {
    longitudinal.abs()
        + match (input_state, output_state) {
            (true, false) | (false, true) => 0.,
            (true, true) => longitudinal,
            (false, false) => -longitudinal,
        }
}

/// Data required to evaluate the hamiltonian.
#[derive(Debug)]
pub struct HamInfo<'a> {
    edges: &'a [(VecEdge, f64)],
    transverse: f64,
    longitudinal: f64,
    nvars: usize,
}

impl<'a> PartialEq for HamInfo<'a> {
    fn eq(&self, other: &Self) -> bool {
        self.edges == other.edges && self.transverse == other.transverse
    }
}

impl<'a> Eq for HamInfo<'a> {}

// Implement clone where available.
impl<R, M> Clone for QmcIsingGraph<R, M>
where
    R: Rng + Clone,
    M: IsingManager + Clone,
{
    fn clone(&self) -> Self {
        Self {
            edges: self.edges.clone(),
            transverse: self.transverse,
            longitudinal: self.longitudinal,
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
pub trait IntoQmc<R, M>
where
    R: Rng,
    M: QmcManager,
{
    /// Convert into QMC.
    fn into_qmc(self) -> Qmc<R, M>;
}

impl<R, M> IntoQmc<R, M> for QmcIsingGraph<R, M>
where
    R: Rng,
    M: IsingManager + QmcManager,
{
    fn into_qmc(self) -> Qmc<R, M> {
        let nvars = self.get_nvars();
        let rng = self.rng.unwrap();
        let state = self.state.as_ref().unwrap().to_vec();
        let mut qmc = Qmc::<R, M>::new_with_state(nvars, rng, state, false);
        let transverse = self.transverse;
        let longitudinal = self.longitudinal;
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
        if longitudinal.abs() > std::f64::EPSILON {
            (0..nvars).for_each(|var| {
                qmc.make_interaction(vec![longitudinal, 0., 0., -longitudinal], vec![var])
                    .unwrap()
            });
        }
        qmc.increase_cutoff_to(self.cutoff);
        qmc.set_manager(self.op_manager.unwrap());
        qmc
    }
}

#[cfg(feature = "autocorrelations")]
impl<R, M> QmcBondAutoCorrelations for QmcIsingGraph<R, M>
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
    pub type DefaultSerializeQmcGraph = SerializeQmcGraph<FastOps>;

    /// A QMC graph without rng for easy serialization.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct SerializeQmcGraph<M: IsingManager> {
        edges: Vec<(VecEdge, f64)>,
        transverse: f64,
        longitudinal: f64,
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

    impl<M> SerializeQmcGraph<M>
    where
        M: IsingManager,
    {
        /// Convert into a proper QMC graph using a new rng instance.
        pub fn into_qmc<R: Rng>(self, rng: R) -> QmcIsingGraph<R, M> {
            QmcIsingGraph {
                edges: self.edges,
                transverse: self.transverse,
                longitudinal: self.longitudinal,
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

    impl<R, M> From<QmcIsingGraph<R, M>> for (SerializeQmcGraph<M>, R)
    where
        R: Rng,
        M: IsingManager,
    {
        fn from(g: QmcIsingGraph<R, M>) -> (SerializeQmcGraph<M>, R) {
            let sg = SerializeQmcGraph {
                edges: g.edges,
                transverse: g.transverse,
                longitudinal: g.longitudinal,
                state: g.state,
                cutoff: g.cutoff,
                op_manager: g.op_manager,
                total_energy_offset: g.total_energy_offset,
                nvars: g.vars.len(),
                run_rvb_steps: g.run_rvb_steps,
                classical_bonds: g.classical_bonds,
                total_rvb_successes: g.total_rvb_successes,
                rvb_clusters_counted: g.rvb_clusters_counted,
                bond_weights: g.bond_weights,
            };
            (sg, g.rng.unwrap())
        }
    }

    impl<R, M> From<QmcIsingGraph<R, M>> for SerializeQmcGraph<M>
    where
        R: Rng,
        M: IsingManager,
    {
        fn from(g: QmcIsingGraph<R, M>) -> SerializeQmcGraph<M> {
            let (sg, _) = g.into();
            sg
        }
    }

    #[cfg(test)]
    mod serialize_test {
        use super::*;
        use rand::prelude::SmallRng;
        use rand::{Error, RngCore, SeedableRng};

        #[derive(Serialize, Deserialize)]
        struct FakeSerializbleRng {}
        impl FakeSerializbleRng {
            fn new() -> Self {
                Self {}
            }
        }
        impl RngCore for FakeSerializbleRng {
            fn next_u32(&mut self) -> u32 {
                0
            }

            fn next_u64(&mut self) -> u64 {
                0
            }

            fn fill_bytes(&mut self, dest: &mut [u8]) {
                dest.iter_mut().for_each(|b| *b = 0)
            }

            fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), Error> {
                dest.iter_mut().for_each(|b| *b = 0);
                Ok(())
            }
        }

        #[test]
        fn test_serialize() {
            let rng = FakeSerializbleRng::new();
            let mut g = DefaultQmcIsingGraph::<FakeSerializbleRng>::new_with_rng(
                vec![((0, 1), 1.0)],
                1.0,
                0.,
                1,
                rng,
                None,
            );
            g.timesteps(1, 1.0);
            let mut v: Vec<u8> = Vec::default();
            serde_json::to_writer_pretty(&mut v, &g).unwrap();
            let _: DefaultQmcIsingGraph<FakeSerializbleRng> = serde_json::from_slice(&v).unwrap();
        }

        #[test]
        fn test_serialize_no_rng() {
            let rng = FakeSerializbleRng::new();
            let mut g = DefaultQmcIsingGraph::<SmallRng>::new_with_rng(
                vec![((0, 1), 1.0)],
                1.0,
                0.,
                1,
                rng,
                None,
            );
            g.timesteps(1, 1.0);
            let mut v: Vec<u8> = Vec::default();
            let sg: DefaultSerializeQmcGraph = g.into();
            serde_json::to_writer_pretty(&mut v, &sg).unwrap();
            println!("{}", String::from_utf8(v.clone()).unwrap());

            let rng = SmallRng::seed_from_u64(1234);
            let sg: DefaultSerializeQmcGraph = serde_json::from_slice(&v).unwrap();
            let _g: DefaultQmcIsingGraph<SmallRng> = sg.into_qmc(rng);
        }
    }
}
