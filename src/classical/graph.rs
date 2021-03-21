use rand::prelude::*;
use std::fmt::{Debug, Error, Formatter};

/// A graph definition for use in classical monte carlo.
pub struct GraphState<R: Rng> {
    pub(crate) edges: Vec<(Edge, f64)>,
    pub(crate) binding_mat: Vec<Vec<(usize, f64)>>,
    pub(crate) biases: Vec<f64>,
    pub(crate) state: Option<Vec<bool>>,
    cumulative_weight: Option<(Vec<f64>, f64)>,
    pub(crate) rng: R,
}

impl<R: Rng> Debug for GraphState<R> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        if let Some(state) = &self.state {
            let s = state
                .iter()
                .map(|b| if *b { "1" } else { "0" })
                .collect::<Vec<_>>()
                .join("");
            let e = self.get_energy();
            f.write_str(&format!("{}\t{}", s, e))
        } else {
            f.write_str("Error")
        }
    }
}

/// An edge between two variables.
pub type Edge = (usize, usize);
impl<R: Rng> GraphState<R> {
    /// Make a new Graph from a list of edges `[((vara, varb), j), ...]` and longitudinal fields.
    pub fn new(edges: &[(Edge, f64)], biases: &[f64], mut rng: R) -> Self {
        let state = make_random_spin_state(biases.len(), &mut rng);
        Self::new_with_state_and_rng(state, edges, biases, rng)
    }

    /// Make a new graph with an initial state, edges, and longitudinal fields.
    pub fn new_with_state_and_rng(
        state: Vec<bool>,
        edges: &[(Edge, f64)],
        biases: &[f64],
        rng: R,
    ) -> Self {
        // Matrix of all bonds.
        let mut binding_mat: Vec<Vec<(usize, f64)>> = vec![vec![]; biases.len() * biases.len()];

        edges.iter().for_each(|((va, vb), j)| {
            binding_mat[*va].push((*vb, *j));
            binding_mat[*vb].push((*va, *j));
        });
        // Sort just in case
        binding_mat.iter_mut().for_each(|vs| {
            vs.sort_by_key(|(i, _)| *i);
        });

        GraphState {
            edges: edges.to_vec(),
            binding_mat,
            biases: biases.to_vec(),
            state: Some(state),
            cumulative_weight: None,
            rng,
        }
    }

    /// Perform a random single spin flip.
    pub fn do_spin_flip(
        rng: &mut R,
        beta: f64,
        binding_mat: &[Vec<(usize, f64)>],
        biases: &[f64],
        state: &mut [bool],
    ) {
        let random_index = rng.gen_range(0..state.len());
        let curr_value = state[random_index];
        // new - old
        let binding_slice = &binding_mat[random_index];
        let delta_e: f64 = binding_slice
            .iter()
            .cloned()
            .map(|(indx, j)| {
                let old_coupling = if !(curr_value ^ state[indx]) {
                    1.0
                } else {
                    -1.0
                };
                // j*new - j*old = j*(-old) - j*(old) = -2j*(old)
                -2.0 * j * old_coupling
            })
            .sum();
        let delta_e = delta_e + (2.0 * biases[random_index] * if curr_value { 1.0 } else { -1.0 });
        if Self::should_flip(rng, beta, delta_e) {
            state[random_index] = !state[random_index]
        }
    }

    /// Randomly flip two spins attached by an edge.
    fn do_edge_flip(
        rng: &mut R,
        beta: f64,
        edges: &[(Edge, f64)],
        binding_mat: &[Vec<(usize, f64)>],
        biases: &[f64],
        state: &mut [bool],
        cumulative_edge_weights: Option<(&[f64], f64)>,
    ) {
        let indx_edge = if let Some((cumulative_edge_weights, totalw)) = cumulative_edge_weights {
            let p = rng.gen_range(0. ..totalw);
            let indx = cumulative_edge_weights
                .binary_search_by(|v| v.partial_cmp(&p).expect("Couldn't compare values"));
            match indx {
                Ok(indx) => indx,
                Err(indx) => indx,
            }
        } else {
            rng.gen_range(0..edges.len())
        };
        let ((va, vb), _) = edges[indx_edge];

        let delta_e = |va: usize, vb: usize| -> f64 {
            let curr_value = state[va];
            let binding_slice = &binding_mat[va];
            let delta_e: f64 = binding_slice
                .iter()
                .cloned()
                .map(|(indx, j)| {
                    // Skip vb since it will also flip.
                    if indx == vb {
                        0.0
                    } else {
                        let old_coupling = if !(curr_value ^ state[indx]) {
                            1.0
                        } else {
                            -1.0
                        };
                        // j*new - j*old = j*(-old) - j*(old) = -2j*(old)
                        -2.0 * j * old_coupling
                    }
                })
                .sum();
            delta_e + (2.0 * biases[va] * if curr_value { 1.0 } else { -1.0 })
        };
        let delta_e = delta_e(va, vb) + delta_e(vb, va);
        if Self::should_flip(rng, beta, delta_e) {
            state[va] = !state[va];
            state[vb] = !state[vb];
        }
    }

    /// Use the weights of edges to decide how frequently to flip them.
    pub fn enable_edge_importance_sampling(&mut self, enable: bool) {
        self.cumulative_weight = if enable {
            let v = Vec::with_capacity(self.edges.len());
            let (v, totalw) =
                self.edges
                    .iter()
                    .map(|(_, w)| *w)
                    .fold((v, 0.), |(mut accv, accw), w| {
                        accv.push(accw + w);
                        (accv, accw + w)
                    });
            Some((v, totalw))
        } else {
            None
        }
    }

    /// Randomly choose if a step should be made based on temperature and energy change.
    pub fn should_flip(rng: &mut R, beta: f64, delta_e: f64) -> bool {
        // If dE <= 0 then it will always flip, don't bother calculating odds.
        if delta_e > 0.0 {
            let chance = (-beta * delta_e).exp();
            rng.gen::<f64>() < chance
        } else {
            true
        }
    }

    /// Perform a monte carlo time step.
    pub fn do_time_step(&mut self, beta: f64, only_basic_moves: bool) -> Result<(), String> {
        // Energy cost of this flip
        if let Some(mut spin_state) = self.state.take() {
            let choice = if only_basic_moves {
                0
            } else {
                self.rng.gen_range(0..2)
            };
            match choice {
                0 => Self::do_spin_flip(
                    &mut self.rng,
                    beta,
                    &self.binding_mat,
                    &self.biases,
                    &mut spin_state,
                ),
                1 => Self::do_edge_flip(
                    &mut self.rng,
                    beta,
                    &self.edges,
                    &self.binding_mat,
                    &self.biases,
                    &mut spin_state,
                    self.cumulative_weight
                        .as_ref()
                        .map(|(v, w)| (v.as_slice(), *w)),
                ),
                _ => unreachable!(),
            }
            self.state = Some(spin_state);
            Ok(())
        } else {
            Err("No state to edit".to_string())
        }
    }

    /// Get the spin state.
    pub fn get_state(self) -> Vec<bool> {
        self.state.unwrap()
    }

    /// Clone the spin state.
    pub fn clone_state(&self) -> Vec<bool> {
        self.state.clone().unwrap()
    }

    /// Get a ref of the spin state.
    pub fn state_ref(&self) -> &[bool] {
        self.state.as_ref().unwrap()
    }

    /// Overwrite the spin state.
    pub fn set_state(&mut self, state: Vec<bool>) {
        assert_eq!(self.state.as_ref().unwrap().len(), state.len());
        self.state = Some(state)
    }

    /// Get the energy of the system.
    pub fn get_energy(&self) -> f64 {
        if let Some(state) = &self.state {
            state.iter().enumerate().fold(0.0, |acc, (i, si)| {
                let binding_slice = &self.binding_mat[i];
                let total_e: f64 = binding_slice
                    .iter()
                    .map(|(indx, j)| -> f64 {
                        let old_coupling = if !(si ^ state[*indx]) { 1.0 } else { -1.0 };
                        j * old_coupling / 2.0
                    })
                    .sum();
                let bias_e = if *si { -self.biases[i] } else { self.biases[i] };
                acc + total_e + bias_e
            })
        } else {
            std::f64::NAN
        }
    }
}

/// Randomly build a spin state.
pub fn make_random_spin_state<R: Rng>(n: usize, rng: &mut R) -> Vec<bool> {
    (0..n).map(|_| -> bool { rng.gen() }).collect()
}
