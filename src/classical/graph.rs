use crate::util::vec_help::remove_doubles;
use rand::prelude::*;
use smallvec::{smallvec, SmallVec};
use std::cmp::max;
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

impl<R: Rng + Clone> Clone for GraphState<R> {
    fn clone(&self) -> Self {
        Self {
            edges: self.edges.clone(),
            binding_mat: self.binding_mat.clone(),
            biases: self.biases.clone(),
            state: self.state.clone(),
            cumulative_weight: self.cumulative_weight.clone(),
            rng: self.rng.clone(),
        }
    }
}

#[derive(Copy, Clone, Debug)]
enum WormMove {
    Single(usize),
    Double(usize, usize),
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
        let mut binding_mat: Vec<Vec<(usize, f64)>> = vec![vec![]; biases.len()];

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
            let delta_e = Self::delta_e(va, Some(vb), state, binding_mat);
            delta_e + (2.0 * biases[va] * if state[va] { 1.0 } else { -1.0 })
        };
        let delta_e = delta_e(va, vb) + delta_e(vb, va);
        if Self::should_flip(rng, beta, delta_e) {
            state[va] = !state[va];
            state[vb] = !state[vb];
        }
    }

    fn delta_e(
        v: usize,
        omit: Option<usize>,
        state: &[bool],
        binding_mat: &[Vec<(usize, f64)>],
    ) -> f64 {
        let curr_value = state[v];
        binding_mat[v]
            .iter()
            .cloned()
            .filter(|(ov, _)| Some(*ov) != omit)
            .map(|(indx, j)| {
                let old_coupling = if !(curr_value ^ state[indx]) {
                    1.0
                } else {
                    -1.0
                };
                // j*new - j*old = j*(-old) - j*(old) = -2j*(old)
                -2.0 * j * old_coupling
            })
            .sum()
    }

    /// Randomly flip a series of spins attached by an edges.
    fn do_worm_flip(
        rng: &mut R,
        beta: f64,
        binding_mat: &[Vec<(usize, f64)>],
        biases: &[f64],
        state: &mut [bool],
        allow_doubles: bool,
    ) {
        let start_index = rng.gen_range(0..state.len());
        let mut visit_path = vec![WormMove::Single(start_index)];
        let mut last_index = start_index;

        let delta_e = |wm: WormMove, state: &mut [bool]| -> f64 {
            match wm {
                WormMove::Single(va) => Self::delta_e(va, None, state, binding_mat),
                WormMove::Double(va, vb) => {
                    let de = Self::delta_e(va, Some(vb), state, binding_mat);
                    de + Self::delta_e(vb, Some(va), state, binding_mat)
                }
            }
        };
        let starting_e = delta_e(WormMove::Single(start_index), state);
        state[start_index] = !state[start_index];

        let mut update_failed = false;

        let mut smallstack = vec![];
        loop {
            smallstack.clear();
            let sel_move = visit_path[visit_path.len() - 1];
            let sel_var = match sel_move {
                WormMove::Single(v) => v,
                WormMove::Double(_, v) => v,
            };
            let mut any_resolve = false;
            binding_mat[sel_var].iter().cloned().for_each(|(ov, _)| {
                if ov != last_index {
                    let de = delta_e(WormMove::Single(ov), state);
                    if de.abs() < f64::EPSILON {
                        smallstack.push((WormMove::Single(ov), de));
                    } else if (de + starting_e).abs() < f64::EPSILON {
                        smallstack.push((WormMove::Single(ov), de));
                        any_resolve = true;
                    }
                    // Now check jumps from here for doubles, pretend the last was flipped.
                    // Same as delta_e(Double...)
                    if allow_doubles {
                        state[ov] = !state[ov];
                        binding_mat[ov].iter().cloned().for_each(|(oov, _)| {
                            if oov != ov && oov != sel_var {
                                let de = delta_e(WormMove::Single(oov), state) + de;
                                if de.abs() < f64::EPSILON {
                                    smallstack.push((WormMove::Double(ov, oov), de));
                                } else if (de + starting_e).abs() < f64::EPSILON {
                                    smallstack.push((WormMove::Double(ov, oov), de));
                                    any_resolve = true;
                                }
                            }
                        });
                        // Return to normal.
                        state[ov] = !state[ov];
                    }
                }
            });
            if any_resolve {
                smallstack.retain(|(_, de)| (de + starting_e).abs() < f64::EPSILON);
            }
            // Smallstack now has symmetric operations as well as ending conditions.
            let (ov, de) = if !smallstack.is_empty() {
                let choice = rng.gen_range(0..smallstack.len());
                let (ov, de) = smallstack[choice];
                visit_path.push(ov);
                (ov, de)
            } else {
                // If there are no options, turn around, undo last move.
                let sel_move = match sel_move {
                    WormMove::Single(_) => sel_move,
                    WormMove::Double(va, vb) => WormMove::Double(vb, va),
                };
                visit_path.push(sel_move);

                let de = delta_e(sel_move, state);
                (sel_move, de)
            };
            match ov {
                WormMove::Single(va) => {
                    state[va] = !state[va];
                }
                WormMove::Double(va, vb) => {
                    state[va] = !state[va];
                    state[vb] = !state[vb];
                }
            }
            last_index = match (ov, sel_move) {
                (WormMove::Single(_), WormMove::Single(v)) => v,
                (WormMove::Single(_), WormMove::Double(_, v)) => v,
                (WormMove::Double(v, _), _) => v,
            };
            if (de + starting_e).abs() < f64::EPSILON {
                // We have gone back to the initial energy.
                break;
            }
            // Path getting too long.
            if visit_path.len() > state.len() {
                update_failed = true;
                break;
            }
        }
        let mut visit_path = visit_path
            .into_iter()
            .map(|wm| -> SmallVec<[usize; 2]> {
                match wm {
                    WormMove::Single(v) => smallvec![v],
                    WormMove::Double(va, vb) => smallvec![va, vb],
                }
            })
            .flatten()
            .collect::<Vec<_>>();
        visit_path.sort_unstable();
        remove_doubles(&mut visit_path);

        if !update_failed {
            // Now calculate the energy change from longitudinal fields.
            let total_he = visit_path
                .iter()
                .cloned()
                .map(|v| 2.0 * biases[v] * if state[v] { 1.0 } else { -1.0 })
                .sum();
            if !Self::should_flip(rng, beta, total_he) {
                visit_path.iter().cloned().for_each(|v| {
                    state[v] = !state[v];
                })
            }
        } else {
            // Fix paths and apologize.
            visit_path.iter().cloned().for_each(|v| {
                state[v] = !state[v];
            })
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
    pub fn do_time_step(
        &mut self,
        beta: f64,
        nspinupdates: Option<usize>,
        nedgeupdates: Option<usize>,
        nwormupdates: Option<usize>,
        only_basic_moves: Option<bool>,
    ) -> Result<(), String> {
        // Energy cost of this flip
        if let Some(mut spin_state) = self.state.take() {
            let only_basic_moves = only_basic_moves.unwrap_or(false);
            let nspinupdates = nspinupdates.unwrap_or_else(|| max(1, spin_state.len() / 2));
            let nedgeupdates = nedgeupdates.unwrap_or_else(|| max(1, self.edges.len() / 2));
            let nwormupdates = nwormupdates.unwrap_or(1);
            let t = if only_basic_moves { 2 } else { 3 };
            let choice: u8 = self.rng.gen_range(0..t);
            match choice {
                0 => (0..nspinupdates).for_each(|_| {
                    Self::do_spin_flip(
                        &mut self.rng,
                        beta,
                        &self.binding_mat,
                        &self.biases,
                        &mut spin_state,
                    )
                }),
                1 => (0..nedgeupdates).for_each(|_| {
                    Self::do_edge_flip(
                        &mut self.rng,
                        beta,
                        &self.edges,
                        &self.binding_mat,
                        &self.biases,
                        &mut spin_state,
                        self.cumulative_weight
                            .as_ref()
                            .map(|(v, w)| (v.as_slice(), *w)),
                    )
                }),
                2 => (0..nwormupdates).for_each(|_| {
                    Self::do_worm_flip(
                        &mut self.rng,
                        beta,
                        &self.binding_mat,
                        &self.biases,
                        &mut spin_state,
                        true,
                    )
                }),
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

#[cfg(test)]
mod classic_tests {
    use super::*;
    use itertools::Itertools;
    use std::cmp::{max, min};

    fn two_d_periodic(l: usize) -> Vec<(Edge, f64)> {
        let indices: Vec<(usize, usize)> = (0usize..l)
            .map(|i| (0usize..l).map(|j| (i, j)).collect::<Vec<(usize, usize)>>())
            .flatten()
            .collect();
        let f = |i, j| j * l + i;

        let right_connects = indices
            .iter()
            .cloned()
            .map(|(i, j)| ((f(i, j), f((i + 1) % l, j)), -1.0));
        let down_connects = indices.iter().cloned().map(|(i, j)| {
            (
                (f(i, j), f(i, (j + 1) % l)),
                if i % 2 == 0 { 1.0 } else { -1.0 },
            )
        });
        right_connects.chain(down_connects).collect()
    }

    #[test]
    fn test_worm_flip() {
        let mut g = GraphState::new_with_state_and_rng(
            vec![false, false, false],
            &[((0, 1), 1.0), ((1, 2), 1.0), ((2, 0), 1.0)],
            &[0., 0., 0.],
            SmallRng::from_entropy(),
        );
        GraphState::do_worm_flip(
            &mut g.rng,
            1.0,
            &g.binding_mat,
            &g.biases,
            g.state.as_mut().unwrap(),
            false,
        );
        assert!(g.state.unwrap().into_iter().all(|b| b))
    }

    #[test]
    fn test_worm_flip_bias() {
        let mut g = GraphState::new_with_state_and_rng(
            vec![false, false, false],
            &[((0, 1), 1.0), ((1, 2), 1.0), ((2, 0), 1.0)],
            &[-1., -1., -1.],
            SmallRng::from_entropy(),
        );
        GraphState::do_worm_flip(
            &mut g.rng,
            1.0,
            &g.binding_mat,
            &g.biases,
            g.state.as_mut().unwrap(),
            false,
        );
        assert!(g.state.unwrap().into_iter().all(|b| b))
    }

    #[test]
    fn test_worm_flip_bias_not() {
        let mut g = GraphState::new_with_state_and_rng(
            vec![false, false, false],
            &[((0, 1), 1.0), ((1, 2), 1.0), ((2, 0), 1.0)],
            &[1., 1., 1.],
            SmallRng::from_entropy(),
        );
        GraphState::do_worm_flip(
            &mut g.rng,
            1000.,
            &g.binding_mat,
            &g.biases,
            g.state.as_mut().unwrap(),
            false,
        );
        assert!(g.state.unwrap().into_iter().all(|b| !b))
    }

    #[test]
    fn test_worm_flip_bounce() {
        let nvars = 20;
        let edges = (0..nvars - 1)
            .map(|x| ((x, x + 1), 1.0))
            .collect::<Vec<_>>();
        let mut biases = vec![0.0; nvars];
        biases[0] = 10.;
        biases[nvars - 1] = 10.;
        let mut g = GraphState::new_with_state_and_rng(
            vec![false; nvars],
            &edges,
            &biases,
            SmallRng::from_entropy(),
        );
        GraphState::do_worm_flip(
            &mut g.rng,
            1000.0,
            &g.binding_mat,
            &g.biases,
            g.state.as_mut().unwrap(),
            false,
        );
        assert!(g.state.unwrap().into_iter().all(|b| !b))
    }

    #[test]
    fn test_worm_flip_doubles() {
        let mut g = GraphState::new_with_state_and_rng(
            vec![false, false, false],
            &[((0, 1), 1.0), ((1, 2), 1.0), ((2, 0), 1.0)],
            &[0., 0., 0.],
            SmallRng::from_entropy(),
        );
        GraphState::do_worm_flip(
            &mut g.rng,
            1.0,
            &g.binding_mat,
            &g.biases,
            g.state.as_mut().unwrap(),
            true,
        );
        assert!(g.state.unwrap().into_iter().all_equal())
    }

    #[test]
    fn test_worm_2d() {
        let l = 4;
        let nvars = l * l;
        let edges = two_d_periodic(l);
        let biases = vec![0.0; nvars];
        let mut g = GraphState::new_with_state_and_rng(
            vec![false; nvars],
            &edges,
            &biases,
            SmallRng::from_entropy(),
        );
        GraphState::do_worm_flip(
            &mut g.rng,
            1000.0,
            &g.binding_mat,
            &g.biases,
            g.state.as_mut().unwrap(),
            true,
        );
    }

    fn bathroom_unit_cells(l: usize) -> Vec<(Edge, f64)> {
        let mut edges = vec![];
        for x in 0..l {
            for y in 0..l {
                for i in 0..4 {
                    let va = y * l * 4 + x * 4 + i;
                    let vb = y * l * 4 + x * 4 + (i + 1) % 4;
                    edges.push(((min(va, vb), max(va, vb)), if i == 0 { 1.0 } else { -1.0 }))
                }
                let va = y * l * 4 + x * 4 + 1;
                let vb = y * l * 4 + ((x + 1) % l) * 4 + 3;
                edges.push(((min(va, vb), max(va, vb)), -1.0));

                let va = y * l * 4 + x * 4 + 0;
                let vb = ((y + 1) % l) * l * 4 + x * 4 + 2;
                edges.push(((min(va, vb), max(va, vb)), -1.0));
            }
        }

        edges
    }

    #[test]
    fn test_worm_2d_bathroom() {
        let l = 16;
        let nvars = l * l * 4;
        let edges = bathroom_unit_cells(l);
        let biases = vec![0.0; nvars];
        let mut g = GraphState::new_with_state_and_rng(
            vec![false; nvars],
            &edges,
            &biases,
            SmallRng::from_entropy(),
        );
        GraphState::do_worm_flip(
            &mut g.rng,
            1000.0,
            &g.binding_mat,
            &g.biases,
            g.state.as_mut().unwrap(),
            true,
        );
    }
}
