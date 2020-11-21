use crate::sse::qmc_traits::{DiagonalUpdater, Hamiltonian};
use crate::sse::Op;
use rand::Rng;
#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};

/// Bond weight storage for fast lookup.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct BondWeights {
    max_weight_and_cumulative: Vec<(usize, f64, f64)>,
}

impl BondWeights {
    /// Make a new BondWeights using an iterator of each individual bond's weight.
    pub fn new<It>(max_bond_weights: It) -> Self
    where
        It: IntoIterator<Item = f64>,
    {
        let max_weight_and_cumulative =
            max_bond_weights
                .into_iter()
                .enumerate()
                .fold(vec![], |mut acc, (b, w)| {
                    if acc.is_empty() {
                        acc.push((b, w, w));
                    } else {
                        acc.push((b, w, w + acc[acc.len() - 1].2));
                    };
                    acc
                });
        Self {
            max_weight_and_cumulative,
        }
    }

    fn get_random_bond_and_max_weight<R: Rng>(&self, mut rng: R) -> Result<(usize, f64), &str> {
        if let Some(total) = self.total() {
            let c = rng.gen_range(0., total);
            let index = self.index_for_cumulative(c);
            Ok((
                self.max_weight_and_cumulative[index].0,
                self.max_weight_and_cumulative[index].1,
            ))
        } else {
            Err("No bonds provided")
        }
    }

    fn total(&self) -> Option<f64> {
        self.max_weight_and_cumulative
            .last()
            .map(|(_, _, tot)| *tot)
    }

    fn index_for_cumulative(&self, val: f64) -> usize {
        self.max_weight_and_cumulative
            .binary_search_by(|(_, _, c)| c.partial_cmp(&val).unwrap())
            .unwrap_or_else(|x| x)
    }

    fn max_weight_for_bond(&self, b: usize) -> f64 {
        self.max_weight_and_cumulative
            .iter()
            .find_map(
                |(bond, maxweight, _)| {
                    if *bond == b {
                        Some(*maxweight)
                    } else {
                        None
                    }
                },
            )
            .unwrap()
    }
}

/// Heatbath updates for a diagonal updater.
pub trait HeatBathDiagonalUpdater: DiagonalUpdater {
    /// Perform a single heatbath update.
    fn make_heatbath_diagonal_update<'b, H: Hamiltonian<'b>>(
        &mut self,
        cutoff: usize,
        beta: f64,
        state: &[bool],
        hamiltonian: &H,
        bond_weights: &BondWeights,
    ) {
        self.make_heatbath_diagonal_update_with_rng(
            cutoff,
            beta,
            state,
            hamiltonian,
            bond_weights,
            &mut rand::thread_rng(),
        )
    }

    /// Perform a single heatbath update.
    fn make_heatbath_diagonal_update_with_rng<'b, H: Hamiltonian<'b>, R: Rng>(
        &mut self,
        cutoff: usize,
        beta: f64,
        state: &[bool],
        hamiltonian: &H,
        bond_weights: &BondWeights,
        rng: &mut R,
    ) {
        let mut state = state.to_vec();
        self.make_heatbath_diagonal_update_with_rng_and_state_ref(
            cutoff,
            beta,
            &mut state,
            hamiltonian,
            bond_weights,
            rng,
        )
    }

    /// Perform a single heatbath update.
    fn make_heatbath_diagonal_update_with_rng_and_state_ref<'b, H: Hamiltonian<'b>, R: Rng>(
        &mut self,
        cutoff: usize,
        beta: f64,
        state: &mut [bool],
        hamiltonian: &H,
        bond_weights: &BondWeights,
        rng: &mut R,
    ) {
        self.mutate_ps(0, cutoff, (state, rng), |s, op, (state, rng)| {
            let op = Self::heat_bath_single_diagonal_update(
                op,
                cutoff,
                s.get_n(),
                beta,
                state,
                (hamiltonian, bond_weights),
                rng,
            );
            (op, (state, rng))
        });
    }

    /// Make the bond weights struct for this container.
    fn make_bond_weights<'b, H, E>(hamiltonian: H, num_bonds: usize, bonds_fn: E) -> BondWeights
    where
        H: Fn(&[usize], usize, &[bool], &[bool]) -> f64,
        E: Fn(usize) -> &'b [usize],
    {
        let max_weights = (0..num_bonds).map(|i| {
            let vars = bonds_fn(i);
            (0..1 << vars.len())
                .map(|substate| {
                    let substate =
                        Self::Op::make_substate((0..vars.len()).map(|v| (substate >> v) & 1 == 1));
                    hamiltonian(vars, i, substate.as_ref(), substate.as_ref())
                })
                .fold(0.0, |acc, w| if w > acc { w } else { acc })
        });
        BondWeights::new(max_weights)
    }

    /// Perform a single heatbath update.
    fn heat_bath_single_diagonal_update<'b, H: Hamiltonian<'b>, R: Rng>(
        op: Option<&Self::Op>,
        cutoff: usize,
        n: usize,
        beta: f64,
        state: &mut [bool],
        hamiltonian_and_weights: (&H, &BondWeights),
        rng: &mut R,
    ) -> Option<Option<Self::Op>> {
        let (hamiltonian, bond_weights) = hamiltonian_and_weights;
        let new_op = match op {
            None => {
                let numerator = beta * bond_weights.total().unwrap();
                let denominator = (cutoff - n) as f64 + numerator;
                if rng.gen_bool(numerator / denominator) {
                    // For usage later.
                    let p = rng.gen_range(0.0, 1.0);
                    // Find the bond to use, weighted by their matrix element.
                    let (b, maxweight) = bond_weights.get_random_bond_and_max_weight(rng).unwrap();
                    let (vars, constant) = hamiltonian.edge_fn(b);
                    let substate = Self::Op::make_substate(vars.iter().map(|v| state[*v]));
                    let vars = Self::Op::make_vars(vars.iter().cloned());

                    let weight = hamiltonian.hamiltonian(
                        vars.as_ref(),
                        b,
                        substate.as_ref(),
                        substate.as_ref(),
                    );

                    // TODO double check this.
                    if p * maxweight < weight {
                        let op = Self::Op::diagonal(vars, b, substate, constant);
                        Some(Some(op))
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            Some(op) if op.is_diagonal() => {
                let numerator = (cutoff - n + 1) as f64;
                let denominator = numerator + beta * bond_weights.total().unwrap();

                // TODO see if any modification is necessary here.
                // let weight = (hamiltonian.hamiltonian)(
                //     op.get_vars(),
                //     op.get_bond(),
                //     op.get_inputs(),
                //     op.get_outputs(),
                // );
                // let maxweight = bond_weights.max_weight_for_bond(op.get_bond());
                // let denominator = denominator * weight / maxweight;

                if rng.gen_bool(numerator / denominator) {
                    Some(None)
                } else {
                    None
                }
            }
            // Update state
            Some(op) => {
                op.get_vars()
                    .iter()
                    .zip(op.get_outputs().iter())
                    .for_each(|(v, b)| state[*v] = *b);
                None
            }
        };
        new_op
    }
}
