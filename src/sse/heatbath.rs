use crate::sse::qmc_traits::{DiagonalUpdater, Hamiltonian};
use crate::sse::qmc_types::Op;
use rand::Rng;
use smallvec::SmallVec;

/// A container for quickly indexing bond weights.
#[derive(Debug)]
pub struct BondWeights {
    weight_and_cumulative: Vec<(f64, f64)>,
    total: f64,
    error: f64,
}

impl BondWeights {
    /// Given a value find the index which tipped the cumulative weight over that threshold.
    fn index_for_cumulative(&self, val: f64) -> usize {
        self.weight_and_cumulative
            .binary_search_by(|(_, c)| c.partial_cmp(&val).unwrap())
            .unwrap_or_else(|x| x)
    }

    /// Update the weight of a bond.
    fn update_weight(&mut self, b: usize, weight: f64) -> f64 {
        let old_weight = self.weight_and_cumulative[b].0;
        if (old_weight - weight).abs() > self.error {
            // TODO:
            // In the heatbath definition in 1812.05326 we see 2|J| used instead of J,
            // should that become a abs(delta) here?
            let delta = weight - old_weight;
            self.total += delta;
            let n = self.weight_and_cumulative.len();
            self.weight_and_cumulative[b].0 += delta;
            self.weight_and_cumulative[b..n]
                .iter_mut()
                .for_each(|(_, c)| *c += delta);
        }
        old_weight
    }
}

/// Add heatbath diagonal updates to a diagonal updater.
pub trait HeatBathDiagonalUpdater: DiagonalUpdater {
    /// Make a heatbath diagonal update using thread rng.
    fn make_heatbath_diagonal_update<'b, H, E>(
        &mut self,
        cutoff: usize,
        beta: f64,
        state: &[bool],
        hamiltonian: &Hamiltonian<'b, H, E>,
        bond_weights: BondWeights,
    ) -> BondWeights
    where
        H: Fn(&[usize], usize, &[bool], &[bool]) -> f64,
        E: Fn(usize) -> &'b [usize],
    {
        self.make_heatbath_diagonal_update_with_rng(
            cutoff,
            beta,
            state,
            hamiltonian,
            bond_weights,
            &mut rand::thread_rng(),
        )
    }

    /// Make a heatbath diagonal update.
    fn make_heatbath_diagonal_update_with_rng<'b, H, E, R: Rng>(
        &mut self,
        cutoff: usize,
        beta: f64,
        state: &[bool],
        hamiltonian: &Hamiltonian<'b, H, E>,
        bond_weights: BondWeights,
        rng: &mut R,
    ) -> BondWeights
    where
        H: Fn(&[usize], usize, &[bool], &[bool]) -> f64,
        E: Fn(usize) -> &'b [usize],
    {
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

    /// Make a heatbath diagonal update, edit state in place but leave it unchanged after.
    fn make_heatbath_diagonal_update_with_rng_and_state_ref<'b, H, E, R: Rng>(
        &mut self,
        cutoff: usize,
        beta: f64,
        state: &mut [bool],
        hamiltonian: &Hamiltonian<'b, H, E>,
        bond_weights: BondWeights,
        rng: &mut R,
    ) -> BondWeights
    where
        H: Fn(&[usize], usize, &[bool], &[bool]) -> f64,
        E: Fn(usize) -> &'b [usize],
    {
        let (_, _, bond_weights) = self.mutate_ps(
            cutoff,
            (state, rng, bond_weights),
            |s, op, (state, rng, bond_weights)| {
                let (op, bond_weights) = Self::heat_bath_single_diagonal_update(
                    op,
                    cutoff,
                    s.get_n(),
                    beta,
                    state,
                    (&hamiltonian, bond_weights),
                    rng,
                );
                (op, (state, rng, bond_weights))
            },
        );
        bond_weights
    }

    /// Make the bond weights for the system.
    fn make_bond_weights<'b, H, E>(
        state: &[bool],
        hamiltonian: H,
        num_bonds: usize,
        bonds_fn: E,
    ) -> BondWeights
    where
        H: Fn(&[usize], usize, &[bool], &[bool]) -> f64,
        E: Fn(usize) -> &'b [usize],
    {
        let mut total = 0.0;
        let weight_and_cumulative = (0..num_bonds)
            .map(|i| {
                let vars = bonds_fn(i);
                let substate = vars
                    .iter()
                    .map(|v| state[*v])
                    .collect::<SmallVec<[bool; 2]>>();
                let weight = hamiltonian(vars, i, &substate, &substate);
                total += weight;
                (weight, total)
            })
            .collect();
        BondWeights {
            weight_and_cumulative,
            total,
            error: 1e-16,
        }
    }

    /// Logic for a heat bath diagonal update.
    fn heat_bath_single_diagonal_update<'b, H, E, R: Rng>(
        op: Option<&Op>,
        cutoff: usize,
        n: usize,
        beta: f64,
        state: &mut [bool],
        hamiltonian_and_weights: (&Hamiltonian<'b, H, E>, BondWeights),
        rng: &mut R,
    ) -> (Option<Option<Op>>, BondWeights)
    where
        H: Fn(&[usize], usize, &[bool], &[bool]) -> f64,
        E: Fn(usize) -> &'b [usize],
    {
        let (hamiltonian, mut bond_weights) = hamiltonian_and_weights;
        let new_op = match op {
            None => {
                let numerator = beta * bond_weights.total;
                let denominator = (cutoff - n) as f64 + numerator;
                if rng.gen_bool(numerator / denominator) {
                    // Find the bond to use, weighted by their matrix element.
                    let val = rng.gen_range(0.0, bond_weights.total);
                    let b = bond_weights.index_for_cumulative(val);
                    let vars = (hamiltonian.edge_fn)(b);
                    let substate = vars
                        .iter()
                        .map(|v| state[*v])
                        .collect::<SmallVec<[bool; 2]>>();
                    let op = Op::diagonal(vars, b, substate);
                    Some(Some(op))
                } else {
                    None
                }
            }
            Some(op) if op.is_diagonal() => {
                let numerator = (cutoff - n + 1) as f64;
                let denominator = numerator + beta * bond_weights.total;
                if rng.gen_bool(numerator / denominator) {
                    Some(None)
                } else {
                    None
                }
            }
            Some(op) => {
                let vars = op.get_vars();
                vars.iter()
                    .zip(op.get_outputs().iter())
                    .for_each(|(v, b)| state[*v] = *b);
                let weight = (hamiltonian.hamiltonian)(
                    vars,
                    op.get_bond(),
                    op.get_inputs(),
                    op.get_outputs(),
                );
                bond_weights.update_weight(op.get_bond(), weight);
                None
            }
        };
        (new_op, bond_weights)
    }
}
