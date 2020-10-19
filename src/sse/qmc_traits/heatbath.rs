use crate::sse::qmc_traits::{DiagonalUpdater, Hamiltonian};
use crate::sse::Op;
use rand::Rng;
use smallvec::SmallVec;

/// Bond weight storage for fast lookup.
#[derive(Debug)]
pub struct BondWeights {
    weight_and_cumulative: Vec<(f64, f64)>,
    total: f64,
    error: f64,
}

impl BondWeights {
    fn index_for_cumulative(&self, val: f64) -> usize {
        self.weight_and_cumulative
            .binary_search_by(|(_, c)| c.partial_cmp(&val).unwrap())
            .unwrap_or_else(|x| x)
    }

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

/// Heatbath updates for a diagonal updater.
pub trait HeatBathDiagonalUpdater: DiagonalUpdater {
    /// Perform a single heatbath update.
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
        E: Fn(usize) -> (&'b [usize], bool),
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

    /// Perform a single heatbath update.
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
        E: Fn(usize) -> (&'b [usize], bool),
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

    /// Perform a single heatbath update.
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
        E: Fn(usize) -> (&'b [usize], bool),
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

    /// Make the bond weights struct for this container.
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

    /// Perform a single heatbath update.
    fn heat_bath_single_diagonal_update<'b, H, E, R: Rng>(
        op: Option<&Self::Op>,
        cutoff: usize,
        n: usize,
        beta: f64,
        state: &mut [bool],
        hamiltonian_and_weights: (&Hamiltonian<'b, H, E>, BondWeights),
        rng: &mut R,
    ) -> (Option<Option<Self::Op>>, BondWeights)
    where
        H: Fn(&[usize], usize, &[bool], &[bool]) -> f64,
        E: Fn(usize) -> (&'b [usize], bool),
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
                    let (vars, constant) = (hamiltonian.edge_fn)(b);
                    let substate = vars.iter().map(|v| state[*v]);
                    let vars = Self::Op::make_vars(vars.iter().cloned());
                    let substate = Self::Op::make_substate(substate);
                    let op = Self::Op::diagonal(vars, b, substate, constant);
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
                op.get_vars()
                    .iter()
                    .zip(op.get_outputs().iter())
                    .for_each(|(v, b)| state[*v] = *b);
                let weight = (hamiltonian.hamiltonian)(
                    op.get_vars(),
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
