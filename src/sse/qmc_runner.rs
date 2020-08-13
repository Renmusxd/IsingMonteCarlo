use crate::classical::graph::GraphState;
use crate::sse::fast_ops::*;
use crate::sse::qmc_traits::*;
#[cfg(feature = "autocorrelations")]
use crate::sse::QMCBondAutoCorrelations;
use rand::Rng;
#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};
use std::cmp::max;

/// Default QMC implementation.
pub type DefaultQMC<R> = QMC<R, FastOps, FastOps>;

/// QMC with adjustable variables..
#[cfg(feature = "const_generics")]
pub type DefaultQMCN<R, const N: usize> = QMC<R, FastOpsN<N>, FastOpsN<N>>;

/// A manager for QMC and interactions.
#[derive(Debug)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct QMC<R, M, L>
where
    R: Rng,
    M: OpContainerConstructor + DiagonalUpdater + Into<L>,
    L: ClusterUpdater + Into<M>,
{
    bonds: Vec<Interaction>,
    diagonal_updater: Option<M>,
    loop_updater: Option<L>,
    cutoff: usize,
    state: Option<Vec<bool>>,
    rng: Option<R>,
    has_cluster_edges: bool,
    breaks_ising_symmetry: bool,
    do_loop_updates: bool,
    offset: f64,
    non_const_diags: Vec<usize>,
}

impl<
        R: Rng,
        M: OpContainerConstructor + DiagonalUpdater + Into<L>,
        L: ClusterUpdater + Into<M>,
    > QMC<R, M, L>
{
    /// Make a new QMC instance with nvars.
    pub fn new(nvars: usize, rng: R, do_loop_updates: bool) -> Self {
        Self::new_with_state(
            nvars,
            rng,
            GraphState::make_random_spin_state(nvars),
            do_loop_updates,
        )
    }

    /// Make a new QMC instance with nvars.
    pub fn new_with_state<I: Into<Vec<bool>>>(
        nvars: usize,
        rng: R,
        state: I,
        do_loop_updates: bool,
    ) -> Self {
        Self {
            bonds: Vec::default(),
            diagonal_updater: Some(M::new(nvars)),
            loop_updater: None,
            cutoff: nvars,
            state: Some(state.into()),
            rng: Some(rng),
            has_cluster_edges: false,
            breaks_ising_symmetry: false,
            do_loop_updates,
            offset: 0.0,
            non_const_diags: Vec::default(),
        }
    }

    /// Add an interaction to the QMC instance.
    fn add_interaction(&mut self, interaction: Interaction) {
        // Check if this interaction can be used as a
        if is_valid_cluster_edge(interaction.is_constant(), interaction.vars.len()) {
            self.has_cluster_edges = true;
        }
        if !interaction.sym_under_ising() {
            self.breaks_ising_symmetry = true;
        }
        if !interaction.is_constant_diag() {
            self.non_const_diags.push(self.bonds.len())
        }

        self.bonds.push(interaction);
    }

    /// Add an interaction to the QMC instance.
    pub fn make_interaction<MAT: Into<Vec<f64>>, VAR: Into<Vec<usize>>>(
        &mut self,
        mat: MAT,
        vars: VAR,
    ) -> Result<(), String> {
        let interaction = Interaction::new(mat, vars)?;
        self.add_interaction(interaction);
        Ok(())
    }

    /// Add an interaction to the QMC instance, adjust with a diagonal offset.
    pub fn make_interaction_and_offset<MAT: Into<Vec<f64>>, VAR: Into<Vec<usize>>>(
        &mut self,
        mat: MAT,
        vars: VAR,
    ) -> Result<(), String> {
        let (interaction, offset) = Interaction::new_offset(mat, vars)?;
        self.add_interaction(interaction);
        self.offset -= offset;
        Ok(())
    }

    /// Add an interaction to the QMC instance.
    pub fn make_diagonal_interaction<MAT: Into<Vec<f64>>, VAR: Into<Vec<usize>>>(
        &mut self,
        mat: MAT,
        vars: VAR,
    ) -> Result<(), String> {
        let interaction = Interaction::new_diagonal(mat, vars)?;
        self.add_interaction(interaction);
        Ok(())
    }

    /// Add an interaction to the QMC instance, adjust with a diagonal offset.
    pub fn make_diagonal_interaction_and_offset<MAT: Into<Vec<f64>>, VAR: Into<Vec<usize>>>(
        &mut self,
        mat: MAT,
        vars: VAR,
    ) -> Result<(), String> {
        let (interaction, offset) = Interaction::new_diagonal_offset(mat, vars)?;
        self.add_interaction(interaction);
        self.offset -= offset;
        Ok(())
    }

    /// Perform a single diagonal update.
    pub fn diagonal_update(&mut self, beta: f64) {
        let mut m = match (self.diagonal_updater.take(), self.loop_updater.take()) {
            (Some(m), None) => m,
            (None, Some(l)) => l.into(),
            _ => unreachable!(),
        };
        let mut state = self.state.take().unwrap();
        let mut rng = self.rng.take().unwrap();

        let bonds = &self.bonds;

        let h = |_vars: &[usize], bond: usize, input_state: &[bool], output_state: &[bool]| {
            bonds[bond].at(input_state, output_state).unwrap()
        };

        let num_bonds = bonds.len();
        let bonds_fn = |b: usize| -> (&[usize], bool) { (&bonds[b].vars, bonds[b].is_constant()) };

        let ham = Hamiltonian::new(h, bonds_fn, num_bonds);

        m.make_diagonal_update_with_rng_and_state_ref(
            self.cutoff,
            beta,
            &mut state,
            &ham,
            &mut rng,
        );
        self.cutoff = max(self.cutoff, m.get_n() + m.get_n() / 2);

        self.state = Some(state);
        self.rng = Some(rng);
        self.diagonal_updater = Some(m);
    }

    /// Perform a single loop update. Will be inefficient without XX terms.
    pub fn loop_update(&mut self) {
        let mut l = match (self.diagonal_updater.take(), self.loop_updater.take()) {
            (Some(m), None) => m.into(),
            (None, Some(l)) => l,
            _ => unreachable!(),
        };
        let mut state = self.state.take().unwrap();
        let mut rng = self.rng.take().unwrap();

        let bonds = &self.bonds;

        let h = |_vars: &[usize], bond: usize, input_state: &[bool], output_state: &[bool]| {
            bonds[bond].at(input_state, output_state).unwrap()
        };
        l.make_loop_update_with_rng(None, &h, &mut state, &mut rng);

        self.loop_updater = Some(l);
        self.state = Some(state);
        self.rng = Some(rng);
    }

    /// Flip spins using quantum cluster updates if QMC has ising symmetry.
    pub fn cluster_update(&mut self) -> Result<(), ()> {
        if self.breaks_ising_symmetry {
            Err(())
        } else {
            let mut l = match (self.diagonal_updater.take(), self.loop_updater.take()) {
                (Some(m), None) => m.into(),
                (None, Some(l)) => l,
                _ => unreachable!(),
            };
            let mut state = self.state.take().unwrap();
            let mut rng = self.rng.take().unwrap();
            l.flip_each_cluster_rng(0.5, &mut rng, &mut state);

            self.loop_updater = Some(l);
            self.state = Some(state);
            self.rng = Some(rng);
            Ok(())
        }
    }

    /// Flip spins using thermal fluctuations.
    pub fn flip_free_bits(&mut self) {
        let l = match (self.diagonal_updater.take(), self.loop_updater.take()) {
            (Some(m), None) => m.into(),
            (None, Some(l)) => l,
            _ => unreachable!(),
        };

        let mut state = self.state.take().unwrap();
        let mut rng = self.rng.take().unwrap();
        state.iter_mut().enumerate().for_each(|(var, state)| {
            if !l.does_var_have_ops(var) {
                *state = rng.gen_bool(0.5);
            }
        });

        self.loop_updater = Some(l);
        self.state = Some(state);
        self.rng = Some(rng);
    }

    /// Change whether loop updates will be performed.
    pub fn set_do_loop_updates(&mut self, do_loop_updates: bool) {
        self.do_loop_updates = do_loop_updates;
    }

    /// Should the model do loop updates.
    pub fn should_do_loop_update(&self) -> bool {
        self.do_loop_updates
    }

    /// Should the model do cluster updates.
    pub fn should_do_cluster_update(&self) -> bool {
        !self.breaks_ising_symmetry && self.has_cluster_edges
    }

    /// Convert the state to a vector.
    pub fn into_vec(self) -> Vec<bool> {
        self.state.unwrap()
    }

    /// Get the total energy offset.
    pub fn get_offset(&self) -> f64 {
        self.offset
    }

    /// Get a reference to the diagonal op manager.
    pub fn get_diagonal_manager_ref(&mut self) -> &M {
        let m = match (self.diagonal_updater.take(), self.loop_updater.take()) {
            (Some(m), None) => m,
            (None, Some(l)) => l.into(),
            _ => unreachable!(),
        };
        self.diagonal_updater = Some(m);
        self.diagonal_updater.as_ref().unwrap()
    }

    /// Get a reference to the loop op manager.
    pub fn get_loop_manager_ref(&mut self) -> &L {
        let l = match (self.diagonal_updater.take(), self.loop_updater.take()) {
            (Some(m), None) => m.into(),
            (None, Some(l)) => l,
            _ => unreachable!(),
        };
        self.loop_updater = Some(l);
        self.loop_updater.as_ref().unwrap()
    }

    /// Set the cutoff to a new value so long as that new value is larger than the old one.
    pub fn increase_cutoff_to(&mut self, cutoff: usize) {
        self.cutoff = max(self.cutoff, cutoff);
    }

    pub(crate) fn set_diagonal_manager(&mut self, manager: M) {
        self.loop_updater = None;
        self.diagonal_updater = Some(manager);
    }
}

impl<R, M, L> QMCStepper for QMC<R, M, L>
where
    R: Rng,
    M: OpContainerConstructor + DiagonalUpdater + Into<L>,
    L: ClusterUpdater + Into<M>,
{
    fn timestep(&mut self, beta: f64) -> &[bool] {
        self.diagonal_update(beta);

        if self.should_do_loop_update() {
            self.loop_update();
        }

        if self.should_do_cluster_update() {
            self.cluster_update().unwrap();
        }

        self.flip_free_bits();

        self.state.as_ref().unwrap()
    }

    fn state_ref(&self) -> &[bool] {
        self.state.as_ref().unwrap()
    }

    fn get_n(&self) -> usize {
        match (self.diagonal_updater.as_ref(), self.loop_updater.as_ref()) {
            (Some(m), None) => m.get_n(),
            (None, Some(l)) => l.get_n(),
            _ => unreachable!(),
        }
    }

    fn get_energy_for_average_n(&self, average_n: f64, beta: f64) -> f64 {
        let average_energy = -(average_n / beta);
        average_energy + self.offset
    }
}

#[derive(Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
enum InteractionType {
    FULL(bool),
    DIAGONAL,
}

#[derive(Debug)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
struct Interaction {
    interaction_type: InteractionType,
    mat: Vec<f64>,
    n: usize,
    vars: Vec<usize>,
    constant_along_diagonal: bool,
}

impl Interaction {
    fn new_diagonal_offset<MAT: Into<Vec<f64>>, VAR: Into<Vec<usize>>>(
        mat: MAT,
        vars: VAR,
    ) -> Result<(Self, f64), String> {
        let mut mat = mat.into();
        let min_diag = mat
            .iter()
            .fold(f64::MAX, |acc, item| if acc < *item { acc } else { *item });
        mat.iter_mut().for_each(|f| *f -= min_diag);
        Self::new_diagonal(mat, vars).map(|int| (int, min_diag))
    }

    fn new_diagonal<MAT: Into<Vec<f64>>, VAR: Into<Vec<usize>>>(
        mat: MAT,
        vars: VAR,
    ) -> Result<Self, String> {
        let mat = mat.into();
        let n = get_power_of_two(mat.len())
            .map_err(|_| format!("Matrix size must be power of 2, was {}", mat.len()))?;
        let vars = vars.into();
        let constant_along_diagonal = mat
            .iter()
            .cloned()
            .try_fold(None, |acc, item| -> Result<Option<f64>, ()> {
                match acc {
                    None => Ok(Some(item)),
                    Some(old) => {
                        if (old - item).abs() < std::f64::EPSILON {
                            Ok(Some(item))
                        } else {
                            Err(())
                        }
                    }
                }
            })
            .is_ok();
        if n == vars.len() {
            Ok(Interaction {
                interaction_type: InteractionType::DIAGONAL,
                mat,
                n,
                vars,
                constant_along_diagonal,
            })
        } else {
            Err(format!("Given {} vars, expected {}", vars.len(), n))
        }
    }

    fn new_offset<MAT: Into<Vec<f64>>, VAR: Into<Vec<usize>>>(
        mat: MAT,
        vars: VAR,
    ) -> Result<(Self, f64), String> {
        let mut mat = mat.into();
        let n = get_mat_var_size(mat.len())
            .map_err(|_| format!("Matrix size must be power of 2, was {}", mat.len()))?;
        let tn = 1 << n;
        let min_diag = (0..tn)
            .map(|indx| mat[(1 + tn) * indx])
            .fold(f64::MAX, |acc, item| if acc < item { acc } else { item });
        (0..tn).for_each(|indx| mat[(1 + tn) * indx] -= min_diag);
        Self::new(mat, vars).map(|int| (int, min_diag))
    }

    /// Make a new interaction
    fn new<MAT: Into<Vec<f64>>, VAR: Into<Vec<usize>>>(
        mat: MAT,
        vars: VAR,
    ) -> Result<Self, String> {
        let mat = mat.into();
        if mat.iter().any(|m| *m < 0.) {
            Err("Interaction contains negative weights".to_string())
        } else {
            let n = get_mat_var_size(mat.len())
                .map_err(|_| format!("Matrix size must be power of 2, was {}", mat.len()))?;
            let vars = vars.into();
            if n != vars.len() {
                Err(format!("Given {} vars, expected {}", vars.len(), n))
            } else {
                let constant = mat
                    .iter()
                    .cloned()
                    .try_fold(None, |acc, item| -> Result<Option<f64>, ()> {
                        match acc {
                            None => Ok(Some(item)),
                            Some(old) => {
                                if (old - item).abs() < std::f64::EPSILON {
                                    Ok(Some(item))
                                } else {
                                    Err(())
                                }
                            }
                        }
                    })
                    .is_ok();
                let constant_along_diagonal = (0..1 << n)
                    .map(|row| mat[(row << n) + row])
                    .try_fold(None, |acc, item| -> Result<Option<f64>, ()> {
                        match acc {
                            None => Ok(Some(item)),
                            Some(old) => {
                                if (old - item).abs() < std::f64::EPSILON {
                                    Ok(Some(item))
                                } else {
                                    Err(())
                                }
                            }
                        }
                    })
                    .is_ok();
                Ok(Self {
                    interaction_type: InteractionType::FULL(constant),
                    mat,
                    n,
                    vars,
                    constant_along_diagonal,
                })
            }
        }
    }

    /// Check if interaction is constant.
    fn is_constant(&self) -> bool {
        self.interaction_type == InteractionType::FULL(true)
    }

    /// Check if constant along diagonal.
    fn is_constant_diag(&self) -> bool {
        self.constant_along_diagonal
    }

    /// Index into the interaction matrix using inputs and outputs.
    /// Last bit is least significant, inputs are less significant than outputs.
    fn at(&self, inputs: &[bool], outputs: &[bool]) -> Result<f64, String> {
        if inputs.len() != self.n || outputs.len() != self.n {
            Err(format!(
                "Interaction covers {} vars, given ({}/{})",
                self.n,
                inputs.len(),
                outputs.len()
            ))
        } else {
            match &self.interaction_type {
                InteractionType::FULL(true) => Ok(self.mat[0]),
                InteractionType::FULL(false) => {
                    let index = Self::index_from_state(inputs, outputs);
                    if index < self.mat.len() {
                        Ok(self.mat[index])
                    } else {
                        Err(format!(
                            "Index {} out of bounds for interaction with {} vars",
                            index, self.n
                        ))
                    }
                }
                InteractionType::DIAGONAL => {
                    if inputs == outputs {
                        let index = Self::index_from_state(inputs, &[]);
                        if index < self.mat.len() {
                            Ok(self.mat[index])
                        } else {
                            Err(format!(
                                "Index {} out of bounds for interaction with {} vars",
                                index, self.n
                            ))
                        }
                    } else {
                        Ok(0.0)
                    }
                }
            }
        }
    }

    fn at_diag_iter<It: Iterator<Item = bool>>(&self, it: It) -> Result<f64, String> {
        let index = if self.constant_along_diagonal {
            0 // Any index works.
        } else {
            match &self.interaction_type {
                InteractionType::FULL(true) => 0, // Any index works.
                InteractionType::DIAGONAL => Self::index_from_iter(it),
                InteractionType::FULL(false) => {
                    let index = Self::index_from_iter(it);
                    (index << self.n) + index
                }
            }
        };
        if index < self.mat.len() {
            Ok(self.mat[index])
        } else {
            Err(format!(
                "Matrix is of length {}, index {} is invalid",
                self.mat.len(),
                index
            ))
        }
    }

    /// Check if all entries are symmetric under global flip.
    fn sym_under_ising(&self) -> bool {
        match &self.interaction_type {
            InteractionType::FULL(true) => true,
            InteractionType::DIAGONAL if self.constant_along_diagonal => true,
            InteractionType::FULL(false) => {
                // Mask is 1s along the lower n+1 bits.
                let mask = !(std::usize::MAX << (self.n << 1));
                // Check that each index up to n is equal to its bit-flip counterpart (up to 2n).
                (0..1usize << self.n).all(|indx| {
                    (self.mat[indx] - self.mat[(!indx) & mask]).abs() < std::f64::EPSILON
                })
            }
            InteractionType::DIAGONAL => {
                // Mask is 1s along the lower n bits.
                let mask = !(std::usize::MAX << self.n);
                // Check that each index up to n is equal to its bit-flip counterpart (up to 2n).
                (0..1usize << (self.n >> 1)).all(|indx| {
                    (self.mat[indx] - self.mat[(!indx) & mask]).abs() < std::f64::EPSILON
                })
            }
        }
    }

    fn index_from_state(inputs: &[bool], outputs: &[bool]) -> usize {
        Self::index_from_iter(outputs.iter().chain(inputs.iter()).cloned())
    }

    fn index_from_iter<It: Iterator<Item = bool>>(it: It) -> usize {
        it.fold(0usize, |mut acc, b| {
            acc <<= 1;
            acc |= if b { 1 } else { 0 };
            acc
        })
    }
}

fn get_mat_var_size(mat_len: usize) -> Result<usize, ()> {
    get_power_of_two(mat_len).map(|i| i >> 1)
}

fn get_power_of_two(n: usize) -> Result<usize, ()> {
    let mut i = 0;
    let mut x = n >> 1;
    while x > 0 {
        x >>= 1;
        i += 1;
    }
    // Now check that it's just 2^2i
    if 1 << i == n {
        Ok(i)
    } else {
        Err(())
    }
}

#[cfg(feature = "autocorrelations")]
impl<R, M, L> QMCBondAutoCorrelations for QMC<R, M, L>
where
    R: Rng,
    M: OpContainerConstructor + DiagonalUpdater + Into<L>,
    L: ClusterUpdater + Into<M>,
{
    fn n_bonds(&self) -> usize {
        self.non_const_diags.len()
    }

    fn value_for_bond(&self, bond: usize, sample: &[bool]) -> f64 {
        let bond = &self.bonds[self.non_const_diags[bond]];
        bond.at_diag_iter(bond.vars.iter().map(|v| sample[*v]))
            .unwrap()
    }
}

#[cfg(test)]
mod qmc_tests {
    use super::*;

    #[test]
    fn mat_var_test_single() -> Result<(), ()> {
        // A 2x2 matrix has 4 entries and covers a single variable.
        let v = get_mat_var_size(4)?;
        assert_eq!(v, 1);
        Ok(())
    }

    #[test]
    fn mat_var_test_double() -> Result<(), ()> {
        // A 4x4 matrix has 16 entries and covers two variables.
        let v = get_mat_var_size(16)?;
        assert_eq!(v, 2);
        Ok(())
    }

    #[test]
    fn mat_var_test_triple() -> Result<(), ()> {
        // An 8x8 matrix has 64 entries and covers three variables.
        let v = get_mat_var_size(64)?;
        assert_eq!(v, 3);
        Ok(())
    }

    #[test]
    fn interaction_indexing_single() -> Result<(), String> {
        let interaction = Interaction {
            interaction_type: InteractionType::FULL(false),
            mat: vec![1.0, 2.0, 3.0, 4.0],
            n: 1,
            vars: vec![0],
        };

        assert_eq!(interaction.at(&[false], &[false])?, 1.0);
        assert_eq!(interaction.at(&[true], &[false])?, 2.0);
        assert_eq!(interaction.at(&[false], &[true])?, 3.0);
        assert_eq!(interaction.at(&[true], &[true])?, 4.0);
        Ok(())
    }

    #[test]
    fn interaction_indexing_double() -> Result<(), String> {
        let interaction = Interaction {
            interaction_type: InteractionType::FULL(false),
            mat: vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10., 11., 12., 13., 14., 15., 16.,
            ],
            n: 2,
            vars: vec![0, 1],
        };

        assert_eq!(interaction.at(&[false, false], &[false, false])?, 1.0);
        assert_eq!(interaction.at(&[false, true], &[false, false])?, 2.0);
        assert_eq!(interaction.at(&[true, false], &[false, false])?, 3.0);
        assert_eq!(interaction.at(&[true, true], &[false, false])?, 4.0);

        assert_eq!(interaction.at(&[false, false], &[false, true])?, 5.0);
        assert_eq!(interaction.at(&[false, true], &[false, true])?, 6.0);
        assert_eq!(interaction.at(&[true, false], &[false, true])?, 7.0);
        assert_eq!(interaction.at(&[true, true], &[false, true])?, 8.0);

        assert_eq!(interaction.at(&[false, false], &[true, false])?, 9.0);
        assert_eq!(interaction.at(&[false, true], &[true, false])?, 10.);
        assert_eq!(interaction.at(&[true, false], &[true, false])?, 11.);
        assert_eq!(interaction.at(&[true, true], &[true, false])?, 12.);

        assert_eq!(interaction.at(&[false, false], &[true, true])?, 13.);
        assert_eq!(interaction.at(&[false, true], &[true, true])?, 14.);
        assert_eq!(interaction.at(&[true, false], &[true, true])?, 15.);
        assert_eq!(interaction.at(&[true, true], &[true, true])?, 16.);
        Ok(())
    }

    #[test]
    fn ising_flip_check_false() {
        let interaction = Interaction {
            interaction_type: InteractionType::FULL(false),
            mat: vec![1.0, 2.0, 3.0, 4.0],
            n: 1,
            vars: vec![0],
        };

        assert!(!interaction.sym_under_ising())
    }

    #[test]
    fn ising_flip_check_false_harder() {
        let interaction = Interaction {
            interaction_type: InteractionType::FULL(false),
            mat: vec![1.0, 2.0, 2.0, 2.0],
            n: 1,
            vars: vec![0],
        };

        assert!(!interaction.sym_under_ising())
    }

    #[test]
    fn ising_flip_check_true() {
        let interaction = Interaction {
            interaction_type: InteractionType::FULL(false),
            mat: vec![1.0, 2.0, 2.0, 1.0],
            n: 1,
            vars: vec![0],
        };

        assert!(interaction.sym_under_ising())
    }

    #[test]
    fn ising_flip_check_true_larger() {
        let interaction = Interaction {
            interaction_type: InteractionType::FULL(false),
            mat: vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0,
            ],
            n: 2,
            vars: vec![0, 1],
        };
        assert!(interaction.sym_under_ising())
    }

    #[test]
    fn ising_flip_check_true_larger_diagonal() {
        let interaction = Interaction {
            interaction_type: InteractionType::DIAGONAL,
            mat: vec![1.0, 6.0, 6.0, 1.0],
            n: 2,
            vars: vec![0, 1],
        };
        assert!(interaction.sym_under_ising())
    }

    #[test]
    fn ising_flip_check_false_larger_diagonal() {
        let interaction = Interaction {
            interaction_type: InteractionType::DIAGONAL,
            mat: vec![1.0, 1.0, 6.0, 6.0],
            n: 2,
            vars: vec![0, 1],
        };
        assert!(!interaction.sym_under_ising())
    }
}
