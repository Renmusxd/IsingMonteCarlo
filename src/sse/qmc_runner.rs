use crate::classical::graph::make_random_spin_state;
use crate::sse::fast_ops::*;
use crate::sse::ham::Ham;
use crate::sse::qmc_traits::*;
#[cfg(feature = "autocorrelations")]
use crate::sse::QmcBondAutoCorrelations;
use crate::sse::{IntoQmc, IsingManager, QmcIsingGraph};
use rand::Rng;
#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};
use std::cmp::max;

/// Default QMC implementation.
pub type DefaultQmc<R> = Qmc<R, FastOps>;

/// QMC with adjustable variables..
#[cfg(feature = "const_generics")]
pub type DefaultQMCN<R, const N: usize> = Qmc<R, FastOpsN<N>>;

/// Trait encompassing requirements for QMC.
pub trait QmcManager: OpContainerConstructor + HeatBathDiagonalUpdater + ClusterUpdater {}

/// A manager for QMC and interactions.
#[derive(Debug)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct Qmc<R, M>
where
    R: Rng,
    M: QmcManager,
{
    bonds: Vec<Interaction>,
    manager: Option<M>,
    cutoff: usize,
    state: Option<Vec<bool>>,
    rng: Option<R>,
    has_cluster_edges: bool,
    breaks_ising_symmetry: bool,
    do_loop_updates: bool,
    offset: f64,
    non_const_diags: Vec<usize>,
    // Heatbath
    do_heatbath: bool,
    bond_weights: Option<BondWeights>,
}

impl<R: Rng, M: QmcManager> Qmc<R, M> {
    /// Make a new QMC instance with `nvars`.
    pub fn new(nvars: usize, mut rng: R, do_loop_updates: bool) -> Self {
        let state = make_random_spin_state(nvars, &mut rng);
        Self::new_with_state(nvars, rng, state, do_loop_updates)
    }

    /// Make a new QMC instance with `nvars`.
    pub fn new_with_state<I: Into<Vec<bool>>>(
        nvars: usize,
        rng: R,
        state: I,
        do_loop_updates: bool,
    ) -> Self {
        Self::new_with_state_with_manager_hook(nvars, rng, state, do_loop_updates, M::new)
    }

    /// Make a new QMC instance with `nvars`. Allows hooking into manager construction with `f`.
    pub fn new_with_state_with_manager_hook<F, I: Into<Vec<bool>>>(
        nvars: usize,
        rng: R,
        state: I,
        do_loop_updates: bool,
        f: F,
    ) -> Self
    where
        F: Fn(usize) -> M,
    {
        Self {
            bonds: Vec::default(),
            manager: Some(f(nvars)),
            cutoff: nvars,
            state: Some(state.into()),
            rng: Some(rng),
            has_cluster_edges: false,
            breaks_ising_symmetry: false,
            do_loop_updates,
            offset: 0.0,
            non_const_diags: Vec::default(),
            do_heatbath: false,
            bond_weights: None,
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

        self.bond_weights = None;
        self.bonds.push(interaction);
    }

    /// Get interactions.
    pub fn get_bonds(&self) -> &[Interaction] {
        &self.bonds
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
        let mut m = self.manager.take().unwrap();
        let mut state = self.state.take().unwrap();
        let mut rng = self.rng.take().unwrap();

        let bonds = &self.bonds;

        let h = |_vars: &[usize], bond: usize, input_state: &[bool], output_state: &[bool]| {
            bonds[bond].at(input_state, output_state).unwrap()
        };

        let num_bonds = bonds.len();
        let bonds_fn = |b: usize| -> (&[usize], bool) { (&bonds[b].vars, bonds[b].is_constant()) };
        let ham = Ham::new(h, bonds_fn, num_bonds);

        if self.do_heatbath {
            if self.bond_weights.is_none() {
                let bond_weights = M::make_bond_weights(&h, num_bonds, |b| bonds_fn(b).0);
                self.bond_weights = Some(bond_weights);
            };
            let bond_weights = self.bond_weights.as_ref().unwrap();
            m.make_heatbath_diagonal_update_with_rng_and_state_ref(
                self.cutoff,
                beta,
                &mut state,
                &ham,
                bond_weights,
                &mut rng,
            )
        } else {
            m.make_diagonal_update_with_rng_and_state_ref(
                self.cutoff,
                beta,
                &mut state,
                &ham,
                &mut rng,
            );
        }
        self.cutoff = max(self.cutoff, m.get_n() + m.get_n() / 2);

        self.state = Some(state);
        self.rng = Some(rng);
        self.manager = Some(m);
    }

    /// Perform a single loop update. Will be inefficient without XX terms.
    pub fn loop_update(&mut self) {
        let mut m = self.manager.take().unwrap();
        let mut state = self.state.take().unwrap();
        let mut rng = self.rng.take().unwrap();

        let bonds = &self.bonds;

        let h = |_vars: &[usize], bond: usize, input_state: &[bool], output_state: &[bool]| {
            bonds[bond].at(input_state, output_state).unwrap()
        };
        m.make_loop_update_with_rng(None, &h, &mut state, &mut rng);

        self.manager = Some(m);
        self.state = Some(state);
        self.rng = Some(rng);
    }

    /// Flip spins using quantum cluster updates if QMC has ising symmetry.
    pub fn cluster_update(&mut self) -> Result<(), &str> {
        if self.breaks_ising_symmetry {
            // TODO remove this restriction.
            Err("Cannot perform cluster updates on graphs that break ising symmetry.")
        } else {
            let mut m = self.manager.take().unwrap();
            let mut state = self.state.take().unwrap();
            let mut rng = self.rng.take().unwrap();
            m.flip_each_cluster_ising_symmetry_rng(0.5, &mut rng, &mut state);

            self.manager = Some(m);
            self.state = Some(state);
            self.rng = Some(rng);
            Ok(())
        }
    }

    /// Flip spins using thermal fluctuations.
    pub fn flip_free_bits(&mut self) {
        let m = self.manager.take().unwrap();

        let mut state = self.state.take().unwrap();
        let mut rng = self.rng.take().unwrap();
        state.iter_mut().enumerate().for_each(|(var, state)| {
            if !m.does_var_have_ops(var) {
                *state = rng.gen_bool(0.5);
            }
        });

        self.manager = Some(m);
        self.state = Some(state);
        self.rng = Some(rng);
    }

    /// Enable or disable the heatbath diagonal update.
    pub fn set_do_heatbath(&mut self, do_heatbath: bool) {
        self.do_heatbath = do_heatbath;
    }

    /// Should the model do heatbath diagonal updates.
    pub fn should_do_heatbath(&self) -> bool {
        self.do_heatbath
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
        // TODO remove ising symmetry restriction.
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
    pub fn get_manager_ref(&self) -> &M {
        self.manager.as_ref().unwrap()
    }

    /// Get the cutoff.
    pub fn get_cutoff(&self) -> usize {
        self.cutoff
    }

    /// Set the cutoff to a new value
    pub fn set_cutoff(&mut self, cutoff: usize) {
        self.cutoff = cutoff;
        self.manager.as_mut().unwrap().set_cutoff(cutoff)
    }

    /// Set the cutoff to a new value so long as that new value is larger than the old one.
    pub fn increase_cutoff_to(&mut self, cutoff: usize) {
        self.set_cutoff(max(self.cutoff, cutoff));
    }

    pub(crate) fn set_manager(&mut self, manager: M) {
        self.manager = Some(manager);
    }

    /// Check if two instances can safely swap managers and initial states
    pub fn can_swap_managers(&self, other: &Self) -> Result<(), String> {
        // TODO check if bonds are multiples of each other
        if self.bonds == other.bonds {
            Ok(())
        } else {
            Err("Bonds not equal".to_string())
        }
    }

    /// Swap managers and initial states
    pub fn swap_manager_and_state(&mut self, other: &mut Self) {
        let m = self.manager.take();
        let s = self.state.take();

        let om = other.manager.take();
        let os = other.state.take();

        self.manager = om;
        self.state = os;

        other.manager = m;
        other.state = s;
    }

    /// Clone the state at p=0.
    pub fn clone_state(&self) -> Vec<bool> {
        self.state.as_ref().unwrap().clone()
    }
}

/// Reference to one of either manager refs
#[derive(Debug)]
pub enum ManagerRef<'a, 'b, M, L> {
    /// Diagonal ref
    Diagonal(&'a M),
    /// Offdiagonal ref
    Looper(&'b L),
}

impl<R, M> QmcStepper for Qmc<R, M>
where
    R: Rng,
    M: QmcManager,
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

    fn get_n(&self) -> usize {
        self.manager.as_ref().unwrap().get_n()
    }

    fn get_energy_for_average_n(&self, average_n: f64, beta: f64) -> f64 {
        let average_energy = -(average_n / beta);
        average_energy + self.offset
    }

    fn state_ref(&self) -> &[bool] {
        self.state.as_ref().unwrap()
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

#[derive(Debug, PartialEq, Eq, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
enum InteractionType {
    Full(bool),
    Diagonal,
}

/// Interactions in QMC.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct Interaction {
    interaction_type: InteractionType,
    mat: Vec<f64>,
    n: usize,
    vars: Vec<usize>,
    constant_along_diagonal: bool,
}

impl PartialEq for Interaction {
    fn eq(&self, other: &Self) -> bool {
        self.interaction_type == other.interaction_type
            && self.n == other.n
            && self.vars == other.vars
            && self.constant_along_diagonal == other.constant_along_diagonal
            && self
                .mat
                .iter()
                .zip(other.mat.iter())
                .all(|(a, b)| (a - b).abs() < std::f64::EPSILON)
    }
}

impl Eq for Interaction {}

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
                interaction_type: InteractionType::Diagonal,
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
                    interaction_type: InteractionType::Full(constant),
                    mat,
                    n,
                    vars,
                    constant_along_diagonal,
                })
            }
        }
    }

    /// Check if interaction is constant.
    pub fn is_constant(&self) -> bool {
        self.interaction_type == InteractionType::Full(true)
    }

    /// Check if constant along diagonal.
    pub fn is_constant_diag(&self) -> bool {
        self.constant_along_diagonal
    }

    /// Index into the interaction matrix using inputs and outputs.
    /// Last bit is least significant, inputs are less significant than outputs.
    pub fn at(&self, inputs: &[bool], outputs: &[bool]) -> Result<f64, String> {
        if inputs.len() != self.n || outputs.len() != self.n {
            Err(format!(
                "Interaction covers {} vars, given ({}/{})",
                self.n,
                inputs.len(),
                outputs.len()
            ))
        } else {
            match &self.interaction_type {
                InteractionType::Full(true) => Ok(self.mat[0]),
                InteractionType::Full(false) => {
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
                InteractionType::Diagonal => {
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

    #[cfg(feature = "autocorrelations")]
    fn at_diag_iter<It>(&self, it: It) -> Result<f64, String>
    where
        It: IntoIterator<Item = bool>,
    {
        let index = if self.constant_along_diagonal {
            0 // Any index works.
        } else {
            match &self.interaction_type {
                InteractionType::Full(true) => 0, // Any index works.
                InteractionType::Diagonal => Self::index_from_iter(it),
                InteractionType::Full(false) => {
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
    pub fn sym_under_ising(&self) -> bool {
        match &self.interaction_type {
            InteractionType::Full(true) => true,
            InteractionType::Diagonal if self.constant_along_diagonal => true,
            InteractionType::Full(false) => {
                // Mask is 1s along the lower n+1 bits.
                let mask = !(std::usize::MAX << (self.n << 1));
                // Check that each index up to n is equal to its bit-flip counterpart (up to 2n).
                (0..1usize << self.n).all(|indx| {
                    (self.mat[indx] - self.mat[(!indx) & mask]).abs() < std::f64::EPSILON
                })
            }
            InteractionType::Diagonal => {
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

    fn index_from_iter<It>(it: It) -> usize
    where
        It: IntoIterator<Item = bool>,
    {
        it.into_iter().fold(0usize, |mut acc, b| {
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

// Allow for conversion to generic QMC type. Clears the internal state, converts edges and field
// into interactions.
impl<R, M> From<QmcIsingGraph<R, M>> for Qmc<R, M>
where
    R: Rng,
    M: IsingManager + QmcManager,
{
    fn from(g: QmcIsingGraph<R, M>) -> Self {
        g.into_qmc()
    }
}

impl<R, M> Clone for Qmc<R, M>
where
    R: Rng + Clone,
    M: QmcManager + Clone,
{
    fn clone(&self) -> Self {
        Self {
            bonds: self.bonds.clone(),
            manager: self.manager.clone(),
            cutoff: self.cutoff,
            state: self.state.clone(),
            rng: self.rng.clone(),
            has_cluster_edges: self.has_cluster_edges,
            breaks_ising_symmetry: self.breaks_ising_symmetry,
            do_loop_updates: self.do_loop_updates,
            offset: self.offset,
            non_const_diags: self.non_const_diags.clone(),
            do_heatbath: self.do_heatbath,
            bond_weights: self.bond_weights.clone(),
        }
    }
}

#[cfg(feature = "autocorrelations")]
impl<R, M> QmcBondAutoCorrelations for Qmc<R, M>
where
    R: Rng,
    M: QmcManager,
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

    fn almost_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < f64::EPSILON
    }

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
            interaction_type: InteractionType::Full(false),
            mat: vec![1.0, 2.0, 3.0, 4.0],
            n: 1,
            vars: vec![0],
            constant_along_diagonal: false,
        };

        assert!(almost_eq(interaction.at(&[false], &[false])?, 1.0));
        assert!(almost_eq(interaction.at(&[true], &[false])?, 2.0));
        assert!(almost_eq(interaction.at(&[false], &[true])?, 3.0));
        assert!(almost_eq(interaction.at(&[true], &[true])?, 4.0));
        Ok(())
    }

    #[test]
    fn interaction_indexing_double() -> Result<(), String> {
        let interaction = Interaction {
            interaction_type: InteractionType::Full(false),
            mat: vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10., 11., 12., 13., 14., 15., 16.,
            ],
            n: 2,
            vars: vec![0, 1],
            constant_along_diagonal: false,
        };

        assert!(almost_eq(
            interaction.at(&[false, false], &[false, false])?,
            1.0
        ));
        assert!(almost_eq(
            interaction.at(&[false, true], &[false, false])?,
            2.0
        ));
        assert!(almost_eq(
            interaction.at(&[true, false], &[false, false])?,
            3.0
        ));
        assert!(almost_eq(
            interaction.at(&[true, true], &[false, false])?,
            4.0
        ));

        assert!(almost_eq(
            interaction.at(&[false, false], &[false, true])?,
            5.0
        ));
        assert!(almost_eq(
            interaction.at(&[false, true], &[false, true])?,
            6.0
        ));
        assert!(almost_eq(
            interaction.at(&[true, false], &[false, true])?,
            7.0
        ));
        assert!(almost_eq(
            interaction.at(&[true, true], &[false, true])?,
            8.0
        ));

        assert!(almost_eq(
            interaction.at(&[false, false], &[true, false])?,
            9.0
        ));
        assert!(almost_eq(
            interaction.at(&[false, true], &[true, false])?,
            10.
        ));
        assert!(almost_eq(
            interaction.at(&[true, false], &[true, false])?,
            11.
        ));
        assert!(almost_eq(
            interaction.at(&[true, true], &[true, false])?,
            12.
        ));

        assert!(almost_eq(
            interaction.at(&[false, false], &[true, true])?,
            13.
        ));
        assert!(almost_eq(
            interaction.at(&[false, true], &[true, true])?,
            14.
        ));
        assert!(almost_eq(
            interaction.at(&[true, false], &[true, true])?,
            15.
        ));
        assert!(almost_eq(
            interaction.at(&[true, true], &[true, true])?,
            16.
        ));
        Ok(())
    }

    #[test]
    fn ising_flip_check_false() {
        let interaction = Interaction {
            interaction_type: InteractionType::Full(false),
            mat: vec![1.0, 2.0, 3.0, 4.0],
            n: 1,
            vars: vec![0],
            constant_along_diagonal: false,
        };

        assert!(!interaction.sym_under_ising())
    }

    #[test]
    fn ising_flip_check_false_harder() {
        let interaction = Interaction {
            interaction_type: InteractionType::Full(false),
            mat: vec![1.0, 2.0, 2.0, 2.0],
            n: 1,
            vars: vec![0],
            constant_along_diagonal: false,
        };

        assert!(!interaction.sym_under_ising())
    }

    #[test]
    fn ising_flip_check_true() {
        let interaction = Interaction {
            interaction_type: InteractionType::Full(false),
            mat: vec![1.0, 2.0, 2.0, 1.0],
            n: 1,
            vars: vec![0],
            constant_along_diagonal: false,
        };

        assert!(interaction.sym_under_ising())
    }

    #[test]
    fn ising_flip_check_true_larger() {
        let interaction = Interaction {
            interaction_type: InteractionType::Full(false),
            mat: vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0,
            ],
            n: 2,
            vars: vec![0, 1],
            constant_along_diagonal: false,
        };
        assert!(interaction.sym_under_ising())
    }

    #[test]
    fn ising_flip_check_true_larger_diagonal() {
        let interaction = Interaction {
            interaction_type: InteractionType::Diagonal,
            mat: vec![1.0, 6.0, 6.0, 1.0],
            n: 2,
            vars: vec![0, 1],
            constant_along_diagonal: false,
        };
        assert!(interaction.sym_under_ising())
    }

    #[test]
    fn ising_flip_check_false_larger_diagonal() {
        let interaction = Interaction {
            interaction_type: InteractionType::Diagonal,
            mat: vec![1.0, 1.0, 6.0, 6.0],
            n: 2,
            vars: vec![0, 1],
            constant_along_diagonal: false,
        };
        assert!(!interaction.sym_under_ising())
    }
}
