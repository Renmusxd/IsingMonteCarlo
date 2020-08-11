use crate::graph::GraphState;
use crate::sse::fast_ops::*;
use crate::sse::qmc_traits::*;
use itertools::Itertools;
use rand::Rng;
#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};

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
}

impl<
        R: Rng,
        M: OpContainerConstructor + DiagonalUpdater + Into<L>,
        L: ClusterUpdater + Into<M>,
    > QMC<R, M, L>
{
    /// Make a new QMC instance with nvars.
    pub fn new(nvars: usize, rng: R) -> Self {
        Self {
            bonds: Vec::default(),
            diagonal_updater: Some(M::new(nvars)),
            loop_updater: None,
            cutoff: nvars,
            state: Some(GraphState::make_random_spin_state(nvars)),
            rng: Some(rng),
            has_cluster_edges: false,
            breaks_ising_symmetry: false,
        }
    }

    /// Add an interaction to the QMC instance.
    pub fn add_interaction<MAT: Into<Vec<f64>>, VAR: Into<Vec<usize>>>(
        &mut self,
        mat: MAT,
        vars: VAR,
    ) -> Result<(), ()> {
        let interaction = Interaction::new(mat, vars)?;
        // Check if this interaction can be used as a
        if is_valid_cluster_edge(interaction.is_constant(), interaction.vars.len()) {
            self.has_cluster_edges = true;
        }
        if !interaction.sym_under_ising() {
            self.breaks_ising_symmetry = true;
        }

        self.bonds.push(interaction);
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
}

impl<R, M, L> QMCStepper for QMC<R, M, L>
where
    R: Rng,
    M: OpContainerConstructor + DiagonalUpdater + Into<L>,
    L: ClusterUpdater + Into<M>,
{
    fn timestep(&mut self, beta: f64) -> &[bool] {
        unimplemented!()
    }

    fn get_n(&self) -> usize {
        match (self.diagonal_updater.as_ref(), self.loop_updater.as_ref()) {
            (Some(m), None) => m.get_n(),
            (None, Some(l)) => l.get_n(),
            _ => unreachable!(),
        }
    }

    fn get_energy_for_average_n(&self, average_n: f64, beta: f64) -> f64 {
        unimplemented!()
    }
}

#[derive(Debug)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
struct Interaction {
    mat: Vec<f64>,
    n: usize,
    vars: Vec<usize>,
    constant: bool,
}

impl Interaction {
    /// Make a new interaction
    fn new<MAT: Into<Vec<f64>>, VAR: Into<Vec<usize>>>(mat: MAT, vars: VAR) -> Result<Self, ()> {
        let mat = mat.into();
        let n = get_mat_var_size(mat.len())?;
        let vars = vars.into();
        if n != vars.len() {
            Err(())
        } else {
            let constant = mat.iter().all_equal();
            Ok(Self {
                mat,
                n,
                vars,
                constant,
            })
        }
    }

    /// Check if interaction is constant.
    fn is_constant(&self) -> bool {
        self.constant
    }

    /// Index into the interaction matrix using inputs and outputs.
    /// Last bit is least significant, inputs are less significant than outputs.
    fn at(&self, inputs: &[bool], outputs: &[bool]) -> Result<f64, ()> {
        if inputs.len() != self.n || outputs.len() != self.n {
            Err(())
        } else {
            let index = outputs
                .iter()
                .chain(inputs.iter())
                .cloned()
                .fold(0usize, |mut acc, b| {
                    acc <<= 1;
                    acc |= if b { 1 } else { 0 };
                    acc
                });
            if index < self.mat.len() {
                Ok(self.mat[index])
            } else {
                Err(())
            }
        }
    }

    /// Check if all entries are symmetric under global flip.
    fn sym_under_ising(&self) -> bool {
        // Mask is 1s along the lower n+1 bits.
        let mask = !(std::usize::MAX << (self.n << 1));
        // Check that each index up to n is equal to its bit-flip counterpart (up to 2n).
        (0..1usize << self.n).all(|indx| self.mat[indx] == self.mat[(!indx) & mask])
    }
}

fn get_mat_var_size(mat_len: usize) -> Result<usize, ()> {
    let mut i = 0;
    let mut x = mat_len >> 1;
    while x > 0 {
        x >>= 1;
        i += 1;
    }
    // Now check that it's just 2^2n
    if 1 << i == mat_len {
        Ok(i >> 1)
    } else {
        Err(())
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
    fn interaction_indexing_single() -> Result<(), ()> {
        let interaction = Interaction {
            mat: vec![1.0, 2.0, 3.0, 4.0],
            n: 1,
            vars: vec![0],
            constant: false,
        };

        assert_eq!(interaction.at(&[false], &[false])?, 1.0);
        assert_eq!(interaction.at(&[true], &[false])?, 2.0);
        assert_eq!(interaction.at(&[false], &[true])?, 3.0);
        assert_eq!(interaction.at(&[true], &[true])?, 4.0);
        Ok(())
    }

    #[test]
    fn interaction_indexing_double() -> Result<(), ()> {
        let interaction = Interaction {
            mat: vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10., 11., 12., 13., 14., 15., 16.,
            ],
            n: 2,
            vars: vec![0, 1],
            constant: false,
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
            mat: vec![1.0, 2.0, 3.0, 4.0],
            n: 1,
            vars: vec![0],
            constant: false,
        };

        assert!(!interaction.sym_under_ising())
    }

    #[test]
    fn ising_flip_check_false_harder() {
        let interaction = Interaction {
            mat: vec![1.0, 2.0, 2.0, 2.0],
            n: 1,
            vars: vec![0],
            constant: false,
        };

        assert!(!interaction.sym_under_ising())
    }

    #[test]
    fn ising_flip_check_true() {
        let interaction = Interaction {
            mat: vec![1.0, 2.0, 2.0, 1.0],
            n: 1,
            vars: vec![0],
            constant: false,
        };

        assert!(interaction.sym_under_ising())
    }

    #[test]
    fn ising_flip_check_true_larger() {
        let interaction = Interaction {
            mat: vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0,
            ],
            n: 2,
            vars: vec![0, 1],
            constant: false,
        };
        assert!(interaction.sym_under_ising())
    }
}
