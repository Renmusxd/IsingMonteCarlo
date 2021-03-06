use crate::sse::QmcStepper;
use rustfft::{num_complex::Complex, FftPlanner};
use std::ops::DivAssign;

/// Calculate autocorrelations for a QMCStepper
pub trait QmcAutoCorrelations: QmcStepper {
    /// Calculate the autcorrelation calculations for the results of f(state).
    fn calculate_autocorrelation<F>(
        &mut self,
        timesteps: usize,
        beta: f64,
        sampling_freq: Option<usize>,
        sample_mapper: F,
    ) -> Vec<f64>
    where
        F: Fn(&Self, Vec<bool>) -> Vec<f64>,
    {
        let acc = Vec::with_capacity(timesteps / sampling_freq.unwrap_or(1) + 1);
        let (samples, _) = self.timesteps_measure(
            timesteps,
            beta,
            acc,
            |mut acc, state| {
                acc.push(state.to_vec());
                acc
            },
            sampling_freq,
        );
        let samples = samples
            .into_iter()
            .map(|s| sample_mapper(self, s))
            .collect::<Vec<Vec<f64>>>();

        fft_autocorrelation(&samples)
    }

    /// Calculate the autocorrelation of variables.
    fn calculate_variable_autocorrelation(
        &mut self,
        timesteps: usize,
        beta: f64,
        sampling_freq: Option<usize>,
    ) -> Vec<f64> {
        self.calculate_autocorrelation(timesteps, beta, sampling_freq, |_, sample| {
            sample
                .into_iter()
                .map(|b| if b { 1.0 } else { -1.0 })
                .collect()
        })
    }

    /// Take autocorrelations of products of spins.
    fn calculate_spin_product_autocorrelation(
        &mut self,
        timesteps: usize,
        beta: f64,
        var_products: &[&[usize]],
        sampling_freq: Option<usize>,
    ) -> Vec<f64> {
        self.calculate_autocorrelation(timesteps, beta, sampling_freq, |_, sample| {
            var_products
                .iter()
                .map(|vs| {
                    vs.iter()
                        .map(|v| if sample[*v] { 1.0 } else { -1.0 })
                        .product()
                })
                .collect()
        })
    }
}

impl<Q: QmcStepper> QmcAutoCorrelations for Q {}

/// Calculate bond autocorrelations for a QMCAutoCorrelations
pub trait QmcBondAutoCorrelations: QmcAutoCorrelations {
    /// Number of bonds
    fn n_bonds(&self) -> usize;

    /// Whether bond is satisfied.
    fn value_for_bond(&self, bond: usize, sample: &[bool]) -> f64;

    /// Calculate the autcorrelation calculations for bonds.
    fn calculate_bond_autocorrelation(
        &mut self,
        timesteps: usize,
        beta: f64,
        sampling_freq: Option<usize>,
    ) -> Vec<f64> {
        let nbonds = self.n_bonds();
        self.calculate_autocorrelation(timesteps, beta, sampling_freq, |s, sample| {
            (0..nbonds)
                .map(|bond| s.value_for_bond(bond, &sample))
                .collect()
        })
    }
}

pub(crate) fn fft_autocorrelation(samples: &[Vec<f64>]) -> Vec<f64> {
    let tmax = samples.len();
    let n = samples[0].len();

    let means = (0..n)
        .map(|i| (0..tmax).map(|t| samples[t][i]).sum::<f64>() / tmax as f64)
        .collect::<Vec<_>>();

    let mut input = (0..n)
        .map(|i| {
            let mut v = (0..tmax)
                .map(|t| Complex::<f64>::new(samples[t][i] - means[i], 0.0))
                .collect::<Vec<Complex<f64>>>();
            let norm = v.iter().map(|v| (v.conj() * v).re).sum::<f64>().sqrt();
            v.iter_mut().for_each(|c| c.div_assign(norm));
            v
        })
        .collect::<Vec<_>>();
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(tmax);
    let mut iplanner = FftPlanner::new();
    let ifft = iplanner.plan_fft_inverse(tmax);

    input.iter_mut().for_each(|input| {
        fft.process(input);
        input
            .iter_mut()
            .for_each(|c| *c = Complex::new(c.norm_sqr(), 0.0));
        ifft.process(input);
    });

    (0..tmax)
        .map(|t| (0..n).map(|i| input[i][t].re).sum::<f64>() / ((n * tmax) as f64))
        .collect()
}
