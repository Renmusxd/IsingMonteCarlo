use crate::sse::QMCStepper;
use rayon::prelude::*;
use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;
use rustfft::FFTplanner;
use std::ops::DivAssign;

/// Calculate autocorrelations for a QMCStepper
pub trait QMCAutoCorrelations: QMCStepper {
    /// Calculate the autcorrelation calculations for the results of f(state).
    fn calculate_autocorrelation<F>(
        &mut self,
        timesteps: usize,
        beta: f64,
        sampling_freq: Option<usize>,
        use_fft: Option<bool>,
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

        if use_fft.unwrap_or(true) {
            fft_autocorrelation(&samples)
        } else {
            naive_autocorrelation(&samples)
        }
    }

    /// Calculate the autocorrelation of variables.
    fn calculate_variable_autocorrelation(
        &mut self,
        timesteps: usize,
        beta: f64,
        sampling_freq: Option<usize>,
        use_fft: Option<bool>,
    ) -> Vec<f64> {
        self.calculate_autocorrelation(timesteps, beta, sampling_freq, use_fft, |_, sample| {
            sample
                .into_iter()
                .map(|b| if b { 1.0 } else { -1.0 })
                .collect()
        })
    }
}

impl<Q: QMCStepper> QMCAutoCorrelations for Q {}

/// Calculate bond autocorrelations for a QMCAutoCorrelations
pub trait QMCBondAutoCorrelations: QMCAutoCorrelations {
    /// Number of bonds
    fn n_bonds(&self) -> usize;

    /// Whether bond is satisfied.
    fn value_for_bond(&self, bond: usize, sample: &[bool]) -> bool;

    /// Calculate the autcorrelation calculations for bonds.
    fn calculate_bond_autocorrelation(
        &mut self,
        timesteps: usize,
        beta: f64,
        sampling_freq: Option<usize>,
        use_fft: Option<bool>,
    ) -> Vec<f64> {
        let nbonds = self.n_bonds();
        self.calculate_autocorrelation(timesteps, beta, sampling_freq, use_fft, |s, sample| {
            (0..nbonds)
                .map(|bond| {
                    if s.value_for_bond(bond, &sample) {
                        1.0
                    } else {
                        -1.0
                    }
                })
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
            let norm = v.iter().map(|v| v.powi(2).re).sum::<f64>().sqrt();
            v.iter_mut().for_each(|c| c.div_assign(norm));
            v
        })
        .collect::<Vec<_>>();
    let mut planner = FFTplanner::new(false);
    let fft = planner.plan_fft(tmax);
    let mut iplanner = FFTplanner::new(true);
    let ifft = iplanner.plan_fft(tmax);

    let mut output = vec![Complex::zero(); tmax];
    input.iter_mut().for_each(|input| {
        fft.process(input, &mut output);
        output
            .iter_mut()
            .for_each(|c| *c = Complex::new(c.norm_sqr(), 0.0));
        ifft.process(&mut output, input);
    });

    (0..tmax)
        .map(|t| (0..n).map(|i| input[i][t].re).sum::<f64>() / ((n * tmax) as f64))
        .collect()
}

pub(crate) fn naive_autocorrelation(samples: &[Vec<f64>]) -> Vec<f64> {
    let tmax = samples.len();
    let n: usize = samples[0].len();
    let mu = (0..n)
        .map(|i| -> f64 {
            let total = samples.iter().map(|sample| sample[i]).sum::<f64>();
            total / samples.len() as f64
        })
        .collect::<Vec<_>>();

    (0..tmax)
        .into_par_iter()
        .map(|tau| {
            (0..tmax)
                .map(|t| (t, (t + tau) % tmax))
                .map(|(ta, tb)| {
                    let sample_a = &samples[ta];
                    let sample_b = &samples[tb];
                    let (d, ma, mb) = sample_a
                        .iter()
                        .enumerate()
                        .zip(sample_b.iter().enumerate())
                        .fold(
                            (0.0, 0.0, 0.0),
                            |(mut dot_acc, mut a_acc, mut b_acc), ((i, a), (j, b))| {
                                let da = a - mu[i];
                                let db = b - mu[j];
                                dot_acc += da * db;
                                a_acc += da.powi(2);
                                b_acc += db.powi(2);
                                (dot_acc, a_acc, b_acc)
                            },
                        );
                    d / (ma * mb).sqrt()
                })
                .sum::<f64>()
                / (tmax as f64)
        })
        .collect::<Vec<_>>()
}
