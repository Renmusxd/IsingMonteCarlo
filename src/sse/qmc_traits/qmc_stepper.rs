/// Provides helpers to structs which take QMC timesteps.
pub trait QmcStepper {
    /// Take a single QMC step and return a reference to the state
    fn timestep(&mut self, beta: f64) -> &[bool];
    /// Get the current number of operators in the graph
    fn get_n(&self) -> usize;
    /// Get the average energy given the average number of ops and beta.
    fn get_energy_for_average_n(&self, average_n: f64, beta: f64) -> f64;

    /// Get a reference to the state.
    fn state_ref(&self) -> &[bool];

    /// Get the number of a given bond
    fn get_bond_count(&self, bond: usize) -> usize;

    /// Take t qmc timesteps at beta.
    fn timesteps(&mut self, t: usize, beta: f64) -> f64 {
        let (_, average_energy) = self.timesteps_measure(t, beta, (), |_acc, _state| (), None);
        average_energy
    }

    /// Take t qmc timesteps at beta and sample states.
    fn timesteps_sample(
        &mut self,
        t: usize,
        beta: f64,
        sampling_freq: Option<usize>,
    ) -> (Vec<Vec<bool>>, f64) {
        let acc = Vec::with_capacity(t / sampling_freq.unwrap_or(1) + 1);
        self.timesteps_measure(
            t,
            beta,
            acc,
            |mut acc, state| {
                acc.push(state.to_vec());
                acc
            },
            sampling_freq,
        )
    }

    /// Take t qmc timesteps at beta and sample states, apply f to each.
    fn timesteps_sample_iter<F>(
        &mut self,
        t: usize,
        beta: f64,
        sampling_freq: Option<usize>,
        iter_fn: F,
    ) -> f64
    where
        F: Fn(&[bool]),
    {
        let (_, e) = self.timesteps_measure(t, beta, (), |_, state| iter_fn(state), sampling_freq);
        e
    }

    /// Take t qmc timesteps at beta and sample states, apply f to each and the zipped iterator.
    fn timesteps_sample_iter_zip<F, I, T>(
        &mut self,
        t: usize,
        beta: f64,
        sampling_freq: Option<usize>,
        zip_with: I,
        iter_fn: F,
    ) -> f64
    where
        F: Fn(T, &[bool]),
        I: IntoIterator<Item = T>,
    {
        self.timesteps_iter_zip_with_self(t, beta, sampling_freq, zip_with, |acc, s| {
            iter_fn(acc, s.state_ref())
        })
    }

    /// Take t qmc timesteps at beta and sample states, fold across states and output results.
    fn timesteps_measure<F, T>(
        &mut self,
        timesteps: usize,
        beta: f64,
        init_t: T,
        state_fold: F,
        sampling_freq: Option<usize>,
    ) -> (T, f64)
    where
        F: Fn(T, &[bool]) -> T,
    {
        self.timesteps_measure_with_self(
            timesteps,
            beta,
            init_t,
            |acc, s| state_fold(acc, s.state_ref()),
            sampling_freq,
        )
    }

    /// Take t qmc timesteps at beta and sample states, apply f to each and the zipped iterator.
    fn timesteps_iter_zip_with_self<F, I, T>(
        &mut self,
        t: usize,
        beta: f64,
        sampling_freq: Option<usize>,
        zip_with: I,
        iter_fn: F,
    ) -> f64
    where
        F: Fn(T, &Self),
        I: IntoIterator<Item = T>,
    {
        let zip_with = zip_with.into_iter();
        let (_, e) = self.timesteps_measure_with_self(
            t,
            beta,
            Some(zip_with),
            |zip_iter, s| {
                if let Some(mut zip_iter) = zip_iter {
                    let next = zip_iter.next();
                    if let Some(next) = next {
                        iter_fn(next, s);
                        Some(zip_iter)
                    } else {
                        None
                    }
                } else {
                    None
                }
            },
            sampling_freq,
        );
        e
    }

    /// Take t qmc timesteps at beta and sample states, fold across states and output results.
    fn timesteps_measure_with_self<F, T>(
        &mut self,
        timesteps: usize,
        beta: f64,
        init_t: T,
        state_fold: F,
        sampling_freq: Option<usize>,
    ) -> (T, f64)
    where
        F: Fn(T, &Self) -> T,
    {
        let mut acc = init_t;
        let mut steps_measured = 0;
        let mut total_n = 0;
        let sampling_freq = sampling_freq.unwrap_or(1);

        for t in 0..timesteps {
            self.timestep(beta);

            // Sample every `sampling_freq`
            // Ignore first one.
            if (t + 1) % sampling_freq == 0 {
                acc = state_fold(acc, self);
                steps_measured += 1;
                total_n += self.get_n();
            }
        }
        let average_n = total_n as f64 / steps_measured as f64;
        (acc, self.get_energy_for_average_n(average_n, beta))
    }

    /// Fold through imaginary time states.
    fn imaginary_time_fold<F, T>(&self, fold_fn: F, init: T) -> T
    where
        F: Fn(T, &[bool]) -> T;
}
