use crate::sse::qmc_traits::op_container::*;
use rand::Rng;

/// An object which provides all required details of the hamiltonian.
pub trait Hamiltonian<'a> {
    /// Maps (vars, bond, inputs, outputs) to a float matrix element.
    fn hamiltonian(&self, vars: &[usize], bond: usize, inputs: &[bool], outputs: &[bool]) -> f64;
    /// Give edges for a bond, and if the bond is a constant.
    fn edge_fn(&self, bond: usize) -> (&'a [usize], bool);
    /// The number of bonds which exist.
    fn num_bonds(&self) -> usize;
}

/// Perform diagonal updates to an op container.
pub trait DiagonalUpdater: OpContainer {
    /// Folds across the p values, passing T down. Mutates op if returned values is Some(...)
    fn mutate_ps<F, T>(&mut self, pstart: usize, pend: usize, t: T, f: F) -> T
    where
        F: Fn(&Self, Option<&Self::Op>, T) -> (Option<Option<Self::Op>>, T);

    /// Mutate only the ops. Override with more efficient solutions if needed.
    fn mutate_ops<F, T>(&mut self, pstart: usize, pend: usize, t: T, f: F) -> T
    where
        F: Fn(&Self, &Self::Op, usize, T) -> (Option<Option<Self::Op>>, T),
    {
        let (_, t) = self.mutate_ps(pstart, pend, (0, t), |s, op, (p, t)| {
            let (op, t) = if let Some(op) = op {
                f(s, op, p, t)
            } else {
                (None, t)
            };
            (op, (p + 1, t))
        });
        t
    }

    /// Iterate through the ops and call f. Exit early with Err(v).
    fn try_iterate_ps<F, T, V>(&self, pstart: usize, pend: usize, t: T, f: F) -> Result<T, V>
    where
        F: Fn(&Self, Option<&Self::Op>, T) -> Result<T, V>;

    /// Iterate through ops only. Can override with more efficient implementation.
    fn try_iterate_ops<F, T, V>(&self, pstart: usize, pend: usize, t: T, f: F) -> Result<T, V>
    where
        // self, op, p, accumulator
        F: Fn(&Self, &Self::Op, usize, T) -> Result<T, V>,
    {
        self.try_iterate_ps(pstart, pend, (0, t), |s, op, (p, t)| {
            let t = if let Some(op) = op {
                f(s, op, p, t)?
            } else {
                t
            };
            Ok((p + 1, t))
        })
        .map(|(_, t)| t)
    }

    /// Iterate through the ops and call f.
    fn iterate_ps<F, T>(&self, pstart: usize, pend: usize, t: T, f: F) -> T
    where
        F: Fn(&Self, Option<&Self::Op>, T) -> T,
    {
        self.try_iterate_ps(pstart, pend, t, |s, op, t| -> Result<T, ()> {
            Ok(f(s, op, t))
        })
        .unwrap()
    }

    /// Iterate through ops only.
    /// Calls try_iterate_ops by default.
    fn iterate_ops<F, T>(&self, pstart: usize, pend: usize, t: T, f: F) -> T
    where
        // self, op, p, accumulator
        F: Fn(&Self, &Self::Op, usize, T) -> T,
    {
        self.try_iterate_ops(pstart, pend, t, |s, op, p, t| -> Result<T, ()> {
            Ok(f(s, op, p, t))
        })
        .unwrap()
    }

    /// Perform a diagonal update step with thread rng.
    fn make_diagonal_update<'b, H: Hamiltonian<'b>>(
        &mut self,
        cutoff: usize,
        beta: f64,
        state: &[bool],
        hamiltonian: &H,
    ) {
        self.make_diagonal_update_with_rng(
            cutoff,
            beta,
            state,
            hamiltonian,
            &mut rand::thread_rng(),
        )
    }

    /// Perform a diagonal update step.
    fn make_diagonal_update_with_rng<'b, H: Hamiltonian<'b>, R: Rng>(
        &mut self,
        cutoff: usize,
        beta: f64,
        state: &[bool],
        hamiltonian: &H,
        rng: &mut R,
    ) {
        let mut state = state.to_vec();
        self.make_diagonal_update_with_rng_and_state_ref(cutoff, beta, &mut state, hamiltonian, rng)
    }

    /// Perform a diagonal update step using in place edits to state.
    fn make_diagonal_update_with_rng_and_state_ref<'b, H: Hamiltonian<'b>, R: Rng>(
        &mut self,
        cutoff: usize,
        beta: f64,
        state: &mut [bool],
        hamiltonian: &H,
        rng: &mut R,
    ) {
        self.mutate_ps(0, cutoff, (state, rng), |s, op, (state, rng)| {
            let op = metropolis_single_diagonal_update(
                op,
                cutoff,
                s.get_n(),
                beta,
                state,
                hamiltonian,
                rng,
            );
            (op, (state, rng))
        });
        self.post_diagonal_update_hook();
    }

    /// Called after an update.
    fn post_diagonal_update_hook(&mut self) {}
}

/// Perform a single metropolis update.
fn metropolis_single_diagonal_update<'b, O: Op, H: Hamiltonian<'b>, R: Rng>(
    op: Option<&O>,
    cutoff: usize,
    n: usize,
    beta: f64,
    state: &mut [bool],
    hamiltonian: &H,
    rng: &mut R,
) -> Option<Option<O>> {
    let b = match op {
        None => rng.gen_range(0, hamiltonian.num_bonds()),
        Some(op) if op.is_diagonal() => op.get_bond(),
        Some(op) => {
            op.get_vars()
                .iter()
                .zip(op.get_outputs().iter())
                .for_each(|(v, b)| state[*v] = *b);
            return None;
        }
    };
    let (vars, constant) = hamiltonian.edge_fn(b);
    let substate = vars.iter().map(|v| state[*v]).collect::<O::SubState>();
    let mat_element = hamiltonian.hamiltonian(vars, b, substate.as_ref(), substate.as_ref());

    // This is based on equations 19a and 19b of arXiv:1909.10591v1 from 23 Sep 2019
    // or A. W. Sandvik, Phys. Rev. B 59, 14157 (1999)
    let numerator = beta * (hamiltonian.num_bonds() as f64) * mat_element;
    let denominator = (cutoff - n) as f64;

    match op {
        None => {
            if numerator > denominator || rng.gen_bool(numerator / denominator) {
                let vars = vars.iter().cloned().collect::<O::Vars>();
                let op = Op::diagonal(vars, b, substate, constant);
                Some(Some(op))
            } else {
                None
            }
        }
        Some(op) if op.is_diagonal() => {
            let denominator = denominator + 1.0;
            if denominator > numerator || rng.gen_bool(denominator / numerator) {
                Some(None)
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Print a diagonal updater using a state.
pub fn debug_print_diagonal<D: DiagonalUpdater + ?Sized>(diagonal: &D, state: &[bool]) {
    let nvars = diagonal.get_nvars();
    for _ in 0..nvars {
        print!("=");
    }
    println!();
    for b in state {
        print!("{}", if *b { "1" } else { "0" });
    }
    println!();

    diagonal.iterate_ps(0, diagonal.get_cutoff(), 0, |_, op, p| {
        if let Some(op) = op {
            let mut last_var = 0;
            let mut v_b: Vec<(usize, bool)> = op
                .get_vars()
                .iter()
                .cloned()
                .zip(op.get_outputs().iter().cloned())
                .collect();
            v_b.sort_unstable();
            for (var, outp) in v_b.into_iter() {
                for _ in last_var..var {
                    print!("|");
                }
                print!("{}", if outp { 1 } else { 0 });
                last_var = var + 1;
            }
            for _ in last_var..nvars {
                print!("|");
            }
            println!("\tp={}\t{}: {:?}", p, op.get_bond(), op.get_vars());
        } else {
            for _ in 0..nvars {
                print!("|");
            }
            println!("\tp={}", p);
        }
        p + 1
    });
}
