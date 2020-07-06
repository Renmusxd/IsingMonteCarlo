#![feature(test)]

extern crate ising_monte_carlo;
extern crate test;

#[cfg(test)]
mod tests {
    use super::*;
    use ising_monte_carlo::sse::qmc_traits::*;
    use ising_monte_carlo::sse::qmc_types::Op;
    use ising_monte_carlo::sse::simple_ops::*;
    use smallvec::smallvec;
    use test::Bencher;

    #[bench]
    fn bench_empty_graph(b: &mut Bencher) {
        let nvars = 10;
        let a = SimpleOpDiagonal::new(nvars);
        b.iter(|| {
            let a = a.clone();
            (0..1000).fold(a, |a, _| a.convert_to_looper().convert_to_diagonal());
        });
    }

    #[bench]
    fn bench_many_graph(b: &mut Bencher) {
        let nvars = 10;
        let mut a = SimpleOpDiagonal::new(nvars);
        (0..nvars.pow(2)).for_each(|i| {
            let vara = i % nvars;
            let varb = (i + 1) % nvars;
            a.set_pth(
                i,
                Some(Op {
                    vars: smallvec![vara, varb],
                    bond: 0,
                    inputs: smallvec![false, false],
                    outputs: smallvec![false, false],
                }),
            );
        });
        b.iter(|| {
            let a = a.clone();
            (0..1000).fold(a, |a, _| a.convert_to_looper().convert_to_diagonal());
        });
    }

    #[bench]
    fn bench_many_spaced_graph(b: &mut Bencher) {
        let nvars = 10;
        let mut a = SimpleOpDiagonal::new(nvars);
        (0..nvars.pow(2)).for_each(|i| {
            let vara = i % nvars;
            let varb = (i + 1) % nvars;
            a.set_pth(
                nvars * i,
                Some(Op {
                    vars: smallvec![vara, varb],
                    bond: 0,
                    inputs: smallvec![false, false],
                    outputs: smallvec![false, false],
                }),
            );
        });
        b.iter(|| {
            let a = a.clone();
            (0..1000).fold(a, |a, _| a.convert_to_looper().convert_to_diagonal());
        });
    }
}
