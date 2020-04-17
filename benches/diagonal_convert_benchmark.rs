#![feature(test)]

extern crate monte_carlo;
extern crate test;

#[cfg(test)]
mod tests {
    use super::*;
    use monte_carlo::sse::qmc_traits::DiagonalUpdater;
    use monte_carlo::sse::qmc_types::Op;
    use monte_carlo::sse::simple_ops::*;
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
                    vars: vec![vara, varb],
                    bond: 0,
                    inputs: vec![false, false],
                    outputs: vec![false, false],
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
                    vars: vec![vara, varb],
                    bond: 0,
                    inputs: vec![false, false],
                    outputs: vec![false, false],
                }),
            );
        });
        b.iter(|| {
            let a = a.clone();
            (0..1000).fold(a, |a, _| a.convert_to_looper().convert_to_diagonal());
        });
    }
}
