#![feature(test)]

extern crate qmc;
extern crate test;

#[cfg(test)]
mod tests {
    use super::*;
    use qmc::sse::qmc_traits::*;
    use qmc::sse::simple_ops::*;
    use smallvec::smallvec;
    use test::Bencher;

    fn convert(s: SimpleOpDiagonal) -> SimpleOpDiagonal {
        let l: SimpleOpLooper = s.into();
        l.into()
    }

    #[bench]
    fn bench_empty_graph(b: &mut Bencher) {
        let nvars = 10;
        let a = SimpleOpDiagonal::new(nvars);
        b.iter(|| {
            let a = a.clone();
            (0..1000).fold(a, |a, _| convert(a));
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
                Some(SimpleOp::offdiagonal(
                    smallvec![vara, varb],
                    0,
                    smallvec![false, false],
                    smallvec![false, false],
                    false,
                )),
            );
        });
        b.iter(|| {
            let a = a.clone();
            (0..1000).fold(a, |a, _| convert(a));
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
                Some(SimpleOp::offdiagonal(
                    smallvec![vara, varb],
                    0,
                    smallvec![false, false],
                    smallvec![false, false],
                    false,
                )),
            );
        });
        b.iter(|| {
            let a = a.clone();
            (0..1000).fold(a, |a, _| convert(a));
        });
    }
}
