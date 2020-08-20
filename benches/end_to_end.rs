#![feature(test)]

extern crate qmc;
extern crate test;

use qmc::classical::graph::Edge;

fn one_d_periodic(l: usize) -> Vec<(Edge, f64)> {
    (0..l).map(|i| ((i, (i + 1) % l), 1.0)).collect()
}

fn two_d_periodic(l: usize) -> Vec<(Edge, f64)> {
    let indices: Vec<(usize, usize)> = (0usize..l)
        .map(|i| (0usize..l).map(|j| (i, j)).collect::<Vec<(usize, usize)>>())
        .flatten()
        .collect();
    let f = |i, j| j * l + i;

    let right_connects = indices
        .iter()
        .cloned()
        .map(|(i, j)| ((f(i, j), f((i + 1) % l, j)), -1.0));
    let down_connects = indices.iter().cloned().map(|(i, j)| {
        (
            (f(i, j), f(i, (j + 1) % l)),
            if i % 2 == 0 { 1.0 } else { -1.0 },
        )
    });
    right_connects.chain(down_connects).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use qmc::sse::fast_ops::*;
    use qmc::sse::qmc_ising::QMCIsingGraph;
    use qmc::sse::simple_ops::*;
    use qmc::sse::*;
    use rand::rngs::SmallRng;
    use rand::SeedableRng;
    use test::Bencher;

    #[bench]
    fn one_d(b: &mut Bencher) {
        let l = 16;
        let rng = SmallRng::seed_from_u64(1234);
        let mut g = QMCIsingGraph::<SmallRng, SimpleOpDiagonal, SimpleOpLooper>::new_with_rng(
            one_d_periodic(l),
            1.0,
            l,
            rng,
            None,
        );
        let beta = 1.0;
        g.timesteps(1000, beta);
        b.iter(|| g.timesteps(1000, beta));
    }

    #[bench]
    fn two_d(b: &mut Bencher) {
        let l = 4;
        let rng = SmallRng::seed_from_u64(1234);
        let mut g = QMCIsingGraph::<SmallRng, SimpleOpDiagonal, SimpleOpLooper>::new_with_rng(
            two_d_periodic(l),
            1.0,
            l,
            rng,
            None,
        );

        let beta = 1.0;
        g.timesteps(1000, beta);
        b.iter(|| g.timesteps(1000, beta));
    }

    #[bench]
    fn one_d_new(b: &mut Bencher) {
        let l = 16;
        let rng = SmallRng::seed_from_u64(1234);
        let mut g = QMCIsingGraph::<SmallRng, FastOps, FastOps>::new_with_rng(
            one_d_periodic(l),
            1.0,
            l,
            rng,
            None,
        );
        let beta = 1.0;
        g.timesteps(1000, beta);
        b.iter(|| g.timesteps(1000, beta));
    }

    #[bench]
    fn two_d_new(b: &mut Bencher) {
        let l = 4;
        let rng = SmallRng::seed_from_u64(1234);
        let mut g = QMCIsingGraph::<SmallRng, FastOps, FastOps>::new_with_rng(
            two_d_periodic(l),
            1.0,
            l,
            rng,
            None,
        );

        let beta = 1.0;
        g.timesteps(1000, beta);
        b.iter(|| g.timesteps(1000, beta));
    }

    #[bench]
    fn two_d_large(b: &mut Bencher) {
        let l = 16;
        let rng = SmallRng::seed_from_u64(1234);
        let mut g = QMCIsingGraph::<SmallRng, SimpleOpDiagonal, SimpleOpLooper>::new_with_rng(
            two_d_periodic(l),
            1.0,
            l,
            rng,
            None,
        );

        let beta = 1.0;
        g.timesteps(1000, beta);
        b.iter(|| g.timesteps(100, beta));
    }

    #[bench]
    fn two_d_large_new(b: &mut Bencher) {
        let l = 16;
        let rng = SmallRng::seed_from_u64(1234);
        let mut g = QMCIsingGraph::<SmallRng, FastOps, FastOps>::new_with_rng(
            two_d_periodic(l),
            1.0,
            l,
            rng,
            None,
        );

        let beta = 1.0;
        g.timesteps(1000, beta);
        b.iter(|| g.timesteps(100, beta));
    }
}
