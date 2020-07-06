#![feature(test)]

extern crate ising_monte_carlo;
extern crate test;

use ising_monte_carlo::graph::Edge;

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
    use ising_monte_carlo::sse::fast_ops::*;
    use ising_monte_carlo::sse::qmc_graph::QMCGraph;
    use ising_monte_carlo::sse::simple_ops::*;
    use rand::prelude::*;
    use rand::rngs::SmallRng;
    use rand::SeedableRng;
    use test::Bencher;

    #[bench]
    fn one_d_no_loopupdate(b: &mut Bencher) {
        let l = 16;
        let rng = rand::thread_rng();
        let mut g =
            QMCGraph::<ThreadRng, SimpleOpNode, SimpleOpDiagonal, SimpleOpLooper>::new_with_rng(
                one_d_periodic(l),
                1.0,
                l,
                false,
                false,
                rng,
                None,
            );
        let beta = 1.0;
        b.iter(|| g.timesteps(1000, beta));
    }

    #[bench]
    fn two_d_no_loopupdate(b: &mut Bencher) {
        let l = 4;
        let rng = rand::thread_rng();
        let mut g =
            QMCGraph::<ThreadRng, SimpleOpNode, SimpleOpDiagonal, SimpleOpLooper>::new_with_rng(
                two_d_periodic(l),
                1.0,
                l,
                false,
                false,
                rng,
                None,
            );

        let beta = 1.0;
        b.iter(|| g.timesteps(1000, beta));
    }

    #[bench]
    fn one_d_no_loopupdate_new(b: &mut Bencher) {
        let l = 16;
        let rng = rand::thread_rng();
        let mut g = QMCGraph::<ThreadRng, FastOpNode, FastOps, FastOps>::new_with_rng(
            one_d_periodic(l),
            1.0,
            l,
            false,
            false,
            rng,
            None,
        );
        let beta = 1.0;
        b.iter(|| g.timesteps(1000, beta));
    }

    #[bench]
    fn two_d_no_loopupdate_new(b: &mut Bencher) {
        let l = 4;
        let rng = rand::thread_rng();
        let mut g = QMCGraph::<ThreadRng, FastOpNode, FastOps, FastOps>::new_with_rng(
            two_d_periodic(l),
            1.0,
            l,
            false,
            false,
            rng,
            None,
        );

        let beta = 1.0;
        b.iter(|| g.timesteps(1000, beta));
    }

    #[bench]
    fn two_d_no_loopupdate_large(b: &mut Bencher) {
        let l = 16;
        let rng = rand::thread_rng();
        let mut g =
            QMCGraph::<ThreadRng, SimpleOpNode, SimpleOpDiagonal, SimpleOpLooper>::new_with_rng(
                two_d_periodic(l),
                1.0,
                l,
                false,
                false,
                rng,
                None,
            );

        let beta = 1.0;
        b.iter(|| g.timesteps(100, beta));
    }

    #[bench]
    fn two_d_no_loopupdate_large_new(b: &mut Bencher) {
        let l = 16;
        let rng = rand::thread_rng();
        let mut g = QMCGraph::<ThreadRng, FastOpNode, FastOps, FastOps>::new_with_rng(
            two_d_periodic(l),
            1.0,
            l,
            false,
            false,
            rng,
            None,
        );

        let beta = 1.0;
        b.iter(|| g.timesteps(100, beta));
    }

    #[bench]
    fn two_d_no_loopupdate_large_smallrng(b: &mut Bencher) {
        let l = 16;
        let rng = SmallRng::from_entropy();
        let mut g =
            QMCGraph::<SmallRng, SimpleOpNode, SimpleOpDiagonal, SimpleOpLooper>::new_with_rng(
                two_d_periodic(l),
                1.0,
                l,
                false,
                false,
                rng,
                None,
            );

        let beta = 1.0;
        b.iter(|| g.timesteps(100, beta));
    }

    #[bench]
    fn two_d_no_loopupdate_large_new_smallrng(b: &mut Bencher) {
        let l = 16;
        let rng = SmallRng::from_entropy();
        let mut g = QMCGraph::<SmallRng, FastOpNode, FastOps, FastOps>::new_with_rng(
            two_d_periodic(l),
            1.0,
            l,
            false,
            false,
            rng,
            None,
        );

        let beta = 1.0;
        b.iter(|| g.timesteps(100, beta));
    }
}
