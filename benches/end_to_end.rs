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
    use qmc::sse::qmc_ising::QmcIsingGraph;
    use qmc::sse::*;
    use rand::rngs::SmallRng;
    use rand::SeedableRng;
    use test::Bencher;

    #[bench]
    fn one_d(b: &mut Bencher) {
        let l = 16;
        let rng = SmallRng::seed_from_u64(1234);
        let mut g = QmcIsingGraph::<SmallRng, FastOps>::new_with_rng(
            one_d_periodic(l),
            1.,
            0.,
            l,
            rng,
            None,
        );
        let beta = 1.0;
        g.timesteps(1000, beta);
        b.iter(|| g.timesteps(1, beta));
    }

    #[bench]
    fn two_d_04(b: &mut Bencher) {
        let l = 4;
        let rng = SmallRng::seed_from_u64(1234);
        let mut g = QmcIsingGraph::<SmallRng, FastOps>::new_with_rng(
            two_d_periodic(l),
            1.,
            0.,
            l,
            rng,
            None,
        );

        let beta = 1.0;
        g.timesteps(1000, beta);
        b.iter(|| g.timesteps(1, beta));
    }

    #[bench]
    fn two_d_08(b: &mut Bencher) {
        let l = 8;
        let rng = SmallRng::seed_from_u64(1234);
        let mut g = QmcIsingGraph::<SmallRng, FastOps>::new_with_rng(
            two_d_periodic(l),
            1.,
            0.,
            l,
            rng,
            None,
        );

        let beta = 1.0;
        g.timesteps(1000, beta);
        b.iter(|| g.timesteps(1, beta));
    }

    #[bench]
    fn two_d_16(b: &mut Bencher) {
        let l = 16;
        let rng = SmallRng::seed_from_u64(1234);
        let mut g = QmcIsingGraph::<SmallRng, FastOps>::new_with_rng(
            two_d_periodic(l),
            1.,
            0.,
            l,
            rng,
            None,
        );

        let beta = 1.0;
        g.timesteps(1000, beta);
        b.iter(|| g.timesteps(1, beta));
    }

    #[bench]
    fn two_d_32(b: &mut Bencher) {
        let l = 32;
        let rng = SmallRng::seed_from_u64(1234);
        let mut g = QmcIsingGraph::<SmallRng, FastOps>::new_with_rng(
            two_d_periodic(l),
            1.,
            0.,
            l,
            rng,
            None,
        );

        let beta = 1.0;
        g.timesteps(1000, beta);
        b.iter(|| g.timesteps(1, beta));
    }

    #[bench]
    fn one_d_heatbath(b: &mut Bencher) {
        let l = 16;
        let rng = SmallRng::seed_from_u64(1234);
        let mut g = QmcIsingGraph::<SmallRng, FastOps>::new_with_rng(
            one_d_periodic(l),
            1.,
            0.,
            l,
            rng,
            None,
        );
        g.set_enable_heatbath(true);

        let beta = 1.0;
        g.timesteps(1000, beta);
        b.iter(|| g.timesteps(1, beta));
    }

    #[bench]
    fn two_d_heatbath_04(b: &mut Bencher) {
        let l = 4;
        let rng = SmallRng::seed_from_u64(1234);
        let mut g = QmcIsingGraph::<SmallRng, FastOps>::new_with_rng(
            two_d_periodic(l),
            1.,
            0.,
            l,
            rng,
            None,
        );
        g.set_enable_heatbath(true);

        let beta = 1.0;
        g.timesteps(1000, beta);
        b.iter(|| g.timesteps(1, beta));
    }

    #[bench]
    fn two_d_heatbath_08(b: &mut Bencher) {
        let l = 8;
        let rng = SmallRng::seed_from_u64(1234);
        let mut g = QmcIsingGraph::<SmallRng, FastOps>::new_with_rng(
            two_d_periodic(l),
            1.,
            0.,
            l,
            rng,
            None,
        );
        g.set_enable_heatbath(true);
        let beta = 1.0;
        g.timesteps(1000, beta);
        b.iter(|| g.timesteps(1, beta));
    }

    #[bench]
    fn two_d_heatbath_16(b: &mut Bencher) {
        let l = 16;
        let rng = SmallRng::seed_from_u64(1234);
        let mut g = QmcIsingGraph::<SmallRng, FastOps>::new_with_rng(
            two_d_periodic(l),
            1.,
            0.,
            l,
            rng,
            None,
        );
        g.set_enable_heatbath(true);
        let beta = 1.0;
        g.timesteps(1000, beta);
        b.iter(|| g.timesteps(1, beta));
    }

    #[bench]
    fn two_d_heatbath_32(b: &mut Bencher) {
        let l = 32;
        let rng = SmallRng::seed_from_u64(1234);
        let mut g = QmcIsingGraph::<SmallRng, FastOps>::new_with_rng(
            two_d_periodic(l),
            1.,
            0.,
            l,
            rng,
            None,
        );
        g.set_enable_heatbath(true);
        let beta = 1.0;
        g.timesteps(1000, beta);
        b.iter(|| g.timesteps(1, beta));
    }

    #[bench]
    fn two_d_rvb_04(b: &mut Bencher) {
        let l = 4;
        let rng = SmallRng::seed_from_u64(1234);
        let mut g = QmcIsingGraph::<SmallRng, FastOps>::new_with_rng(
            two_d_periodic(l),
            1.,
            0.,
            l,
            rng,
            None,
        );
        g.set_run_rvb(true);

        let beta = 10.0;
        g.timesteps(1000, beta);
        b.iter(|| g.timesteps(1, beta));
    }

    #[bench]
    fn two_d_rvb_08(b: &mut Bencher) {
        let l = 8;
        let rng = SmallRng::seed_from_u64(1234);
        let mut g = QmcIsingGraph::<SmallRng, FastOps>::new_with_rng(
            two_d_periodic(l),
            1.,
            0.,
            l,
            rng,
            None,
        );
        g.set_run_rvb(true);

        let beta = 10.0;
        g.timesteps(1000, beta);
        b.iter(|| g.timesteps(1, beta));
    }

    #[bench]
    fn two_d_rvb_16(b: &mut Bencher) {
        let l = 16;
        let rng = SmallRng::seed_from_u64(1234);
        let mut g = QmcIsingGraph::<SmallRng, FastOps>::new_with_rng(
            two_d_periodic(l),
            1.,
            0.,
            l,
            rng,
            None,
        );
        g.set_run_rvb(true);

        let beta = 10.0;
        g.timesteps(100, beta);
        b.iter(|| g.timesteps(1, beta));
    }

    #[bench]
    fn two_d_rvb_32(b: &mut Bencher) {
        let l = 32;
        let rng = SmallRng::seed_from_u64(1234);
        let mut g = QmcIsingGraph::<SmallRng, FastOps>::new_with_rng(
            two_d_periodic(l),
            1.,
            0.,
            l,
            rng,
            None,
        );
        g.set_run_rvb(true);

        let beta = 10.0;
        g.timesteps(100, beta);
        b.iter(|| g.timesteps(1, beta));
    }

    #[bench]
    fn two_d_rvb_cold_4(b: &mut Bencher) {
        let l = 4;
        let rng = SmallRng::seed_from_u64(1234);
        let mut g = QmcIsingGraph::<SmallRng, FastOps>::new_with_rng(
            two_d_periodic(l),
            1.,
            0.,
            l,
            rng,
            None,
        );
        g.set_run_rvb(true);

        let beta = 100.0;
        g.timesteps(1000, beta);
        b.iter(|| g.timesteps(1, beta));
    }

    #[bench]
    fn two_d_rvb_cold_5(b: &mut Bencher) {
        let l = 5;
        let rng = SmallRng::seed_from_u64(1234);
        let mut g = QmcIsingGraph::<SmallRng, FastOps>::new_with_rng(
            two_d_periodic(l),
            1.,
            0.,
            l,
            rng,
            None,
        );
        g.set_run_rvb(true);

        let beta = 100.0;
        g.timesteps(1000, beta);
        b.iter(|| g.timesteps(1, beta));
    }

    #[bench]
    fn two_d_rvb_cold_6(b: &mut Bencher) {
        let l = 6;
        let rng = SmallRng::seed_from_u64(1234);
        let mut g = QmcIsingGraph::<SmallRng, FastOps>::new_with_rng(
            two_d_periodic(l),
            1.,
            0.,
            l,
            rng,
            None,
        );
        g.set_run_rvb(true);

        let beta = 100.0;
        g.timesteps(1000, beta);
        b.iter(|| g.timesteps(1, beta));
    }

    #[bench]
    fn two_d_rvb_cold_7(b: &mut Bencher) {
        let l = 7;
        let rng = SmallRng::seed_from_u64(1234);
        let mut g = QmcIsingGraph::<SmallRng, FastOps>::new_with_rng(
            two_d_periodic(l),
            1.,
            0.,
            l,
            rng,
            None,
        );
        g.set_run_rvb(true);

        let beta = 100.0;
        g.timesteps(1000, beta);
        b.iter(|| g.timesteps(1, beta));
    }

    #[bench]
    fn two_d_rvb_cold_8(b: &mut Bencher) {
        let l = 8;
        let rng = SmallRng::seed_from_u64(1234);
        let mut g = QmcIsingGraph::<SmallRng, FastOps>::new_with_rng(
            two_d_periodic(l),
            1.,
            0.,
            l,
            rng,
            None,
        );
        g.set_run_rvb(true);

        let beta = 100.0;
        g.timesteps(1000, beta);
        b.iter(|| g.timesteps(1, beta));
    }
}
