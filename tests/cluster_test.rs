extern crate rand;
use qmc::sse::fast_ops::{FastOp, FastOps};
use qmc::sse::qmc_traits::*;
use smallvec::smallvec;

#[test]
fn single_cluster_test() {
    let mut manager = FastOps::new_from_ops(
        1,
        vec![(
            0,
            FastOp::<2>::offdiagonal(smallvec![0], 0, smallvec![false], smallvec![false], true),
        )]
        .into_iter(),
    );

    let mut rng = rand::thread_rng();
    let mut state = vec![false; manager.get_nvars()];
    manager.flip_each_cluster_ising_symmetry_rng(0.5, &mut rng, &mut state);
    println!("{:?}", state);
}

#[test]
fn simple_cluster_test() {
    let mut manager = FastOps::new_from_ops(
        1,
        vec![
            (
                0,
                FastOp::<2>::offdiagonal(smallvec![0], 0, smallvec![false], smallvec![false], true),
            ),
            (
                1,
                FastOp::<2>::offdiagonal(smallvec![0], 1, smallvec![false], smallvec![false], true),
            ),
        ]
        .into_iter(),
    );

    let mut rng = rand::thread_rng();
    let mut state = vec![false; manager.get_nvars()];
    manager.flip_each_cluster_ising_symmetry_rng(0.5, &mut rng, &mut state);
    println!("{:?}", state);
}

#[test]
fn multi_cluster_test() {
    let mut manager = FastOps::new_from_ops(
        2,
        vec![
            (
                0,
                FastOp::<2>::offdiagonal(smallvec![0], 0, smallvec![false], smallvec![false], true),
            ),
            (
                1,
                FastOp::<2>::offdiagonal(smallvec![0], 1, smallvec![false], smallvec![false], true),
            ),
            (
                2,
                FastOp::<2>::offdiagonal(smallvec![1], 2, smallvec![false], smallvec![false], true),
            ),
            (
                3,
                FastOp::<2>::offdiagonal(smallvec![1], 3, smallvec![false], smallvec![false], true),
            ),
        ]
        .into_iter(),
    );

    let mut rng = rand::thread_rng();
    let mut state = vec![false; manager.get_nvars()];
    manager.flip_each_cluster_ising_symmetry_rng(0.5, &mut rng, &mut state);
    println!("{:?}", state);
}
