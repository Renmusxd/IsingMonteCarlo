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
            FastOp::offdiagonal(smallvec![0], 0, smallvec![false], smallvec![false], true),
        )]
        .into_iter(),
    );

    let mut rng = rand::thread_rng();
    let mut state = vec![false; manager.get_nvars()];
    manager.flip_each_cluster_rng(0.5, &mut rng, &mut state);
    println!("{:?}", state);
}

#[test]
fn simple_cluster_test() {
    let mut manager = FastOps::new_from_ops(
        1,
        vec![
            (
                0,
                FastOp::offdiagonal(smallvec![0], 0, smallvec![false], smallvec![false], true),
            ),
            (
                1,
                FastOp::offdiagonal(smallvec![0], 1, smallvec![false], smallvec![false], true),
            ),
        ]
        .into_iter(),
    );

    let mut rng = rand::thread_rng();
    let mut state = vec![false; manager.get_nvars()];
    manager.flip_each_cluster_rng(0.5, &mut rng, &mut state);
    println!("{:?}", state);
}

#[test]
fn multi_cluster_test() {
    let mut manager = FastOps::new_from_ops(
        2,
        vec![
            (
                0,
                FastOp::offdiagonal(smallvec![0], 0, smallvec![false], smallvec![false], true),
            ),
            (
                1,
                FastOp::offdiagonal(smallvec![0], 1, smallvec![false], smallvec![false], true),
            ),
            (
                2,
                FastOp::offdiagonal(smallvec![1], 2, smallvec![false], smallvec![false], true),
            ),
            (
                3,
                FastOp::offdiagonal(smallvec![1], 3, smallvec![false], smallvec![false], true),
            ),
        ]
        .into_iter(),
    );

    let mut rng = rand::thread_rng();
    let mut state = vec![false; manager.get_nvars()];
    manager.flip_each_cluster_rng(0.5, &mut rng, &mut state);
    println!("{:?}", state);
}

// TODO fix these too.
// #[test]
// fn multi_twosite_cluster_test() {
//     let mut manager = FastOps::new(4);
//     manager.set_pth(
//         0,
//         Some(SimpleOp::offdiagonal(
//             smallvec![0],
//             0,
//             smallvec![false],
//             smallvec![false],
//             true,
//         )),
//     );
//     manager.set_pth(
//         1,
//         Some(SimpleOp::offdiagonal(
//             smallvec![0],
//             1,
//             smallvec![false],
//             smallvec![false],
//             true,
//         )),
//     );
//
//     manager.set_pth(
//         2,
//         Some(SimpleOp::offdiagonal(
//             smallvec![1, 2],
//             2,
//             smallvec![false, false],
//             smallvec![false, false],
//             false,
//         )),
//     );
//     manager.set_pth(
//         3,
//         Some(SimpleOp::offdiagonal(
//             smallvec![2, 3],
//             3,
//             smallvec![false, false],
//             smallvec![false, false],
//             false,
//         )),
//     );
//
//     let mut rng = rand::thread_rng();
//     let mut state = vec![false; manager.get_nvars()];
//     manager.flip_each_cluster_rng(0.5, &mut rng, &mut state);
//     println!("{:?}", state);
// }
//
// #[test]
// fn multi_multisite_cluster_test() {
//     let mut manager = FastOps::new(3);
//     manager.set_pth(
//         0,
//         Some(SimpleOp::offdiagonal(
//             smallvec![0],
//             0,
//             smallvec![false],
//             smallvec![false],
//             true,
//         )),
//     );
//     manager.set_pth(
//         1,
//         Some(SimpleOp::offdiagonal(
//             smallvec![1, 2],
//             1,
//             smallvec![false, false],
//             smallvec![false, false],
//             false,
//         )),
//     );
//     manager.set_pth(
//         2,
//         Some(SimpleOp::offdiagonal(
//             smallvec![1],
//             2,
//             smallvec![false],
//             smallvec![false],
//             true,
//         )),
//     );
//
//     let mut rng = rand::thread_rng();
//     let mut state = vec![false; manager.get_nvars()];
//     manager.flip_each_cluster_rng(0.5, &mut rng, &mut state);
//     println!("{:?}", state);
// }
