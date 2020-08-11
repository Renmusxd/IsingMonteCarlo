extern crate ising_monte_carlo;
extern crate rand;
use ising_monte_carlo::sse::qmc_traits::*;
use ising_monte_carlo::sse::simple_ops::*;
use smallvec::smallvec;

#[test]
fn single_cluster_test() {
    let mut manager = SimpleOpDiagonal::new(1);
    manager.set_pth(
        0,
        Some(SimpleOp::offdiagonal(
            smallvec![0],
            0,
            smallvec![false],
            smallvec![false],
            true,
        )),
    );
    let mut manager: SimpleOpLooper = manager.into();

    let mut rng = rand::thread_rng();
    let mut state = vec![false; manager.get_nvars()];
    manager.flip_each_cluster_rng(0.5, &mut rng, &mut state);
    println!("{:?}", state);
}

#[test]
fn simple_cluster_test() {
    let mut manager = SimpleOpDiagonal::new(1);
    manager.set_pth(
        0,
        Some(SimpleOp::offdiagonal(
            smallvec![0],
            0,
            smallvec![false],
            smallvec![false],
            true,
        )),
    );
    manager.set_pth(
        1,
        Some(SimpleOp::offdiagonal(
            smallvec![0],
            1,
            smallvec![false],
            smallvec![false],
            true,
        )),
    );
    let mut manager: SimpleOpLooper = manager.into();

    let mut rng = rand::thread_rng();
    let mut state = vec![false; manager.get_nvars()];
    manager.flip_each_cluster_rng(0.5, &mut rng, &mut state);
    println!("{:?}", state);
}

#[test]
fn multi_cluster_test() {
    let mut manager = SimpleOpDiagonal::new(2);
    manager.set_pth(
        0,
        Some(SimpleOp::offdiagonal(
            smallvec![0],
            0,
            smallvec![false],
            smallvec![false],
            true,
        )),
    );
    manager.set_pth(
        1,
        Some(SimpleOp::offdiagonal(
            smallvec![0],
            1,
            smallvec![false],
            smallvec![false],
            true,
        )),
    );

    manager.set_pth(
        2,
        Some(SimpleOp::offdiagonal(
            smallvec![1],
            2,
            smallvec![false],
            smallvec![false],
            true,
        )),
    );
    manager.set_pth(
        3,
        Some(SimpleOp::offdiagonal(
            smallvec![1],
            3,
            smallvec![false],
            smallvec![false],
            true,
        )),
    );

    let mut manager: SimpleOpLooper = manager.into();

    let mut rng = rand::thread_rng();
    let mut state = vec![false; manager.get_nvars()];
    manager.flip_each_cluster_rng(0.5, &mut rng, &mut state);
    println!("{:?}", state);
}

#[test]
fn multi_twosite_cluster_test() {
    let mut manager = SimpleOpDiagonal::new(4);
    manager.set_pth(
        0,
        Some(SimpleOp::offdiagonal(
            smallvec![0],
            0,
            smallvec![false],
            smallvec![false],
            true,
        )),
    );
    manager.set_pth(
        1,
        Some(SimpleOp::offdiagonal(
            smallvec![0],
            1,
            smallvec![false],
            smallvec![false],
            true,
        )),
    );

    manager.set_pth(
        2,
        Some(SimpleOp::offdiagonal(
            smallvec![1, 2],
            2,
            smallvec![false, false],
            smallvec![false, false],
            false,
        )),
    );
    manager.set_pth(
        3,
        Some(SimpleOp::offdiagonal(
            smallvec![2, 3],
            3,
            smallvec![false, false],
            smallvec![false, false],
            false,
        )),
    );

    let mut manager: SimpleOpLooper = manager.into();

    let mut rng = rand::thread_rng();
    let mut state = vec![false; manager.get_nvars()];
    manager.flip_each_cluster_rng(0.5, &mut rng, &mut state);
    println!("{:?}", state);
}

#[test]
fn multi_multisite_cluster_test() {
    let mut manager = SimpleOpDiagonal::new(3);
    manager.set_pth(
        0,
        Some(SimpleOp::offdiagonal(
            smallvec![0],
            0,
            smallvec![false],
            smallvec![false],
            true,
        )),
    );
    manager.set_pth(
        1,
        Some(SimpleOp::offdiagonal(
            smallvec![1, 2],
            1,
            smallvec![false, false],
            smallvec![false, false],
            false,
        )),
    );
    manager.set_pth(
        2,
        Some(SimpleOp::offdiagonal(
            smallvec![1],
            2,
            smallvec![false],
            smallvec![false],
            true,
        )),
    );

    let mut manager: SimpleOpLooper = manager.into();

    let mut rng = rand::thread_rng();
    let mut state = vec![false; manager.get_nvars()];
    manager.flip_each_cluster_rng(0.5, &mut rng, &mut state);
    println!("{:?}", state);
}
