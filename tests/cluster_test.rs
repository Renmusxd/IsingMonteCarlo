extern crate ising_monte_carlo;
extern crate rand;
use ising_monte_carlo::sse::qmc_traits::*;
use ising_monte_carlo::sse::qmc_types::Op;
use ising_monte_carlo::sse::simple_ops::*;
use smallvec::smallvec;

#[test]
fn single_cluster_test() {
    let mut manager = SimpleOpDiagonal::new(1);
    manager.set_pth(
        0,
        Some(Op {
            vars: smallvec![0],
            bond: 0,
            inputs: smallvec![false],
            outputs: smallvec![false],
        }),
    );
    let mut manager = manager.convert_to_looper();

    let mut rng = rand::thread_rng();
    let state_updates = manager.flip_each_cluster_rng(0.5, &mut rng);
    println!("{:?}", state_updates);
}

#[test]
fn simple_cluster_test() {
    let mut manager = SimpleOpDiagonal::new(1);
    manager.set_pth(
        0,
        Some(Op {
            vars: smallvec![0],
            bond: 0,
            inputs: smallvec![false],
            outputs: smallvec![false],
        }),
    );
    manager.set_pth(
        1,
        Some(Op {
            vars: smallvec![0],
            bond: 0,
            inputs: smallvec![false],
            outputs: smallvec![false],
        }),
    );
    let mut manager = manager.convert_to_looper();

    let mut rng = rand::thread_rng();
    let state_updates = manager.flip_each_cluster_rng(0.5, &mut rng);
    println!("{:?}", state_updates);
}

#[test]
fn multi_cluster_test() {
    let mut manager = SimpleOpDiagonal::new(2);
    manager.set_pth(
        0,
        Some(Op {
            vars: smallvec![0],
            bond: 0,
            inputs: smallvec![false],
            outputs: smallvec![false],
        }),
    );
    manager.set_pth(
        1,
        Some(Op {
            vars: smallvec![0],
            bond: 0,
            inputs: smallvec![false],
            outputs: smallvec![false],
        }),
    );

    manager.set_pth(
        2,
        Some(Op {
            vars: smallvec![1],
            bond: 0,
            inputs: smallvec![false],
            outputs: smallvec![false],
        }),
    );
    manager.set_pth(
        3,
        Some(Op {
            vars: smallvec![1],
            bond: 0,
            inputs: smallvec![false],
            outputs: smallvec![false],
        }),
    );

    let mut manager = manager.convert_to_looper();

    let mut rng = rand::thread_rng();
    let state_updates = manager.flip_each_cluster_rng(0.5, &mut rng);
    println!("{:?}", state_updates);
}

#[test]
fn multi_twosite_cluster_test() {
    let mut manager = SimpleOpDiagonal::new(4);
    manager.set_pth(
        0,
        Some(Op {
            vars: smallvec![0],
            bond: 0,
            inputs: smallvec![false],
            outputs: smallvec![false],
        }),
    );
    manager.set_pth(
        1,
        Some(Op {
            vars: smallvec![0],
            bond: 0,
            inputs: smallvec![false],
            outputs: smallvec![false],
        }),
    );

    manager.set_pth(
        2,
        Some(Op {
            vars: smallvec![1, 2],
            bond: 0,
            inputs: smallvec![false, false],
            outputs: smallvec![false, false],
        }),
    );
    manager.set_pth(
        3,
        Some(Op {
            vars: smallvec![2, 3],
            bond: 0,
            inputs: smallvec![false, false],
            outputs: smallvec![false, false],
        }),
    );

    let mut manager = manager.convert_to_looper();

    let mut rng = rand::thread_rng();
    let state_updates = manager.flip_each_cluster_rng(0.5, &mut rng);
    println!("{:?}", state_updates);
}

#[test]
fn multi_multisite_cluster_test() {
    let mut manager = SimpleOpDiagonal::new(3);
    manager.set_pth(
        0,
        Some(Op {
            vars: smallvec![0],
            bond: 0,
            inputs: smallvec![false],
            outputs: smallvec![false],
        }),
    );
    manager.set_pth(
        1,
        Some(Op {
            vars: smallvec![1, 2],
            bond: 1,
            inputs: smallvec![false, false],
            outputs: smallvec![false, false],
        }),
    );
    manager.set_pth(
        2,
        Some(Op {
            vars: smallvec![1],
            bond: 2,
            inputs: smallvec![false],
            outputs: smallvec![false],
        }),
    );

    let mut manager = manager.convert_to_looper();

    let mut rng = rand::thread_rng();
    let state_updates = manager.flip_each_cluster_rng(0.5, &mut rng);
    println!("{:?}", state_updates);
}
