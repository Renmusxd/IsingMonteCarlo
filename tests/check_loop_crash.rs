use qmc::sse::fast_ops::*;
use qmc::sse::*;
use rand::prelude::*;
use smallvec::smallvec;

#[test]
fn run_single_bond() {
    let mut manager = FastOps::new_from_ops(
        2,
        vec![(
            0,
            FastOp::<2>::diagonal(smallvec![0, 1], 0, smallvec![false, false], false),
        )]
        .into_iter(),
    );
    let mut state = vec![false, false];
    let mut rng = SmallRng::seed_from_u64(0);

    (0..100).for_each(|_| {
        manager.make_loop_update_with_rng(
            None,
            |_, _, inputs, outputs| {
                if inputs == outputs {
                    1.0
                } else if inputs.iter().zip(outputs.iter().rev()).all(|(a, b)| a == b) {
                    1.0
                } else {
                    0.0
                }
            },
            &mut state,
            &mut rng,
        )
    });
    println!("{:?}", state);
    assert!(manager.verify(&state));
}

#[test]
fn run_double_bond() {
    let mut manager = FastOps::new_from_ops(
        3,
        vec![
            (
                0,
                FastOp::<2>::diagonal(smallvec![0, 1], 0, smallvec![false, false], false),
            ),
            (
                1,
                FastOp::<2>::diagonal(smallvec![1, 2], 1, smallvec![false, false], false),
            ),
        ]
        .into_iter(),
    );
    let mut state = vec![false, false, false];
    let mut rng = SmallRng::seed_from_u64(0);

    (0..100).for_each(|_| {
        manager.make_loop_update_with_rng(
            None,
            |_, _, inputs, outputs| {
                if inputs == outputs {
                    1.0
                } else if inputs.iter().zip(outputs.iter().rev()).all(|(a, b)| a == b) {
                    1.0
                } else {
                    0.0
                }
            },
            &mut state,
            &mut rng,
        )
    });
    println!("{:?}", state);
    assert!(manager.verify(&state));
}
