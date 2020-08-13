use qmc::sse::*;
use rand::prelude::ThreadRng;

fn main() {
    let edges = vec![((0, 1), -1.0), ((1, 2), 1.0), ((2, 3), 1.0), ((3, 0), 1.0)];
    let transverse = 1.0;

    let rng = rand::thread_rng();
    let mut g = DefaultQMCIsingGraph::<ThreadRng>::new_with_rng(edges, transverse, 3, rng, None);
    let _a = g.timesteps(1000, 1.0);
}
