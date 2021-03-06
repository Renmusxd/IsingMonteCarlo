use qmc::sse::qmc_ising::new_qmc;
use qmc::sse::*;

fn main() {
    let side_len = 24;
    let nvars = side_len * side_len;
    let beta = 1.0;
    let transverse = 1.0;

    let timesteps = 1000;

    let indices: Vec<(usize, usize)> = (0usize..side_len)
        .map(|i| {
            (0usize..side_len)
                .map(|j| (i, j))
                .collect::<Vec<(usize, usize)>>()
        })
        .flatten()
        .collect();
    let f = |i, j| j * side_len + i;

    let right_connects = indices
        .iter()
        .cloned()
        .map(|(i, j)| (f(i, j), f((i + 1) % side_len, j)));
    let down_connects = indices
        .iter()
        .cloned()
        .map(|(i, j)| (f(i, j), f(i, (j + 1) % side_len)));
    let edges = right_connects
        .chain(down_connects)
        .map(|(i, j)| ((i, j), 1.0))
        .collect::<Vec<_>>();

    let cutoff = nvars;
    let mut qmc_graph = new_qmc(edges, transverse, 0., cutoff, None);

    let _plot = qmc_graph.timesteps_sample(timesteps, beta, None);
}
