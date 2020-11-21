use crate::sse::Hamiltonian;

/// A hamiltonian for the graph.
pub(crate) struct Ham<'a, H, E>
where
    H: Fn(&[usize], usize, &[bool], &[bool]) -> f64,
    E: Fn(usize) -> (&'a [usize], bool),
{
    h: H,
    n_edges: usize,
    e_fn: E,
}

impl<'a, H, E> Ham<'a, H, E>
where
    H: Fn(&[usize], usize, &[bool], &[bool]) -> f64,
    E: Fn(usize) -> (&'a [usize], bool),
{
    /// Construct a new hamiltonian with a function, edge lookup function, and the number of bonds.
    pub(crate) fn new(hamiltonian: H, edge_fn: E, num_edges: usize) -> Self {
        Self {
            h: hamiltonian,
            n_edges: num_edges,
            e_fn: edge_fn,
        }
    }
}

impl<'a, H, E> Hamiltonian<'a> for Ham<'a, H, E>
where
    H: Fn(&[usize], usize, &[bool], &[bool]) -> f64,
    E: Fn(usize) -> (&'a [usize], bool),
{
    fn hamiltonian(&self, vars: &[usize], bond: usize, inputs: &[bool], outputs: &[bool]) -> f64 {
        (self.h)(vars, bond, inputs, outputs)
    }

    fn edge_fn(&self, bond: usize) -> (&'a [usize], bool) {
        (self.e_fn)(bond)
    }

    fn num_edges(&self) -> usize {
        self.n_edges
    }
}
