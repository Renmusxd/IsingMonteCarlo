use crate::sse::arena::*;
use crate::sse::qmc_traits::*;
use crate::sse::qmc_types::Op;

#[derive(Clone)]
pub struct SimpleOpDiagonal {
    ops: Vec<Option<Op>>,
    n: usize,
    nvars: usize,
    arena: Arena<Option<usize>>,
}

impl SimpleOpDiagonal {
    pub(crate) fn set_min_size(&mut self, n: usize) {
        if self.ops.len() < n {
            self.ops.resize_with(n, || None)
        }
    }

    pub fn debug_print<H>(&self, h: H)
    where
        H: Fn(&[usize], usize, &[bool], &[bool]) -> f64,
    {
        fn lines_for(a: usize, b: usize) {
            (a..b).for_each(|_| {
                print!("| ");
            });
        }

        print!("    \t");
        lines_for(0, self.nvars);
        println!();
        let empty: [usize; 0] = [];
        self.ops.iter().enumerate().for_each(|(p, op)| {
            let (vars, d): (&[usize], bool) = match op {
                Some(op) => (&op.vars, op.is_diagonal()),
                None => (&empty, false),
            };
            print!("{:>5}\t", p);
            let last = vars.iter().fold(0, |acc, v| {
                lines_for(acc, *v);
                if d {
                    print!("O ");
                } else {
                    print!("X ");
                }
                *v + 1
            });
            lines_for(last, self.nvars);
            match op {
                Some(op) => println!(
                    "\t{:?}\t{}",
                    op,
                    h(&op.vars, op.bond, &op.inputs, &op.outputs)
                ),
                None => println!(),
            };
        })
    }
}

impl OpContainerConstructor for SimpleOpDiagonal {
    fn new(nvars: usize) -> Self {
        Self {
            ops: vec![],
            n: 0,
            nvars,
            arena: Arena::new(None),
        }
    }
}

impl OpContainer for SimpleOpDiagonal {
    fn set_cutoff(&mut self, cutoff: usize) {
        self.set_min_size(cutoff)
    }

    fn get_n(&self) -> usize {
        self.n
    }

    fn get_nvars(&self) -> usize {
        self.nvars
    }

    fn get_pth(&self, p: usize) -> Option<&Op> {
        if p >= self.ops.len() {
            None
        } else {
            self.ops[p].as_ref()
        }
    }

    fn weight<H>(&self, h: H) -> f64
    where
        H: Fn(&[usize], usize, &[bool], &[bool]) -> f64,
    {
        self.ops
            .iter()
            .filter(|op| op.is_some())
            .fold(1.0, |t, op| {
                let op = op.as_ref().unwrap();
                h(&op.vars, op.bond, &op.inputs, &op.outputs) * t
            })
    }
}

impl DiagonalUpdater for SimpleOpDiagonal {
    fn set_pth(&mut self, p: usize, op: Option<Op>) -> Option<Op> {
        self.set_min_size(p + 1);
        let temp = self.ops[p].take();
        self.ops[p] = op;
        temp
    }
}

#[derive(Clone)]
pub struct SimpleOpNode {
    pub(crate) op: Op,
    pub(crate) previous_p: Option<usize>,
    pub(crate) next_p: Option<usize>,
    pub(crate) previous_for_vars: ArenaIndex,
    pub(crate) next_for_vars: ArenaIndex,
}

impl SimpleOpNode {
    fn new(op: Op, previous_for_vars: ArenaIndex, next_for_vars: ArenaIndex) -> Self {
        let nvars = op.vars.len();
        assert_eq!(previous_for_vars.size(), nvars);
        assert_eq!(next_for_vars.size(), nvars);
        Self {
            op,
            previous_p: None,
            next_p: None,
            previous_for_vars,
            next_for_vars,
        }
    }
}

impl OpNode for SimpleOpNode {
    fn get_op(&self) -> Op {
        self.op.clone()
    }

    fn get_op_ref(&self) -> &Op {
        &self.op
    }

    fn get_op_mut(&mut self) -> &mut Op {
        &mut self.op
    }
}

pub struct SimpleOpLooper {
    ops: Vec<Option<SimpleOpNode>>,
    nth_ps: Vec<usize>,
    p_ends: Option<(usize, usize)>,
    var_ends: Vec<Option<(usize, usize)>>,
    arena: Arena<Option<usize>>,
}

impl OpContainer for SimpleOpLooper {
    fn set_cutoff(&mut self, cutoff: usize) {
        if cutoff > self.ops.len() {
            self.ops.resize_with(cutoff, || None);
        }
    }

    fn get_n(&self) -> usize {
        self.nth_ps.len()
    }

    fn get_nvars(&self) -> usize {
        self.var_ends.len()
    }

    fn get_pth(&self, p: usize) -> Option<&Op> {
        self.ops[p].as_ref().map(|opnode| &opnode.op)
    }

    fn weight<H>(&self, h: H) -> f64
    where
        H: Fn(&[usize], usize, &[bool], &[bool]) -> f64,
    {
        let mut t = 1.0;
        let mut p = self.p_ends.map(|(p, _)| p);
        while p.is_some() {
            let op = self.ops[p.unwrap()].as_ref().unwrap();
            t *= h(&op.op.vars, op.op.bond, &op.op.inputs, &op.op.outputs);
            p = op.next_p;
        }
        t
    }
}

impl LoopUpdater<SimpleOpNode> for SimpleOpLooper {
    fn get_node_ref(&self, p: usize) -> Option<&SimpleOpNode> {
        self.ops[p].as_ref()
    }

    fn get_node_mut(&mut self, p: usize) -> Option<&mut SimpleOpNode> {
        self.ops[p].as_mut()
    }

    fn get_first_p(&self) -> Option<usize> {
        self.p_ends.map(|(p, _)| p)
    }

    fn get_last_p(&self) -> Option<usize> {
        self.p_ends.map(|(_, p)| p)
    }

    fn get_first_p_for_var(&self, var: usize) -> Option<usize> {
        self.var_ends[var].map(|(p, _)| p)
    }

    fn get_last_p_for_var(&self, var: usize) -> Option<usize> {
        self.var_ends[var].map(|(_, p)| p)
    }

    fn get_previous_p(&self, node: &SimpleOpNode) -> Option<usize> {
        node.previous_p
    }

    fn get_next_p(&self, node: &SimpleOpNode) -> Option<usize> {
        node.next_p
    }

    fn get_previous_p_for_rel_var(&self, revar: usize, node: &SimpleOpNode) -> Option<usize> {
        self.arena[&node.previous_for_vars][revar]
    }

    fn get_next_p_for_rel_var(&self, revar: usize, node: &SimpleOpNode) -> Option<usize> {
        self.arena[&node.next_for_vars][revar]
    }

    fn get_nth_p(&self, n: usize) -> usize {
        self.nth_ps[n]
    }
}

impl ClusterUpdater<SimpleOpNode> for SimpleOpLooper {}

impl ConvertsToDiagonal<SimpleOpDiagonal> for SimpleOpLooper {
    fn convert_to_diagonal(self) -> SimpleOpDiagonal {
        let n = self.get_n();
        let nvars = self.get_nvars();
        let ops = self
            .ops
            .into_iter()
            .map(|opnode| opnode.map(|opnode| opnode.op))
            .collect();
        let mut arena = self.arena;
        arena.clear();
        SimpleOpDiagonal {
            ops,
            n,
            nvars,
            arena,
        }
    }
}

impl ConvertsToLooper<SimpleOpNode, SimpleOpLooper> for SimpleOpDiagonal {
    fn convert_to_looper(self) -> SimpleOpLooper {
        let mut p_ends = None;
        let mut var_ends = vec![None; self.nvars];
        let mut arena = self.arena;
        let mut opnodes = self
            .ops
            .iter()
            .map(|op| {
                op.clone().map(|op| {
                    let previous_slice = arena.get_alloc(op.vars.len());
                    let next_slice = arena.get_alloc(op.vars.len());
                    SimpleOpNode::new(op, previous_slice, next_slice)
                })
            })
            .collect::<Vec<_>>();

        let nth_ps = self
            .ops
            .iter()
            .enumerate()
            .filter_map(|(n, op)| op.as_ref().map(|op| (n, op)))
            .map(|(p, op)| {
                match p_ends {
                    None => p_ends = Some((p, p)),
                    Some((_, last_p)) => {
                        let last_op = opnodes[last_p].as_mut().unwrap();
                        last_op.next_p = Some(p);
                        p_ends.as_mut().unwrap().1 = p;
                        let this_opnode = opnodes[p].as_mut().unwrap();
                        this_opnode.previous_p = Some(last_p);
                    }
                }
                op.vars
                    .iter()
                    .cloned()
                    .enumerate()
                    .for_each(|(indx, v)| match var_ends.get(v) {
                        Some(None) => var_ends[v] = Some((p, p)),
                        Some(Some((_, last_p))) => {
                            let last_p = *last_p;
                            let last_op = opnodes[last_p].as_mut().unwrap();
                            let last_relvar = last_op.op.index_of_var(v).unwrap();
                            arena[&last_op.next_for_vars][last_relvar] = Some(p);
                            var_ends[v].as_mut().unwrap().1 = p;
                            let this_opnode = opnodes[p].as_mut().unwrap();
                            arena[&this_opnode.previous_for_vars][indx] = Some(last_p);
                        },
                        None => unreachable!()
                    });
                p
            })
            .collect();

        SimpleOpLooper {
            ops: opnodes,
            nth_ps,
            p_ends,
            var_ends,
            arena,
        }
    }
}
