use crate::memory::allocator::{Allocator, Factory};
use crate::sse::qmc_ising::IsingManager;
use crate::sse::qmc_runner::QMCManager;
use crate::sse::qmc_traits::*;
use crate::sse::qmc_types::{Leg, OpSide};
#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};
use smallvec::{smallvec, SmallVec};
use std::cmp::min;

/// Underlying op for storing graph data, good for 2-variable ops.
pub type FastOp = BasicOp<SmallVec<[usize; 2]>, SmallVec<[bool; 2]>>;
/// A default implementation of the FastOps container, good for 2-variable ops.
pub type FastOps = FastOpsTemplate<FastOp>;
/// A default implementation of the FastOpNode container, good for 2-variable ops.
pub type FastOpNode = FastOpNodeTemplate<FastOp>;

/// Underlying op for storing graph data, good for 2-variable ops.
#[cfg(feature = "const_generics")]
pub type FastOpN<const N: usize> = BasicOp<SmallVec<[usize; N]>, SmallVec<[bool; N]>>;
/// A default implementation of the FastOps container, good for 2-variable ops.
#[cfg(feature = "const_generics")]
pub type FastOpsN<const N: usize> = FastOpsTemplate<FastOpN<N>>;
/// A default implementation of the FastOpNode container, good for 2-variable ops.
#[cfg(feature = "const_generics")]
pub type FastOpNodeN<const N: usize> = FastOpNodeTemplate<FastOpN<N>>;

/// A fast op container.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct FastOpsTemplate<O: Op> {
    pub(crate) ops: Vec<Option<FastOpNodeTemplate<O>>>,
    pub(crate) n: usize,
    pub(crate) p_ends: Option<(usize, usize)>,
    pub(crate) var_ends: Vec<Option<(PRel, PRel)>>,

    // Optional bond counting.
    bond_counters: Option<Vec<usize>>,

    // Allocators to save malloc calls.
    usize_alloc: Allocator<Vec<usize>>,
    bool_alloc: Allocator<Vec<bool>>,
    opside_alloc: Allocator<Vec<OpSide>>,
    leg_alloc: Allocator<Vec<Leg>>,
    option_usize_alloc: Allocator<Vec<Option<usize>>>,
    f64_alloc: Allocator<Vec<f64>>,
    bond_container_alloc: Allocator<BondContainer<usize>>,
}

impl<O: Op + Clone> FastOpsTemplate<O> {
    fn new_from_nvars_and_nbonds(nvars: usize, nbonds: Option<usize>) -> Self {
        Self {
            ops: vec![],
            n: 0,
            p_ends: None,
            var_ends: vec![None; nvars],
            bond_counters: nbonds.map(|nbonds| vec![0; nbonds]),
            // Set bounds to make sure there are not "leaks"
            usize_alloc: Allocator::new_with_max_in_flight(4),
            bool_alloc: Allocator::new_with_max_in_flight(2),
            opside_alloc: Allocator::new_with_max_in_flight(1),
            leg_alloc: Allocator::new_with_max_in_flight(1),
            option_usize_alloc: Allocator::new_with_max_in_flight(2),
            f64_alloc: Allocator::new_with_max_in_flight(1),
            bond_container_alloc: Allocator::new_with_max_in_flight(2),
        }
    }

    fn new_from_nvars(nvars: usize) -> Self {
        Self::new_from_nvars_and_nbonds(nvars, None)
    }

    /// Make a new Manager from an interator of ops, and number of variables.
    pub fn new_from_ops<I: Iterator<Item = (usize, O)>>(nvars: usize, ps_and_ops: I) -> Self {
        let mut man = Self::new_from_nvars(nvars);
        man.clear_and_install_ops(ps_and_ops);
        man
    }

    fn clear_and_install_ops<I: Iterator<Item = (usize, O)>>(&mut self, ps_and_ops: I) {
        let nvars = self.var_ends.len();
        let ps_and_ops = ps_and_ops.collect::<Vec<_>>();
        if ps_and_ops.is_empty() {
            return;
        }
        let opslen = ps_and_ops.iter().map(|(p, _)| p).max().unwrap() + 1;
        self.ops.clear();
        self.ops.resize_with(opslen, || None);
        self.var_ends.clear();
        self.var_ends.resize_with(nvars, || None);
        self.p_ends = None;

        let last_p: Option<usize> = None;
        let mut last_vars: Vec<Option<usize>> = self.get_instance();
        let mut last_rels: Vec<Option<usize>> = self.get_instance();
        last_vars.resize(nvars, None);
        last_rels.resize(nvars, None);

        let (last_p, last_vars, last_rels) = ps_and_ops.into_iter().fold(
            (last_p, last_vars, last_rels),
            |(last_p, mut last_vars, mut last_rels), (p, op)| {
                assert!(last_p.map(|last_p| p > last_p).unwrap_or(true));

                if let Some(last_p) = last_p {
                    let last_node = self.ops[last_p].as_mut().unwrap();
                    last_node.next_p = Some(p);
                } else {
                    self.p_ends = Some((p, p))
                }

                let previous_for_vars = op
                    .get_vars()
                    .iter()
                    .cloned()
                    .enumerate()
                    .map(|(relv, v)| {
                        let last_tup = last_vars[v].zip(last_rels[v]);
                        if let Some((last_p, last_rel)) = last_tup {
                            let node = self.ops[last_p].as_mut().unwrap();
                            node.next_for_vars[last_rel] = Some(PRel { p, relv });
                        } else {
                            let end = PRel { p, relv };
                            self.var_ends[v] = Some((end, end));
                        }
                        last_vars[v] = Some(p);
                        last_rels[v] = Some(relv);
                        last_tup.map(PRel::from)
                    })
                    .collect();
                let n_opvars = op.get_vars().len();
                let mut node =
                    FastOpNodeTemplate::<O>::new(op, previous_for_vars, smallvec![None; n_opvars]);
                node.previous_p = last_p;
                self.ops[p] = Some(node);
                self.n += 1;

                (Some(p), last_vars, last_rels)
            },
        );
        if let Some((_, p_end)) = self.p_ends.as_mut() {
            *p_end = last_p.unwrap()
        }
        self.var_ends
            .iter_mut()
            .zip(
                last_vars
                    .iter()
                    .cloned()
                    .zip(last_rels.iter().cloned())
                    .map(|(a, b)| a.zip(b)),
            )
            .for_each(|(ends, last_v)| {
                if let Some((_, v_end)) = ends {
                    let (last_p, last_relv) = last_v.unwrap();
                    v_end.p = last_p;
                    v_end.relv = last_relv;
                }
            });
        self.return_instance(last_vars);
        self.return_instance(last_rels);
    }
}

// TODO make this const generic somehow? Possible to reuse op::vars type stuff?
// For each var, (p, relvar)
type LinkVars = SmallVec<[Option<PRel>; 2]>;

/// A node which contains ops for FastOps.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct FastOpNodeTemplate<O: Op> {
    pub(crate) op: O,
    pub(crate) previous_p: Option<usize>,
    pub(crate) next_p: Option<usize>,
    pub(crate) previous_for_vars: LinkVars,
    pub(crate) next_for_vars: LinkVars,
}

impl<O: Op> FastOpNodeTemplate<O> {
    fn new(op: O, previous_for_vars: LinkVars, next_for_vars: LinkVars) -> Self {
        let nvars = op.get_vars().len();
        assert_eq!(previous_for_vars.len(), nvars);
        assert_eq!(next_for_vars.len(), nvars);
        Self {
            op,
            previous_p: None,
            next_p: None,
            previous_for_vars,
            next_for_vars,
        }
    }
}

impl<O: Op + Clone> OpNode<O> for FastOpNodeTemplate<O> {
    fn get_op(&self) -> O {
        self.op.clone()
    }

    fn get_op_ref(&self) -> &O {
        &self.op
    }

    fn get_op_mut(&mut self) -> &mut O {
        &mut self.op
    }
}

/// Args for fast op mutation.
#[derive(Debug)]
pub struct FastOpMutateArgs {
    /// Last p seen
    last_p: Option<usize>,
    /// Last p seen per var
    last_vars: Vec<Option<usize>>,
    /// Relative index of var.
    last_rels: Vec<Option<usize>>,
    /// (vars->subvars, subvars)
    subvar_mapping: Option<(Vec<usize>, Vec<usize>)>,
    /// Unfilled vars
    unfilled: usize,
}

impl FastOpMutateArgs {
    fn new<Fact>(nvars: usize, vars: Option<&[usize]>, fact: &mut Fact) -> Self
    where
        Fact: Factory<Vec<usize>> + Factory<Vec<Option<usize>>>,
    {
        let last_p: Option<usize> = None;
        let mut last_vars: Vec<Option<usize>> = fact.get_instance();
        let mut last_rels: Vec<Option<usize>> = fact.get_instance();
        last_vars.resize(nvars, None);
        last_rels.resize(nvars, None);

        FastOpMutateArgs {
            last_p,
            last_vars,
            last_rels,
            subvar_mapping: vars.map(|vars| {
                let mut vars_to_subvars: Vec<usize> = fact.get_instance();
                let mut subvars_to_vars: Vec<usize> = fact.get_instance();
                vars_to_subvars.resize(nvars, std::usize::MAX);
                subvars_to_vars.extend_from_slice(vars);
                subvars_to_vars
                    .iter()
                    .enumerate()
                    .for_each(|(i, v)| vars_to_subvars[*v] = i);
                (vars_to_subvars, subvars_to_vars)
            }),
            unfilled: 0,
        }
    }
}

impl MutateArgs for FastOpMutateArgs {
    type SubvarIndex = usize;

    fn n_subvars(&self) -> usize {
        self.last_vars.len()
    }

    fn subvar_to_var(&self, index: Self::SubvarIndex) -> usize {
        match &self.subvar_mapping {
            None => index,
            Some((_, subvars)) => subvars[index],
        }
    }

    fn var_to_subvar(&self, var: usize) -> Option<Self::SubvarIndex> {
        match &self.subvar_mapping {
            None => Some(var),
            Some((allvars, _)) => {
                let i = allvars[var];
                if i == std::usize::MAX {
                    None
                } else {
                    Some(i)
                }
            }
        }
    }
}

impl<O: Op + Clone> DiagonalSubsection for FastOpsTemplate<O> {
    type Args = FastOpMutateArgs;
    fn mutate_p<T, F>(&mut self, f: F, p: usize, t: T, mut args: Self::Args) -> (T, Self::Args)
    where
        F: Fn(&Self, Option<&Self::Op>, T) -> (Option<Option<Self::Op>>, T),
    {
        let op_ref = self.get_pth(p);
        let (new_op, t) = f(&self, op_ref, t);

        // If we are making a change.
        if let Some(new_op) = new_op {
            // Lets check that all vars are included in the subvars of `args`.
            debug_assert!(
                {
                    op_ref
                        .map(|op| {
                            op.get_vars()
                                .iter()
                                .all(|v| args.var_to_subvar(*v).is_some())
                        })
                        .unwrap_or(true)
                        && new_op
                        .as_ref()
                        .map(|op| {
                            op.get_vars()
                                .iter()
                                .all(|v| args.var_to_subvar(*v).is_some())
                        })
                        .unwrap_or(true)

                },
                "Trying to mutate from or into an op which spans variables not prepared in the args."
            );

            let old_op_node = self.ops[p].take();

            // Check if the nodes share all the same variables, in which case we can do a
            // quick install since all linked list components are the same.
            let same_vars = new_op
                .as_ref()
                .and_then(|new_op| {
                    old_op_node
                        .as_ref()
                        .map(|node| node.op.get_vars() == new_op.get_vars())
                })
                .unwrap_or(false);
            if same_vars {
                // Can do a quick install
                let new_op = new_op.unwrap();
                let old_op_node = old_op_node.unwrap();
                let mut node_ref = FastOpNodeTemplate::new(
                    new_op,
                    old_op_node.previous_for_vars,
                    old_op_node.next_for_vars,
                );
                node_ref.previous_p = old_op_node.previous_p;
                node_ref.next_p = old_op_node.next_p;
                if let Some(bond_counters) = self.bond_counters.as_mut() {
                    let bond = old_op_node.op.get_bond();
                    bond_counters[bond] -= 1;
                    let bond = node_ref.op.get_bond();
                    bond_counters[bond] += 1;
                }
                self.ops[p] = Some(node_ref);
            } else {
                if let Some(node_ref) = old_op_node {
                    // Uninstall the old op.
                    // If there's a previous p, point it towards the next one
                    if let Some(last_p) = args.last_p {
                        let node = self.ops[last_p].as_mut().unwrap();
                        node.next_p = node_ref.next_p;
                    } else {
                        // No previous p, so we are removing the head. Make necessary adjustments.
                        self.p_ends = if let Some((head, tail)) = self.p_ends {
                            debug_assert_eq!(head, p);
                            node_ref.next_p.map(|new_head| {
                                debug_assert_ne!(p, tail);
                                (new_head, tail)
                            })
                        } else {
                            unreachable!()
                        }
                    }
                    // If there's a next p, point it towards the previous one
                    let next_p_node = node_ref.next_p.and_then(|p| self.ops[p].as_mut());
                    if let Some(next_p_node) = next_p_node {
                        next_p_node.previous_p = args.last_p
                    } else {
                        // No next p, so we are removing the tail. Adjust.
                        self.p_ends = if let Some((head, tail)) = self.p_ends {
                            debug_assert_eq!(tail, p);
                            node_ref.previous_p.map(|new_tail| {
                                debug_assert_ne!(head, p);
                                (head, new_tail)
                            })
                        } else {
                            // Normally not allowed, but could have been set to None up above.
                            None
                        }
                    }

                    // Now do the same for variables.
                    let vars = node_ref.op.get_vars();
                    vars.iter().cloned().enumerate().for_each(|(relv, v)| {
                        // Check the previous node using this variable.
                        let subvar = args.var_to_subvar(v).unwrap();
                        if let Some((prev_p_for_v, prev_rel_indx)) =
                            args.last_vars[subvar].zip(args.last_rels[subvar])
                        {
                            let prev_p_for_v_ref = self.ops[prev_p_for_v].as_mut().unwrap();
                            prev_p_for_v_ref.next_for_vars[prev_rel_indx] =
                                node_ref.next_for_vars[relv];
                        } else {
                            // This was the first one, need to edit vars list.
                            self.var_ends[v] = if let Some((head, tail)) = self.var_ends[v] {
                                debug_assert_eq!(head, PRel { p, relv });
                                // If None then we are removing the head and the tail.
                                node_ref.next_for_vars[relv].map(|new_head| {
                                    debug_assert_ne!(tail, PRel { p, relv });
                                    (new_head, tail)
                                })
                            } else {
                                unreachable!()
                            }
                        }

                        // Check the next nodes using this variable.
                        if let Some(PRel {
                            p: next_p_for_v,
                            relv: next_rel_index,
                        }) = node_ref.next_for_vars[relv]
                        {
                            let next_p_for_v_ref = self.ops[next_p_for_v].as_mut().unwrap();
                            next_p_for_v_ref.previous_for_vars[next_rel_index] = args.last_vars
                                [subvar]
                                .zip(args.last_rels[subvar])
                                .map(PRel::from);
                        } else {
                            self.var_ends[v] = if let Some((head, tail)) = self.var_ends[v] {
                                debug_assert_eq!(tail, PRel { p, relv });
                                // If None then we are removing the head and the tail.
                                node_ref.previous_for_vars[relv].map(|new_tail| {
                                    debug_assert_ne!(head, PRel { p, relv });
                                    (head, new_tail)
                                })
                            } else {
                                // Could have been set to none previously.
                                None
                            }
                        }
                    });
                    self.n -= 1;
                    if let Some(bond_counters) = self.bond_counters.as_mut() {
                        let bond = node_ref.op.get_bond();
                        bond_counters[bond] -= 1;
                    }
                }

                if let Some(new_op) = new_op {
                    // Install the new one
                    let (prevs, nexts): (LinkVars, LinkVars) = new_op
                        .get_vars()
                        .iter()
                        .cloned()
                        .map(|v| -> (Option<PRel>, Option<PRel>) {
                            let subvar = args.var_to_subvar(v).unwrap();
                            let prev_p_and_rel = args.last_vars[subvar]
                                .zip(args.last_rels[subvar])
                                .map(PRel::from);

                            let next_p_and_rel = if let Some(prev_p_for_v) = args.last_vars[subvar]
                            {
                                // If there's a previous node for the var, check its next entry.
                                let prev_node_for_v = self.ops[prev_p_for_v].as_ref().unwrap();
                                let indx = prev_node_for_v.op.index_of_var(v).unwrap();
                                prev_node_for_v.next_for_vars[indx]
                            } else if let Some((head, _)) = self.var_ends[v] {
                                // Otherwise just look at the head (this is the new head).
                                debug_assert_eq!(prev_p_and_rel, None);
                                Some(head)
                            } else {
                                // This is the new tail.
                                None
                            };

                            (prev_p_and_rel, next_p_and_rel)
                        })
                        .unzip();

                    // Now adjust other nodes and ends
                    prevs
                        .iter()
                        .zip(new_op.get_vars().iter())
                        .enumerate()
                        .for_each(|(relv, (prev, v))| {
                            if let Some(PRel {
                                p: prev_p,
                                relv: prev_rel,
                            }) = prev
                            {
                                let prev_node = self.ops[*prev_p].as_mut().unwrap();
                                debug_assert_eq!(prev_node.get_op_ref().get_vars()[*prev_rel], *v);
                                prev_node.next_for_vars[*prev_rel] = Some(PRel::from((p, relv)));
                            } else {
                                self.var_ends[*v] = if let Some((head, tail)) = self.var_ends[*v] {
                                    debug_assert!(head.p >= p);
                                    Some((PRel { p, relv }, tail))
                                } else {
                                    Some((PRel { p, relv }, PRel { p, relv }))
                                }
                            }
                        });

                    nexts
                        .iter()
                        .zip(new_op.get_vars().iter())
                        .enumerate()
                        .for_each(|(relv, (next, v))| {
                            if let Some(PRel {
                                p: next_p,
                                relv: next_rel,
                            }) = next
                            {
                                let next_node = self.ops[*next_p].as_mut().unwrap();
                                debug_assert_eq!(next_node.get_op_ref().get_vars()[*next_rel], *v);
                                next_node.previous_for_vars[*next_rel] = Some(PRel { p, relv });
                            } else {
                                self.var_ends[*v] = if let Some((head, tail)) = self.var_ends[*v] {
                                    debug_assert!(tail.p <= p);
                                    Some((head, PRel { p, relv }))
                                } else {
                                    Some((PRel { p, relv }, PRel { p, relv }))
                                }
                            }
                        });

                    let mut node_ref = FastOpNodeTemplate::new(new_op, prevs, nexts);
                    node_ref.previous_p = args.last_p;
                    node_ref.next_p = if let Some(last_p) = args.last_p {
                        let last_p_node = self.ops[last_p].as_ref().unwrap();
                        last_p_node.next_p
                    } else if let Some((head, _)) = self.p_ends {
                        Some(head)
                    } else {
                        None
                    };

                    // Based on what these were set to, adjust the p_ends and neighboring nodes.
                    if let Some(prev) = node_ref.previous_p {
                        let prev_node = self.ops[prev].as_mut().unwrap();
                        prev_node.next_p = Some(p);
                    } else {
                        self.p_ends = if let Some((head, tail)) = self.p_ends {
                            debug_assert!(head >= p);
                            Some((p, tail))
                        } else {
                            Some((p, p))
                        }
                    };

                    if let Some(next) = node_ref.next_p {
                        let next_node = self.ops[next].as_mut().unwrap();
                        next_node.previous_p = Some(p);
                    } else {
                        self.p_ends = if let Some((head, tail)) = self.p_ends {
                            debug_assert!(tail <= p);
                            Some((head, p))
                        } else {
                            Some((p, p))
                        }
                    };
                    if let Some(bond_counters) = self.bond_counters.as_mut() {
                        let bond = node_ref.op.get_bond();
                        bond_counters[bond] += 1;
                    }
                    self.ops[p] = Some(node_ref);
                    self.n += 1;
                }
            }
        }

        if let Some(op) = self.ops[p].as_ref().map(|r| &r.op) {
            op.get_vars()
                .iter()
                .cloned()
                .enumerate()
                .for_each(|(relv, v)| {
                    args.last_vars[v] = Some(p);
                    args.last_rels[v] = Some(relv);
                });
            args.last_p = Some(p)
        }
        (t, args)
    }

    fn mutate_subsection<T, F>(
        &mut self,
        pstart: usize,
        pend: usize,
        t: T,
        f: F,
        args: Option<Self::Args>,
    ) -> T
    where
        F: Fn(&Self, Option<&Self::Op>, T) -> (Option<Option<Self::Op>>, T),
    {
        if pend > self.ops.len() {
            self.ops.resize(pend, None)
        };
        let args = match args {
            Some(args) => args,
            None => {
                let args = self.get_empty_args(SubvarAccess::ALL);
                self.fill_args_at_p(pstart, args)
            }
        };
        let (t, args) =
            (pstart..pend).fold((t, args), |(t, args), p| self.mutate_p(&f, p, t, args));
        self.return_args(args);

        t
    }

    fn mutate_subsection_ops<T, F>(
        &mut self,
        pstart: usize,
        pend: usize,
        mut t: T,
        f: F,
        args: Option<Self::Args>,
    ) -> T
    where
        F: Fn(&Self, &Self::Op, usize, T) -> (Option<Option<Self::Op>>, T),
    {
        if pend > self.ops.len() {
            self.ops.resize(pend, None)
        };

        let mut args = match args {
            Some(args) => args,
            None => {
                let args = self.get_empty_args(SubvarAccess::ALL);
                self.fill_args_at_p(pstart, args)
            }
        };

        // Find starting position.
        let mut p = self.p_ends.and_then(|(start, pend)| {
            if pstart <= start {
                Some(start)
            } else if start > pend {
                None
            } else {
                // Find the first p with an op.
                self.ops[pstart..]
                    .iter()
                    .enumerate()
                    .find(|(_, op)| op.is_some())
                    .map(|(x, _)| x + pstart)
            }
        });
        while let Some(node_p) = p {
            if node_p > pend {
                break;
            } else {
                let ret = self.mutate_p(|s, op, t| f(s, op.unwrap(), node_p, t), node_p, t, args);
                t = ret.0;
                args = ret.1;

                let node = self.ops[node_p].as_ref().unwrap();
                p = node.next_p;
            }
        }
        self.return_args(args);

        t
    }

    fn get_empty_args(&mut self, vars: SubvarAccess<FastOpMutateArgs>) -> FastOpMutateArgs {
        match vars {
            SubvarAccess::ALL => {
                let mut args = FastOpMutateArgs::new(self.get_nvars(), None, self);
                // Can subtract off vars with no ops. nvars - noops = nvars_with_ops
                args.unfilled = self.var_ends.iter().filter(|end| end.is_some()).count();
                args
            }
            SubvarAccess::VARLIST(vars) => {
                let mut args = FastOpMutateArgs::new(self.get_nvars(), Some(vars), self);
                // Can subtract off vars with no ops. nvars - noops = nvars_with_ops
                args.unfilled = vars.iter().filter(|v| self.var_ends[**v].is_some()).count();
                args
            }
            SubvarAccess::ARGS(mut args) => {
                // Count how many need to be set (are None).
                args.unfilled = (0..args.n_subvars())
                    .filter(|subvar| {
                        let v = args.subvar_to_var(*subvar);
                        args.last_vars[*subvar].is_none() && self.var_ends[v].is_some()
                    })
                    .count();
                args
            }
        }
    }

    // TODO test get_args_at_p for nonzero values, and for var subsections. Test empty vars!
    fn fill_args_at_p(&self, p: usize, empty_args: Self::Args) -> Self::Args {
        let args = empty_args;
        if args.unfilled > 0 {
            self.iter_ops_above_p(
                p,
                args,
                |p, node, mut args| {
                    if args.last_p.is_none() {
                        args.last_p = Some(p);
                    }
                    node.get_op_ref()
                        .get_vars()
                        .iter()
                        .cloned()
                        .enumerate()
                        .for_each(|(relv, v)| {
                            let subvar = args.var_to_subvar(v);
                            // Only set vars which we are looking at.
                            if let Some(subvar) = subvar {
                                if args.last_vars[subvar].is_none() {
                                    debug_assert_eq!(args.last_rels[subvar], None);
                                    args.last_vars[subvar] = Some(p);
                                    args.last_rels[subvar] = Some(relv);
                                    args.unfilled -= 1;
                                }
                            }
                        });
                    let cont = args.unfilled > 0;
                    (args, cont)
                },
                |node, mut args| {
                    node.get_op_ref()
                        .get_vars()
                        .iter()
                        .zip(node.previous_for_vars.iter())
                        .filter_map(|(v, prel)| prel.as_ref().map(|prel| (*v, prel)))
                        .for_each(|(v, prel)| {
                            let subvar = args.var_to_subvar(v);
                            // Only set vars which we are looking at.
                            if let Some(subvar) = subvar {
                                if args.last_vars[subvar].is_none() {
                                    debug_assert_eq!(args.last_rels[subvar], None);
                                    args.unfilled -= 1;
                                    args.last_vars[subvar] = Some(prel.p);
                                    args.last_rels[subvar] = Some(prel.relv);
                                }
                            }
                        });
                    args.last_p = node.previous_p;
                    let cont = args.unfilled > 0;
                    (args, cont)
                },
            )
        } else {
            args
        }
    }

    fn fill_args_at_p_with_hint<It>(
        &self,
        p: usize,
        args: &mut Self::Args,
        vars: &[usize],
        hint: It,
    ) where
        It: Iterator<Item = Option<usize>>,
    {
        let psel = p;
        // Need to find an op for each var that has ops on worldline.
        hint.zip(vars.iter().cloned())
            .enumerate()
            .for_each(|(subvar, (phint, var))| {
                // Get starting spot.
                let (mut pcheck, mut relv) = if let Some(phint) = phint {
                    // Use hint
                    let relv = self
                        .get_node_ref(phint)
                        .unwrap()
                        .get_op_ref()
                        .index_of_var(var)
                        .unwrap();
                    (phint, relv)
                } else if let Some((pstart, _)) = self.var_ends[var] {
                    // Manually look.
                    (pstart.p, pstart.relv)
                } else {
                    // No need to update substate.
                    return;
                };

                // Iterate to valid location.
                loop {
                    let node = self.get_node_ref(pcheck).unwrap();
                    let next = self
                        .get_next_p_for_rel_var(relv, node)
                        .unwrap_or_else(|| self.var_ends[var].unwrap().0);
                    // If psel in (pcheck, next.p)
                    if p_crosses(pcheck, next.p, psel) {
                        // Leave as None if wraps around.
                        if pcheck < next.p {
                            args.last_vars[subvar] = Some(pcheck);
                            args.last_rels[subvar] = Some(relv);
                        }
                        break;
                    } else {
                        pcheck = next.p;
                        relv = next.relv;
                    }
                }
            });

        // Set last_p to the first p found.
        args.last_p = (0..psel)
            .rev()
            .filter(|p| self.get_node_ref(*p).is_some())
            .next();
    }

    // fn fill_args_at_p_with_hint<It>(
    //     &self,
    //     p: usize,
    //     empty_args: Self::Args,
    //     hint: It,
    // ) -> Result<Self::Args, Self::Args>
    // where
    //     It: Iterator<Item = usize>,
    // {
    //     let psel = p;
    //     let mut args = empty_args;
    //
    //     // We can only use ps with nodes in them.
    //     let mut filled_hints = hint.filter_map(|p| self.get_node_ref(p));
    //     let res = filled_hints
    //         .try_for_each(|node| {
    //             // Check for last_p.
    //             let next_p = match self.get_next_p(node) {
    //                 Some(p) => p,
    //                 // There must be at least one node in the graph since node
    //                 // is not None.
    //                 None => self.p_ends.unwrap().0,
    //             };
    //             if p_crosses(p, next_p, psel) {
    //                 args.last_p = Some(p);
    //             }
    //
    //             // On this node look at each variable.
    //             node.get_op_ref()
    //                 .get_vars()
    //                 .iter()
    //                 .cloned()
    //                 .enumerate()
    //                 .try_for_each(|(relvar, v)| {
    //                     // Get subvar, if already set then set to None and return Ok(())
    //                     args.var_to_subvar(v)
    //                         .and_then(|subvar| match args.last_vars[subvar] {
    //                             None => Some(subvar),
    //                             Some(_) => None,
    //                         })
    //                         .map(|subvar| {
    //                             debug_assert_eq!(args.last_rels[subvar], None);
    //                             // So now lets check if this is a valid op to use for reference.
    //                             let next_p_for_var =
    //                                 self.get_next_p_for_rel_var(relvar, node).map(|prel| prel.p);
    //                             let first_p_for_var = self.var_ends[v].map(|(prel, _)| prel.p);
    //                             let next_p = match (next_p_for_var, first_p_for_var) {
    //                                 // Simply the next p.
    //                                 (Some(p), _) => p,
    //                                 // Wrap around pbc.
    //                                 (None, Some(p)) => p,
    //                                 // This var doesn't have anything on it, not possible since
    //                                 // we know at least one node is on it.
    //                                 (None, None) => unreachable!(),
    //                             };
    //                             // if p is before psel and next_p is after.
    //                             if p_crosses(p, next_p, psel) {
    //                                 args.last_vars[subvar] = Some(p);
    //                                 args.last_rels[subvar] = Some(relvar);
    //                                 args.unfilled -= 1;
    //                             }
    //                             if args.unfilled == 0 {
    //                                 Err(())
    //                             } else {
    //                                 Ok(())
    //                             }
    //                         })
    //                         .unwrap_or(Ok(()))
    //                 })
    //         })
    //         // If we ended early, check remaining for last_p if needed.
    //         .map_err(|_| {
    //             if args.last_p.is_none() {
    //                 filled_hints.try_for_each(|node| {
    //                     // Check for last_p.
    //                     let next_p = match self.get_next_p(node) {
    //                         Some(p) => p,
    //                         // There must be at least one node in the graph since node
    //                         // is not None.
    //                         None => self.p_ends.unwrap().0,
    //                     };
    //                     if p_crosses(p, next_p, psel) {
    //                         args.last_p = Some(p);
    //                         Err(())
    //                     } else {
    //                         Ok(())
    //                     }
    //                 })
    //             } else {
    //                 Err(())
    //             }
    //         });
    //     // Check the end result.
    //     // If we got through all ps without reducing count, we have failed.
    //     // If we ended early but were not able to find the last_p after continuing we also
    //     // failed to fill args.
    //     match res {
    //         Ok(()) | Err(Ok(())) => Err(args),
    //         _ => Ok(args),
    //     }
    // }

    fn return_args(&mut self, args: Self::Args) {
        self.return_instance(args.last_vars);
        self.return_instance(args.last_rels);
        if let Some((vars_to_subvars, subvars_to_vars)) = args.subvar_mapping {
            self.return_instance(vars_to_subvars);
            self.return_instance(subvars_to_vars);
        }
    }

    fn get_propagated_substate_with_hint<It>(
        &self,
        p: usize,
        substate: &mut [bool],
        vars: &[usize],
        hint: It,
    ) where
        It: Iterator<Item = Option<usize>>,
    {
        let psel = p;
        // Need to find an op for each var that has ops on worldline.
        hint.zip(vars.iter().cloned())
            .enumerate()
            .for_each(|(subvar, (phint, var))| {
                // Get starting spot.
                let (mut pcheck, mut relv) = if let Some(phint) = phint {
                    // Use hint
                    let relv = self
                        .get_node_ref(phint)
                        .unwrap()
                        .get_op_ref()
                        .index_of_var(var)
                        .unwrap();
                    (phint, relv)
                } else if let Some((pstart, _)) = self.var_ends[var] {
                    // Manually look.
                    (pstart.p, pstart.relv)
                } else {
                    // No need to update substate.
                    return;
                };

                // Iterate to valid location.
                loop {
                    let node = self.get_node_ref(pcheck).unwrap();
                    let next = self
                        .get_next_p_for_rel_var(relv, node)
                        .unwrap_or_else(|| self.var_ends[var].unwrap().0);
                    if p_crosses(pcheck, next.p, psel) {
                        substate[subvar] = node.get_op_ref().get_outputs()[relv];
                        break;
                    } else {
                        pcheck = next.p;
                        relv = next.relv;
                    }
                }
            });
    }
}

fn p_crosses(pstart: usize, pend: usize, psel: usize) -> bool {
    match (pstart, pend) {
        (pstart, pend) if pstart < pend => (pstart < psel) && (psel <= pend),
        (pstart, pend) if pstart > pend => (pend < psel) || (psel <= pstart),
        // Only one var. pstart == pend
        _ => true,
    }
}

impl<O: Op + Clone> DiagonalUpdater for FastOpsTemplate<O> {
    fn mutate_ps<F, T>(&mut self, pstart: usize, pend: usize, t: T, f: F) -> T
    where
        F: Fn(&Self, Option<&Self::Op>, T) -> (Option<Option<Self::Op>>, T),
    {
        self.mutate_subsection(pstart, pend, t, f, None)
    }

    fn mutate_ops<F, T>(&mut self, pstart: usize, pend: usize, t: T, f: F) -> T
    where
        F: Fn(&Self, &Self::Op, usize, T) -> (Option<Option<Self::Op>>, T),
    {
        self.mutate_subsection_ops(pstart, pend, t, f, None)
    }

    fn try_iterate_ps<F, T, V>(&self, pstart: usize, pend: usize, t: T, f: F) -> Result<T, V>
    where
        F: Fn(&Self, Option<&Self::Op>, T) -> Result<T, V>,
    {
        self.ops[min(pstart, self.ops.len())..min(pend, self.ops.len())]
            .iter()
            .try_fold(t, |t, op| f(self, op.as_ref().map(|op| op.get_op_ref()), t))
    }

    fn try_iterate_ops<F, T, V>(&self, pstart: usize, pend: usize, mut t: T, f: F) -> Result<T, V>
    where
        F: Fn(&Self, &Self::Op, usize, T) -> Result<T, V>,
    {
        // Find starting position.
        let mut p = self.p_ends.and_then(|(start, pend)| {
            if pstart <= start {
                Some(start)
            } else if start > pend {
                None
            } else {
                // Find the first p with an op.
                self.ops[pstart..]
                    .iter()
                    .enumerate()
                    .find(|(_, op)| op.is_some())
                    .map(|(x, _)| x + pstart)
            }
        });
        while let Some(node_p) = p {
            if node_p > pend {
                break;
            } else {
                let node = self.ops[node_p].as_ref().unwrap();
                t = f(self, node.get_op_ref(), node_p, t)?;
                p = node.next_p;
            }
        }
        Ok(t)
    }
}

impl<O: Op + Clone> HeatBathDiagonalUpdater for FastOpsTemplate<O> {}

impl<O: Op + Clone> OpContainerConstructor for FastOpsTemplate<O> {
    fn new(nvars: usize) -> Self {
        Self::new_from_nvars(nvars)
    }

    fn new_with_bonds(nvars: usize, nbonds: usize) -> Self {
        Self::new_from_nvars_and_nbonds(nvars, Some(nbonds))
    }
}

impl<O: Op + Clone> OpContainer for FastOpsTemplate<O> {
    type Op = O;

    // TODO remove if not needed
    // fn get_propagated_substate(
    //     &mut self,
    //     p: usize,
    //     state: &[bool],
    //     vars: &[usize],
    //     substate: &mut [bool],
    // ) {
    //     let vars_to_set = vars.len();
    //     let mut var_set: Vec<bool> = self.get_instance();
    //     var_set.resize(vars.len(), false);
    //     let mut var_lookup: Vec<Option<usize>> = self.get_instance();
    //     var_lookup.resize(state.len(), None);
    //     vars.iter()
    //         .enumerate()
    //         .for_each(|(i, v)| var_lookup[*v] = Some(i));
    //     let var_lookup = var_lookup;
    //
    //     let t = (vars_to_set, var_set, substate);
    //     let run_on_bools = |node: &FastOpNodeTemplate<O>,
    //                         bools: &[bool],
    //                         vars_to_set: &mut usize,
    //                         var_set: &mut [bool],
    //                         substate: &mut [bool]| {
    //         node.get_op_ref()
    //             .get_vars()
    //             .iter()
    //             .zip(bools.iter())
    //             .for_each(|(v, b)| {
    //                 if let Some(i) = var_lookup[*v] {
    //                     if !var_set[i] {
    //                         var_set[i] = true;
    //                         *vars_to_set -= 1;
    //                         substate[i] = *b;
    //                     }
    //                 }
    //             });
    //     };
    //
    //     let (vars_to_set, var_set, substate) = self.iter_ops_above_p(
    //         p,
    //         t,
    //         |_, node, t| {
    //             let (mut vars_to_set, mut var_set, substate) = t;
    //             run_on_bools(
    //                 node,
    //                 node.get_op_ref().get_outputs(),
    //                 &mut vars_to_set,
    //                 &mut var_set,
    //                 substate,
    //             );
    //             ((vars_to_set, var_set, substate), vars_to_set > 0)
    //         },
    //         |node, t| {
    //             let (mut vars_to_set, mut var_set, substate) = t;
    //             run_on_bools(
    //                 node,
    //                 node.get_op_ref().get_inputs(),
    //                 &mut vars_to_set,
    //                 &mut var_set,
    //                 substate,
    //             );
    //             ((vars_to_set, var_set, substate), vars_to_set > 0)
    //         },
    //     );
    //
    //     // If there are more to set.
    //     if vars_to_set > 0 {
    //         vars.iter()
    //             .zip(var_set.iter().zip(substate.iter_mut()))
    //             .for_each(|(v, (set, b))| {
    //                 if !set {
    //                     *b = state[*v]
    //                 }
    //             });
    //     }
    //
    //     self.return_instance(var_set);
    //     self.return_instance(var_lookup);
    // }

    fn get_cutoff(&self) -> usize {
        self.ops.len()
    }

    fn set_cutoff(&mut self, cutoff: usize) {
        if cutoff > self.ops.len() {
            self.ops.resize(cutoff, None)
        }
    }

    fn get_n(&self) -> usize {
        self.n
    }

    fn get_nvars(&self) -> usize {
        self.var_ends.len()
    }

    fn get_pth(&self, p: usize) -> Option<&Self::Op> {
        if p < self.ops.len() {
            self.ops[p].as_ref().map(|opnode| &opnode.op)
        } else {
            None
        }
    }

    fn get_count(&self, bond: usize) -> usize {
        self.bond_counters
            .as_ref()
            .map(|bc| bc.get(bond).copied().unwrap_or(0))
            .unwrap_or_else(|| {
                self.iterate_ops(0, self.get_cutoff(), 0, |_, op, _, count| {
                    if op.get_bond() == bond {
                        count + 1
                    } else {
                        count
                    }
                })
            })
    }
}

impl<O: Op + Clone> LoopUpdater for FastOpsTemplate<O> {
    type Node = FastOpNodeTemplate<O>;

    fn get_node_ref(&self, p: usize) -> Option<&Self::Node> {
        self.ops[p].as_ref()
    }

    fn get_node_mut(&mut self, p: usize) -> Option<&mut Self::Node> {
        self.ops[p].as_mut()
    }

    fn get_first_p(&self) -> Option<usize> {
        self.p_ends.map(|(p, _)| p)
    }

    fn get_last_p(&self) -> Option<usize> {
        self.p_ends.map(|(_, p)| p)
    }

    fn get_first_p_for_var(&self, var: usize) -> Option<PRel> {
        self.var_ends[var].map(|(start, _)| start)
    }

    fn get_last_p_for_var(&self, var: usize) -> Option<PRel> {
        self.var_ends[var].map(|(_, end)| end)
    }

    fn get_previous_p(&self, node: &Self::Node) -> Option<usize> {
        node.previous_p
    }

    fn get_next_p(&self, node: &Self::Node) -> Option<usize> {
        node.next_p
    }

    fn get_previous_p_for_rel_var(&self, relvar: usize, node: &Self::Node) -> Option<PRel> {
        node.previous_for_vars[relvar]
    }

    fn get_next_p_for_rel_var(&self, relvar: usize, node: &Self::Node) -> Option<PRel> {
        node.next_for_vars[relvar]
    }

    fn get_nth_p(&self, n: usize) -> usize {
        let n = n % self.n;
        let init = self.p_ends.map(|(head, _)| head).unwrap();
        (0..n).fold(init, |p, _| self.ops[p].as_ref().unwrap().next_p.unwrap())
    }
}

impl<O: Op + Clone> Factory<Vec<bool>> for FastOpsTemplate<O> {
    fn get_instance(&mut self) -> Vec<bool> {
        self.bool_alloc.get_instance()
    }

    fn return_instance(&mut self, t: Vec<bool>) {
        self.bool_alloc.return_instance(t)
    }
}

impl<O: Op + Clone> Factory<Vec<usize>> for FastOpsTemplate<O> {
    fn get_instance(&mut self) -> Vec<usize> {
        self.usize_alloc.get_instance()
    }

    fn return_instance(&mut self, t: Vec<usize>) {
        self.usize_alloc.return_instance(t)
    }
}

impl<O: Op + Clone> Factory<Vec<Option<usize>>> for FastOpsTemplate<O> {
    fn get_instance(&mut self) -> Vec<Option<usize>> {
        self.option_usize_alloc.get_instance()
    }

    fn return_instance(&mut self, t: Vec<Option<usize>>) {
        self.option_usize_alloc.return_instance(t)
    }
}

impl<O: Op + Clone> Factory<Vec<OpSide>> for FastOpsTemplate<O> {
    fn get_instance(&mut self) -> Vec<OpSide> {
        self.opside_alloc.get_instance()
    }

    fn return_instance(&mut self, t: Vec<OpSide>) {
        self.opside_alloc.return_instance(t)
    }
}
impl<O: Op + Clone> Factory<Vec<Leg>> for FastOpsTemplate<O> {
    fn get_instance(&mut self) -> Vec<Leg> {
        self.leg_alloc.get_instance()
    }

    fn return_instance(&mut self, t: Vec<Leg>) {
        self.leg_alloc.return_instance(t)
    }
}

impl<O: Op + Clone> Factory<Vec<f64>> for FastOpsTemplate<O> {
    fn get_instance(&mut self) -> Vec<f64> {
        self.f64_alloc.get_instance()
    }

    fn return_instance(&mut self, t: Vec<f64>) {
        self.f64_alloc.return_instance(t)
    }
}
impl<O: Op + Clone> Factory<BondContainer<usize>> for FastOpsTemplate<O> {
    fn get_instance(&mut self) -> BondContainer<usize> {
        self.bond_container_alloc.get_instance()
    }

    fn return_instance(&mut self, t: BondContainer<usize>) {
        self.bond_container_alloc.return_instance(t)
    }
}

impl<O: Op + Clone> ClusterUpdater for FastOpsTemplate<O> {}

impl<O: Op + Clone> RVBUpdater for FastOpsTemplate<O> {
    fn constant_ops_on_var(&self, var: usize, ps: &mut Vec<usize>) {
        let mut p_and_rel = self.get_first_p_for_var(var);
        while let Some(PRel {
            p: node_p,
            relv: node_relv,
        }) = p_and_rel
        {
            let node = self.get_node_ref(node_p).unwrap();
            debug_assert_eq!(node.get_op_ref().get_vars()[node_relv], var);
            if node.get_op_ref().is_constant() {
                ps.push(node_p);
            }
            p_and_rel = self.get_next_p_for_rel_var(node_relv, node);
        }
    }

    fn spin_flips_on_var(&self, var: usize, ps: &mut Vec<usize>) {
        let mut p_and_rel = self.get_first_p_for_var(var);
        while let Some(PRel {
            p: node_p,
            relv: node_relv,
        }) = p_and_rel
        {
            let node = self.get_node_ref(node_p).unwrap();
            let op = node.get_op_ref();
            debug_assert_eq!(op.get_vars()[node_relv], var);
            if op.get_inputs()[node_relv] != op.get_outputs()[node_relv] {
                ps.push(node_p)
            };
            p_and_rel = self.get_next_p_for_rel_var(node_relv, node);
        }
    }
}

impl<O: Op> IsingManager for FastOpsTemplate<O> {}
impl<O: Op> QMCManager for FastOpsTemplate<O> {}
