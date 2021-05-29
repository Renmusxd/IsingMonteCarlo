use crate::sse::fast_op_alloc::{DefaultFastOpAllocator, FastOpAllocator};
use crate::sse::qmc_ising::IsingManager;
use crate::sse::qmc_runner::QmcManager;
use crate::sse::qmc_traits::*;
use crate::sse::qmc_types::{Leg, OpSide};
use crate::util::allocator::Factory;
use crate::util::bondcontainer::BondContainer;
use crate::util::cmpby::CmpBy;
use crate::util::typed_vec::TypedVec;
#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};
use smallvec::{smallvec, SmallVec};
use std::cmp::{min, Reverse};
use std::collections::BinaryHeap;
use std::ops::{Index, IndexMut};

/// Underlying op for storing graph data, good for 2-variable ops.
pub type FastOp = BasicOp<SmallVec<[usize; 2]>, SmallVec<[bool; 2]>>;
/// A default implementation of the FastOps container, good for 2-variable ops.
pub type FastOps = FastOpsTemplate<FastOp>;
/// A default implementation of the FastOpNode container, good for 2-variable ops.
pub type FastOpNode = FastOpNodeTemplate<FastOp>;

/// Underlying op for storing graph data.
#[cfg(feature = "const_generics")]
pub type FastOpN<const N: usize> = BasicOp<SmallVec<[usize; N]>, SmallVec<[bool; N]>>;
/// A default implementation of the FastOps container.
#[cfg(feature = "const_generics")]
pub type FastOpsN<const N: usize> = FastOpsTemplate<FastOpN<N>>;
/// A default implementation of the FastOpNode container.
#[cfg(feature = "const_generics")]
pub type FastOpNodeN<const N: usize> =
    FastOpNodeTemplate<FastOpN<N>, SmallVec<[Option<PRel<FastOpIndex>>; N]>>;

/// Location in imaginary time guarenteed to have an operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct FastOpIndex {
    index: usize,
}

impl Into<usize> for FastOpIndex {
    fn into(self) -> usize {
        self.index
    }
}
impl From<usize> for FastOpIndex {
    fn from(i: usize) -> Self {
        Self { index: i }
    }
}

/// A fast op container.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct FastOpsTemplate<O: Op, ALLOC: FastOpAllocator = DefaultFastOpAllocator> {
    // TODO add way to swap out the FastOpNodeTemplate to allow const generics for LV.
    // right now all FastOpsTemplates pick N=2 for the LinkVars
    // once const generics have been out for a while can maybe just make it all N dependent.
    pub(crate) ops: TypedVec<FastOpIndex, (usize, FastOpNodeTemplate<O>)>,
    pub(crate) op_indices: Vec<Option<FastOpIndex>>,

    pub(crate) n: usize,
    pub(crate) loc_ends: Option<(FastOpIndex, FastOpIndex)>,
    pub(crate) var_ends: Vec<Option<(PRel<FastOpIndex>, PRel<FastOpIndex>)>>,

    // Optional bond counting.
    bond_counters: Option<Vec<usize>>,

    // Allocator
    alloc: ALLOC,
}

impl<O: Op + Clone, ALLOC: FastOpAllocator> FastOpsTemplate<O, ALLOC> {
    /// A new manager which can handle nvars and is optimized for nbonds, will allocate memory using
    /// the given allocator.
    pub fn new_from_nvars_and_nbonds_and_alloc(
        nvars: usize,
        nbonds: Option<usize>,
        alloc: ALLOC,
    ) -> Self {
        Self {
            ops: Default::default(),
            op_indices: Default::default(),
            n: 0,
            loc_ends: None,
            var_ends: vec![None; nvars],
            bond_counters: nbonds.map(|nbonds| vec![0; nbonds]),
            alloc,
        }
    }

    /// A new manager which can handle nvars and is optimized for nbonds.
    pub fn new_from_nvars_and_nbonds(nvars: usize, nbonds: Option<usize>) -> Self {
        Self::new_from_nvars_and_nbonds_and_alloc(nvars, nbonds, Default::default())
    }

    /// New manager which can handle nvars
    pub fn new_from_nvars(nvars: usize) -> Self {
        Self::new_from_nvars_and_nbonds(nvars, None)
    }

    /// Make a new Manager from an interator of ops, and number of variables.
    pub fn new_from_ops<I>(nvars: usize, ps_and_ops: I) -> Self
    where
        I: IntoIterator<Item = (usize, O)>,
    {
        let mut man = Self::new_from_nvars(nvars);
        man.clear_and_install_ops(ps_and_ops);
        man
    }

    fn clear_and_install_ops<I>(&mut self, ps_and_ops: I)
    where
        I: IntoIterator<Item = (usize, O)>,
    {
        let nvars = self.var_ends.len();
        let ps_and_ops = ps_and_ops.into_iter().collect::<Vec<_>>();
        if ps_and_ops.is_empty() {
            return;
        }
        let opslen = ps_and_ops.iter().map(|(p, _)| p).max().unwrap() + 1;
        self.ops.clear();
        self.op_indices.resize_with(opslen, || None);
        self.var_ends.clear();
        self.var_ends.resize_with(nvars, || None);
        self.loc_ends = None;

        let last_loc: Option<FastOpIndex> = None;
        let mut last_prels: Vec<Option<PRel<FastOpIndex>>> = self.get_instance();
        last_prels.resize(nvars, None);

        // let mut last_p = None;

        let (last_loc, last_prels) = ps_and_ops.into_iter().fold(
            (last_loc, last_prels),
            |(last_loc, mut last_prels), (p, op): (usize, O)| {
                // TODO enable assertion.
                // assert!(
                //     last_p.map(|last_p| p > last_p).unwrap_or(true),
                //     "P values must be provided in increasing order."
                // );

                // This is the location this op will live in.
                let loc_for_this_op = self.ops.next_index();

                if let Some(last_loc) = last_loc {
                    let last_node = &mut self.ops[last_loc].1;
                    last_node.next_loc = Some(last_loc);
                } else {
                    self.loc_ends = Some((loc_for_this_op, loc_for_this_op))
                }

                let previous_for_vars = op
                    .get_vars()
                    .iter()
                    .cloned()
                    .enumerate()
                    .map(|(relv, v)| {
                        match last_prels[v] {
                            // Set the previous nodes to point to this one.
                            Some(PRel {
                                loc,
                                relv: last_relv,
                            }) => {
                                let node = &mut self.ops[loc].1;
                                node.next_for_vars[last_relv] = Some(PRel {
                                    loc: loc_for_this_op,
                                    relv,
                                });
                            }
                            // If no previous then set the var ends to contain this node.
                            None => {
                                let end = PRel {
                                    loc: loc_for_this_op,
                                    relv,
                                };
                                self.var_ends[v] = Some((end, end));
                            }
                        }
                        // Now save for the next op.
                        let prel = PRel {
                            loc: loc_for_this_op,
                            relv,
                        };
                        last_prels[v] = Some(prel);
                        Some(prel)
                    })
                    .collect();
                let n_opvars = op.get_vars().len();
                let mut node =
                    FastOpNodeTemplate::<O>::new(op, previous_for_vars, smallvec![None; n_opvars]);
                node.previous_loc = last_loc;
                let loc = self.ops.push((p, node));
                debug_assert_eq!(loc, loc_for_this_op);
                self.n += 1;

                (Some(loc_for_this_op), last_prels)
            },
        );
        if let Some((_, loc_end)) = self.loc_ends.as_mut() {
            *loc_end = last_loc.unwrap()
        }
        self.var_ends
            .iter_mut()
            .zip(last_prels.iter().cloned())
            .for_each(|(ends, last_prel)| {
                if let Some((_, v_end)) = ends {
                    *v_end = last_prel.unwrap();
                }
            });
        self.return_instance(last_prels);
    }
}

type LinkVars = SmallVec<[Option<PRel<FastOpIndex>>; 2]>;

/// A node which contains ops for FastOps.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct FastOpNodeTemplate<O: Op, LV = LinkVars>
where
    LV: Index<usize, Output = Option<PRel<FastOpIndex>>>
        + IndexMut<usize, Output = Option<PRel<FastOpIndex>>>,
{
    pub(crate) op: O,
    pub(crate) previous_loc: Option<FastOpIndex>,
    pub(crate) next_loc: Option<FastOpIndex>,
    pub(crate) previous_for_vars: LV,
    pub(crate) next_for_vars: LV,
}

impl<O: Op, LV> FastOpNodeTemplate<O, LV>
where
    LV: Index<usize, Output = Option<PRel<FastOpIndex>>>
        + IndexMut<usize, Output = Option<PRel<FastOpIndex>>>,
{
    fn new(op: O, previous_for_vars: LV, next_for_vars: LV) -> Self {
        // TODO find way to check that the sizes all line up
        // --> need a collections len trait but those are blocked by HKT apparently.
        Self {
            op,
            previous_loc: None,
            next_loc: None,
            previous_for_vars,
            next_for_vars,
        }
    }
}

impl<O: Op + Clone, LV> OpNode<O> for FastOpNodeTemplate<O, LV>
where
    LV: Index<usize, Output = Option<PRel<FastOpIndex>>>
        + IndexMut<usize, Output = Option<PRel<FastOpIndex>>>,
{
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
    /// Last index seen
    last_loc: Option<FastOpIndex>,
    /// Last p seen per var
    last_prels: Vec<Option<PRel<FastOpIndex>>>,
    /// (vars->subvars, subvars)
    subvar_mapping: Option<(Vec<usize>, Vec<usize>)>,
    /// Unfilled vars
    unfilled: usize,
}

impl FastOpMutateArgs {
    fn new<Fact>(nvars: usize, vars: Option<&[usize]>, fact: &mut Fact) -> Self
    where
        Fact: Factory<Vec<usize>> + Factory<Vec<Option<PRel<FastOpIndex>>>>,
    {
        let mut last_prels: Vec<Option<PRel<FastOpIndex>>> = fact.get_instance();
        let last_size = vars.map(|vars| vars.len()).unwrap_or(nvars);
        last_prels.resize(last_size, None);

        FastOpMutateArgs {
            last_loc: None,
            last_prels,
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
        self.last_prels.len()
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

impl<O: Op + Clone, ALLOC: FastOpAllocator> DiagonalSubsection for FastOpsTemplate<O, ALLOC> {
    type Args = FastOpMutateArgs;

    fn mutate_p<T, F>(&mut self, f: F, p: usize, t: T, mut args: Self::Args) -> (T, Self::Args)
    where
        F: Fn(&Self, Option<&Self::Op>, T) -> (Option<Option<Self::Op>>, T),
    {
        let op_ref = self.get_pth_op(p);
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

            let old_op_index = self.op_indices[p].take();
            let old_op_node = old_op_index.map(|loc| self.ops[loc].1);

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
                node_ref.previous_loc = old_op_node.previous_loc;
                node_ref.next_loc = old_op_node.next_loc;
                if let Some(bond_counters) = self.bond_counters.as_mut() {
                    let bond = old_op_node.op.get_bond();
                    bond_counters[bond] -= 1;
                    let bond = node_ref.op.get_bond();
                    bond_counters[bond] += 1;
                }
                self.op_indices[p] = Some(node_ref);
            } else {
                if let Some(node_ref) = old_op_node {
                    debug_assert_eq!(args.last_p, node_ref.previous_p);

                    // Uninstall the old op.
                    // If there's a previous p, point it towards the next one
                    if let Some(last_p) = args.last_p {
                        let node = self.op_indices[last_p].as_mut().unwrap();
                        node.next_p = node_ref.next_p;
                    } else {
                        // No previous p, so we are removing the head. Make necessary adjustments.
                        self.loc_ends = if let Some((head, tail)) = self.loc_ends {
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
                    let next_p_node = node_ref.next_p.and_then(|p| self.op_indices[p].as_mut());
                    if let Some(next_p_node) = next_p_node {
                        next_p_node.previous_p = args.last_p
                    } else {
                        // No next p, so we are removing the tail. Adjust.
                        self.loc_ends = if let Some((head, tail)) = self.loc_ends {
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
                        debug_assert_eq!(
                            args.last_vars[subvar],
                            node_ref.previous_for_vars[relv].map(|prel| prel.p),
                            "P position in args must match op being removed"
                        );
                        debug_assert_eq!(
                            args.last_rels[subvar],
                            node_ref.previous_for_vars[relv].map(|prel| prel.relv),
                            "Relative var in args must match op being removed"
                        );

                        if let Some((prev_p_for_v, prev_rel_indx)) =
                            args.last_vars[subvar].zip(args.last_rels[subvar])
                        {
                            let prev_p_for_v_ref = self.op_indices[prev_p_for_v].as_mut().unwrap();
                            prev_p_for_v_ref.next_for_vars[prev_rel_indx] =
                                node_ref.next_for_vars[relv];
                        } else {
                            // This was the first one, need to edit vars list.
                            self.var_ends[v] = if let Some((head, tail)) = self.var_ends[v] {
                                debug_assert_eq!(head, PRel { loc: p, relv });
                                // If None then we are removing the head and the tail.
                                node_ref.next_for_vars[relv].map(|new_head| {
                                    debug_assert_ne!(tail, PRel { loc: p, relv });
                                    (new_head, tail)
                                })
                            } else {
                                unreachable!()
                            }
                        }

                        // Check the next nodes using this variable.
                        if let Some(PRel {
                            loc: next_p_for_v,
                            relv: next_rel_index,
                        }) = node_ref.next_for_vars[relv]
                        {
                            let next_p_for_v_ref = self.op_indices[next_p_for_v].as_mut().unwrap();
                            next_p_for_v_ref.previous_for_vars[next_rel_index] = args.last_vars
                                [subvar]
                                .zip(args.last_rels[subvar])
                                .map(PRel::from);
                        } else {
                            self.var_ends[v] = if let Some((head, tail)) = self.var_ends[v] {
                                debug_assert_eq!(tail, PRel { loc: p, relv });
                                // If None then we are removing the head and the tail.
                                node_ref.previous_for_vars[relv].map(|new_tail| {
                                    debug_assert_ne!(head, PRel { loc: p, relv });
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
                                let prev_node_for_v =
                                    self.op_indices[prev_p_for_v].as_ref().unwrap();
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
                            if let Some(prel) = prev {
                                let prev_p = prel.loc;
                                let prev_rel = prel.relv;
                                let prev_node = self.op_indices[prev_p].as_mut().unwrap();
                                debug_assert_eq!(prev_node.get_op_ref().get_vars()[prev_rel], *v);
                                prev_node.next_for_vars[prev_rel] = Some(PRel::from((p, relv)));
                            } else {
                                self.var_ends[*v] = if let Some((head, tail)) = self.var_ends[*v] {
                                    debug_assert!(head.p >= p);
                                    Some((PRel { loc: p, relv }, tail))
                                } else {
                                    Some((PRel { loc: p, relv }, PRel { loc: p, relv }))
                                }
                            }
                        });

                    nexts
                        .iter()
                        .zip(new_op.get_vars().iter())
                        .enumerate()
                        .for_each(|(relv, (next, v))| {
                            if let Some(PRel {
                                loc: next_p,
                                relv: next_rel,
                            }) = next
                            {
                                let next_node = self.op_indices[*next_p].as_mut().unwrap();
                                debug_assert_eq!(next_node.get_op_ref().get_vars()[*next_rel], *v);
                                next_node.previous_for_vars[*next_rel] =
                                    Some(PRel { loc: p, relv });
                            } else {
                                self.var_ends[*v] = if let Some((head, tail)) = self.var_ends[*v] {
                                    debug_assert!(tail.p <= p);
                                    Some((head, PRel { loc: p, relv }))
                                } else {
                                    Some((PRel { loc: p, relv }, PRel { loc: p, relv }))
                                }
                            }
                        });

                    let mut node_ref = FastOpNodeTemplate::new(new_op, prevs, nexts);
                    node_ref.previous_loc = args.last_p;
                    node_ref.next_loc = if let Some(last_p) = args.last_p {
                        let last_p_node = self.op_indices[last_p].as_ref().unwrap();
                        last_p_node.next_p
                    } else if let Some((head, _)) = self.loc_ends {
                        Some(head)
                    } else {
                        None
                    };

                    // Based on what these were set to, adjust the p_ends and neighboring nodes.
                    if let Some(prev) = node_ref.previous_loc {
                        let prev_node = self.op_indices[prev].as_mut().unwrap();
                        prev_node.next_p = Some(p);
                    } else {
                        self.loc_ends = if let Some((head, tail)) = self.loc_ends {
                            debug_assert!(head >= p);
                            Some((p, tail))
                        } else {
                            Some((p, p))
                        }
                    };

                    if let Some(next) = node_ref.next_loc {
                        let next_node = self.op_indices[next].as_mut().unwrap();
                        next_node.previous_p = Some(p);
                    } else {
                        self.loc_ends = if let Some((head, tail)) = self.loc_ends {
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
                    self.op_indices[p] = Some(node_ref);
                    self.n += 1;
                }
            }
        }

        if let Some(op) = self.op_indices[p].as_ref().map(|r| &r.op) {
            op.get_vars()
                .iter()
                .cloned()
                .enumerate()
                .for_each(|(relv, v)| {
                    if let Some(subvar) = args.var_to_subvar(v) {
                        args.last_vars[subvar] = Some(p);
                        args.last_rels[subvar] = Some(relv);
                    }
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
        if pend > self.op_indices.len() {
            self.op_indices.resize(pend, None)
        };
        let args = match args {
            Some(args) => args,
            None => {
                let args = self.get_empty_args(SubvarAccess::All);
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
        F: Fn(&Self, &Self::Op, Self::OpIndex, T) -> (Option<Option<Self::Op>>, T),
    {
        if pend > self.op_indices.len() {
            self.op_indices.resize(pend, None)
        };

        let mut args = match args {
            Some(args) => args,
            None => {
                let args = self.get_empty_args(SubvarAccess::All);
                self.fill_args_at_p(pstart, args)
            }
        };

        let last_prels = &mut args.last_prels;
        let subvars = args
            .subvar_mapping
            .as_ref()
            .map(|(_, subvars)| subvars.as_slice());

        if let Some(vars) = subvars {
            let mut p_heap: BinaryHeap<CmpBy<Reverse<usize>, FastOpIndex>> = self.get_instance();

            vars.iter()
                .cloned()
                .zip(last_prels)
                .for_each(|(var, prel)| {
                    if let Some(prel) = prel {
                        let node = self.get_node_ref_loc(prel.loc);
                        let next_prel = self.get_next_prel_for_rel_var(prel.relv, node);

                        // last_p is final before pstart.
                        let last_p = self.get_p_for_opindex(prel.loc);
                        debug_assert!(last_p < pstart);
                        debug_assert!(next_prel
                            .map(|next| self.get_p_for_opindex(next.loc) >= pstart)
                            .unwrap_or(true));

                        // Push the p just inside the pstart-pend range.
                        if let Some(PRel { loc: next_loc, .. }) = next_prel {
                            let next_p = self.get_p_for_opindex(next_loc);
                            p_heap.push(CmpBy::new(Reverse(next_p), next_loc));
                        }
                    } else if let Some((start, _)) = self.var_ends[var] {
                        let start_p = self.get_p_for_opindex(start.loc);
                        debug_assert!(start_p >= pstart);
                        p_heap.push(CmpBy::new(Reverse(start_p), start.loc));
                    }
                });

            // pheap includes ops which leave the set of subvars, it's up to f(...) to not make
            // changes to those. TODO add this to description.

            // Pop from heap, move, repeat.
            while let Some(p) = p_heap.pop().map(|rp| rp.0) {
                if p > pend {
                    break;
                }
                // Pop duplicates.
                while p_heap
                    .peek()
                    .map(|rp| rp.0)
                    .map(|rp| rp <= p)
                    .unwrap_or(false)
                {
                    let popped = p_heap.pop();
                    debug_assert_eq!(Some(Reverse(p)), popped);
                }
                debug_assert!(p_heap.iter().all(|rp| rp.0 > p));

                let node = self.get_node_ref(p).unwrap();
                // Add next ps
                let op = node.get_op_ref();
                // Current op must have at least 1 var which appears in subvars.
                debug_assert!(op
                    .get_vars()
                    .iter()
                    .cloned()
                    .any(|v| args.var_to_subvar(v).is_some()));

                op.get_vars()
                    .iter()
                    .cloned()
                    .enumerate()
                    .filter_map(|(relv, v)| args.var_to_subvar(v).map(|_| relv))
                    .for_each(|relv| {
                        if let Some(prel) = self.get_next_prel_for_rel_var(relv, node) {
                            p_heap.push(Reverse(prel.loc));
                        }
                    });
                // While the last_vars and last_rels should be correct (since all within subvars)
                // it's possible that the last_p is incorrect since it was out of the subvars. To
                // work around this we restrict f to not be able to remove ops.
                let last_p = self.get_previous_p(node);
                // If there's a disagreement is must be because the previous op had none of the
                // focused subvars.
                debug_assert!(
                    {
                        if last_p.is_none() {
                            args.last_p.is_none()
                        } else {
                            true
                        }
                    },
                    "The graph claims there are no prior ops but the args claim otherwise."
                );
                debug_assert!(
                    {
                        if last_p != args.last_p {
                            let last_node = self.get_node_ref(last_p.unwrap()).unwrap();
                            last_node
                                .get_op_ref()
                                .get_vars()
                                .iter()
                                .cloned()
                                .all(|v| args.var_to_subvar(v).is_none())
                        } else {
                            true
                        }
                    },
                    "Args and graph disagree on last_p even though last op overlaps with subvars."
                );

                args.last_p = last_p;

                let ret = self.mutate_p(|s, op, t| f(s, op.unwrap(), p, t), p, t, args);
                t = ret.0;
                args = ret.1;
            }
            self.return_instance(p_heap);
        } else {
            // Find starting position.
            let mut p = self.loc_ends.and_then(|(start, pend)| {
                if pstart <= start {
                    Some(start)
                } else if start > pend {
                    None
                } else {
                    // Find the first p with an op.
                    self.op_indices[pstart..]
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
                    let ret =
                        self.mutate_p(|s, op, t| f(s, op.unwrap(), node_p, t), node_p, t, args);
                    t = ret.0;
                    args = ret.1;

                    let node = self.op_indices[node_p].as_ref().unwrap();
                    p = node.next_p;
                }
            }
        }
        self.return_args(args);

        t
    }

    fn get_empty_args(&mut self, vars: SubvarAccess<FastOpMutateArgs>) -> FastOpMutateArgs {
        match vars {
            SubvarAccess::All => {
                let mut args = FastOpMutateArgs::new(self.get_nvars(), None, self);
                // Can subtract off vars with no ops. nvars - noops = nvars_with_ops
                args.unfilled = self.var_ends.iter().filter(|end| end.is_some()).count();
                args
            }
            SubvarAccess::Varlist(vars) => {
                let mut args = FastOpMutateArgs::new(self.get_nvars(), Some(vars), self);
                // Can subtract off vars with no ops. nvars - noops = nvars_with_ops
                args.unfilled = vars.iter().filter(|v| self.var_ends[**v].is_some()).count();
                args
            }
            SubvarAccess::Args(mut args) => {
                // Count how many need to be set (are None).
                args.unfilled = (0..args.n_subvars())
                    .filter(|subvar| {
                        let v = args.subvar_to_var(*subvar);
                        args.last_prels[*subvar].is_none() && self.var_ends[v].is_some()
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
                    if args.last_loc.is_none() {
                        args.last_loc = self.op_indices[p];
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
                                if args.last_prels[subvar].is_none() {
                                    args.last_prels[subvar] = Some(PRel {
                                        loc: self.op_indices[p].unwrap(),
                                        relv,
                                    });
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
                        .filter_map(|(v, prel)| prel.as_ref().map(|prel| (*v, *prel)))
                        .for_each(|(v, prel)| {
                            let subvar = args.var_to_subvar(v);
                            // Only set vars which we are looking at.
                            if let Some(subvar) = subvar {
                                if args.last_prels[subvar].is_none() {
                                    args.last_prels[subvar] = Some(prel);
                                    args.unfilled -= 1;
                                }
                            }
                        });
                    args.last_loc = node.previous_loc;
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
        It: IntoIterator<Item = Option<FastOpIndex>>,
    {
        let psel = p;
        let iter_and_set = |mut loc_check: FastOpIndex,
                            var: usize,
                            mut relv: usize,
                            subvar: usize,
                            args: &mut Self::Args| {
            loop {
                let pcheck = self.get_p_for_opindex(loc_check);
                debug_assert!(pcheck < p, "Hints must point to ops before p");
                let node = self.get_node_ref_loc(loc_check);
                let next = self
                    .get_next_prel_for_rel_var(relv, node)
                    .unwrap_or_else(|| self.var_ends[var].unwrap().0);
                let nextp = self.get_p_for_opindex(next.loc);
                // If psel in (pcheck, next.p)
                if p_crosses(pcheck, nextp, psel) {
                    // Leave as None if wraps around.
                    if pcheck < psel {
                        args.last_prels[subvar] = Some(PRel {
                            loc: loc_check,
                            relv,
                        });
                    }
                    break;
                } else {
                    loc_check = next.loc;
                    relv = next.relv;
                }
            }
        };

        let set_using = |prel: PRel<FastOpIndex>, subvar: usize, args: &mut Self::Args| {
            let node = self.get_node_ref_loc(prel.loc);
            let prel = node.previous_for_vars[prel.relv];
            args.last_prels[subvar] = prel;
        };

        // Need to find an op for each var that has ops on worldline.
        hint.into_iter()
            .zip(vars.iter().cloned())
            .enumerate()
            .for_each(
                |(subvar, (loc_hint, var)): (usize, (Option<FastOpIndex>, usize))| {
                    let var_start: Option<PRel<FastOpIndex>> =
                        self.var_ends[var].map(|(prel, _)| prel);
                    let phint = loc_hint.map(|loc| self.get_p_for_opindex(loc));
                    let can_use_hint_iter = phint.map(|phint| phint < p).unwrap_or(false);
                    let can_use_start_iter = var_start
                        .map(|prel| self.get_p_for_opindex(prel.loc) < p)
                        .unwrap_or(false);
                    let use_exact: Option<PRel<FastOpIndex>> = match (loc_hint, phint, var_start) {
                        (Some(loc_hint), Some(phint), _) if phint == p => {
                            let node = self.get_node_ref_loc(loc_hint);
                            let relv = node
                                .get_op_ref()
                                .index_of_var(var)
                                .expect("Gave a hint with an op with the wrong variable.");
                            Some(PRel {
                                loc: loc_hint,
                                relv,
                            })
                        }
                        (_, _, Some(prel)) if self.get_p_for_opindex(prel.loc) == p => Some(prel),
                        _ => None,
                    };

                    if let Some(use_exact) = use_exact {
                        set_using(use_exact, subvar, args)
                    } else {
                        // Neither the hint nor the start line up exactly, use iteration.
                        match (loc_hint.zip(phint), var_start) {
                            // Nothing available.
                            (None, None) => {}

                            // Only one available
                            (None, Some(prel)) => {
                                if can_use_start_iter {
                                    iter_and_set(prel.loc, var, prel.relv, subvar, args)
                                }
                            }

                            // Both available
                            (Some((loc_hint, phint)), Some(prel)) => {
                                let node = self.get_node_ref_loc(loc_hint);
                                let relv = node
                                    .get_op_ref()
                                    .index_of_var(var)
                                    .expect("Gave a hint with an op with the wrong variable.");
                                match (can_use_hint_iter, can_use_start_iter) {
                                    (false, false) => {}
                                    (true, false) => {
                                        iter_and_set(loc_hint, var, relv, subvar, args)
                                    }
                                    (false, true) => {
                                        iter_and_set(prel.loc, var, prel.relv, subvar, args)
                                    }
                                    (true, true) => {
                                        let prel_p = self.get_p_for_opindex(prel.loc);
                                        let prel = if phint < prel_p {
                                            PRel {
                                                loc: loc_hint,
                                                relv,
                                            }
                                        } else {
                                            prel
                                        };
                                        iter_and_set(prel.loc, var, prel.relv, subvar, args)
                                    }
                                }
                            }

                            // Impossible
                            (Some(_), None) => {
                                panic!("Gave a hint for a variable with no ops!")
                            }
                        }
                    }
                },
            );

        // Set last_p to the first p found.
        args.last_loc = (0..psel).rev().find_map(|p| self.get_opindex_for_p(p));
    }

    fn return_args(&mut self, args: Self::Args) {
        self.return_instance(args.last_prels);
        if let Some((vars_to_subvars, subvars_to_vars)) = args.subvar_mapping {
            self.return_instance(vars_to_subvars);
            self.return_instance(subvars_to_vars);
        }
    }

    fn get_propagated_substate_with_hint<It>(
        &self,
        p: usize,
        substate: &mut [bool],
        state: &[bool],
        vars: &[usize],
        hint: It,
    ) where
        It: IntoIterator<Item = Option<FastOpIndex>>,
    {
        let psel = p;
        let iter_and_set = |mut loccheck: Self::OpIndex,
                            var: usize,
                            mut relv: usize,
                            subvar: usize,
                            substate: &mut [bool]| {
            loop {
                let pcheck = self.get_p_for_opindex(loccheck);
                debug_assert!(pcheck < p);
                let node = self.get_node_ref_loc(loccheck);
                let next = self
                    .get_next_prel_for_rel_var(relv, node)
                    .unwrap_or_else(|| self.var_ends[var].unwrap().0);
                // If psel in (pcheck, next.p)
                let nextp = self.get_p_for_opindex(next.loc);
                if p_crosses(pcheck, nextp, psel) {
                    // Leave as None if wraps around.
                    if pcheck < nextp {
                        substate[subvar] = node.get_op_ref().get_outputs()[relv];
                    }
                    break;
                } else {
                    loccheck = next.loc;
                    relv = next.relv;
                }
            }
        };

        let set_using =
            |loc_hint: FastOpIndex, relv: usize, subvar: usize, substate: &mut [bool]| {
                debug_assert_eq!(self.get_p_for_opindex(loc_hint), p);
                let node = self.get_node_ref_loc(loc_hint);
                substate[subvar] = node.get_op_ref().get_inputs()[relv];
            };

        // Need to find an op for each var that has ops on worldline.
        hint.into_iter()
            .zip(vars.iter().cloned())
            .enumerate()
            .for_each(|(subvar, it)| {
                // Help the type system.
                let (loc_hint, var): (Option<FastOpIndex>, usize) = it;
                debug_assert!(
                    loc_hint
                        .map(|loc_hint| self
                            .get_node_ref_loc(loc_hint)
                            .get_op_ref()
                            .index_of_var(var)
                            .is_some())
                        .unwrap_or(true),
                    "Hints must point to ops with relevant variable."
                );
                substate[subvar] = state[var];
                let var_start: Option<PRel<FastOpIndex>> = self.var_ends[var].map(|(prel, _)| prel);
                let loc_hint = loc_hint.and_then(|mut loc_hint: FastOpIndex| {
                    let mut node = self.get_node_ref_loc(loc_hint);
                    let mut relv = node
                        .get_op_ref()
                        .index_of_var(var)
                        .expect("Hints must point to ops with relevant variables");
                    let mut phint = self.get_p_for_opindex(loc_hint);
                    while phint >= p {
                        let prev = self.get_previous_prel_for_rel_var(relv, node)?;
                        loc_hint = prev.loc;
                        relv = prev.relv;
                        phint = self.get_p_for_opindex(loc_hint);
                        node = self.get_node_ref_loc(loc_hint);
                    }
                    Some(loc_hint)
                });
                let phint = loc_hint.map(|loc_hint| self.get_p_for_opindex(loc_hint));
                let can_use_hint_iter = phint.map(|phint| phint < p).unwrap_or(false);
                let var_start_p = var_start.map(|prel| self.get_p_for_opindex(prel.loc));
                let can_use_start_iter = var_start_p.map(|pp| pp < p).unwrap_or(false);
                let use_exact = match (loc_hint.zip(phint), var_start.zip(var_start_p)) {
                    (Some((loc_hint, phint)), _) if phint == p => {
                        let node = self.get_node_ref_loc(loc_hint);
                        let relv = node
                            .get_op_ref()
                            .index_of_var(var)
                            .expect("Gave a hint with an op with the wrong variable.");
                        Some(PRel {
                            loc: loc_hint,
                            relv,
                        })
                    }
                    (_, Some((prel, prel_p))) if prel_p == p => Some(prel),
                    _ => None,
                };

                if let Some(use_exact) = use_exact {
                    set_using(use_exact.loc, use_exact.relv, subvar, substate)
                } else {
                    // Neither the hint nor the start line up exactly, use iteration.
                    match (loc_hint, var_start) {
                        // Nothing available.
                        (None, None) => {}

                        // Only one available
                        (None, Some(prel)) => {
                            if can_use_start_iter {
                                iter_and_set(prel.loc, var, prel.relv, subvar, substate)
                            }
                        }

                        // Both available
                        (Some(loc_hint), Some(prel)) => {
                            let node = self.get_node_ref_loc(loc_hint);
                            let relv = node
                                .get_op_ref()
                                .index_of_var(var)
                                .expect("Gave a hint with an op with the wrong variable.");
                            match (can_use_hint_iter, can_use_start_iter) {
                                (false, false) => {}
                                (true, false) => {
                                    iter_and_set(loc_hint, var, relv, subvar, substate)
                                }
                                (false, true) => {
                                    iter_and_set(prel.loc, var, prel.relv, subvar, substate)
                                }
                                (true, true) => {
                                    let phint = self.get_p_for_opindex(loc_hint);
                                    let prel_p = self.get_p_for_opindex(prel.loc);
                                    let prel = if phint < prel_p {
                                        PRel {
                                            loc: loc_hint,
                                            relv,
                                        }
                                    } else {
                                        prel
                                    };
                                    iter_and_set(prel.loc, var, prel.relv, subvar, substate)
                                }
                            }
                        }

                        // Impossible
                        (Some(_), None) => {
                            panic!("Gave a hint for a variable with no ops!")
                        }
                    }
                }
            });
    }
}

fn p_crosses(pstart: usize, pend: usize, psel: usize) -> bool {
    match (pstart, pend) {
        (pstart, pend) if pstart < pend => (pstart < psel) && (psel <= pend),
        (pstart, pend) if pstart > pend => !((pend < psel) && (psel <= pstart)),
        // Only one var. pstart == pend
        _ => true,
    }
}

impl<O: Op + Clone, ALLOC: FastOpAllocator> DiagonalUpdater for FastOpsTemplate<O, ALLOC> {
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
        self.op_indices[min(pstart, self.op_indices.len())..min(pend, self.op_indices.len())]
            .iter()
            .try_fold(t, |t, op| f(self, op.as_ref().map(|op| op.get_op_ref()), t))
    }

    fn try_iterate_ops<F, T, V>(&self, pstart: usize, pend: usize, mut t: T, f: F) -> Result<T, V>
    where
        F: Fn(&Self, &Self::Op, usize, T) -> Result<T, V>,
    {
        // Find starting position.
        let mut p = self.loc_ends.and_then(|(start, pend)| {
            if pstart <= start {
                Some(start)
            } else if start > pend {
                None
            } else {
                // Find the first p with an op.
                self.op_indices[pstart..]
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
                let node = self.op_indices[node_p].as_ref().unwrap();
                t = f(self, node.get_op_ref(), node_p, t)?;
                p = node.next_p;
            }
        }
        Ok(t)
    }
}

impl<O: Op + Clone, ALLOC: FastOpAllocator> HeatBathDiagonalUpdater for FastOpsTemplate<O, ALLOC> {}

impl<O: Op + Clone, ALLOC: FastOpAllocator> OpContainerConstructor for FastOpsTemplate<O, ALLOC> {
    fn new(nvars: usize) -> Self {
        Self::new_from_nvars(nvars)
    }

    fn new_with_bonds(nvars: usize, nbonds: usize) -> Self {
        Self::new_from_nvars_and_nbonds(nvars, Some(nbonds))
    }
}

impl<O: Op + Clone, ALLOC: FastOpAllocator> OpContainer for FastOpsTemplate<O, ALLOC> {
    type Op = O;
    type OpIndex = FastOpIndex;

    fn get_cutoff(&self) -> usize {
        self.op_indices.len()
    }

    fn set_cutoff(&mut self, cutoff: usize) {
        if cutoff > self.op_indices.len() {
            self.op_indices.resize(cutoff, None)
        }
    }

    fn get_n(&self) -> usize {
        self.n
    }

    fn get_nvars(&self) -> usize {
        self.var_ends.len()
    }

    fn get_pth_op(&self, p: usize) -> Option<&Self::Op> {
        if p < self.op_indices.len() {
            self.op_indices[p].as_ref().map(|opnode| &opnode.op)
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

    fn itime_fold<F, T>(&self, state: &mut [bool], fold_fn: F, init: T) -> T
    where
        F: Fn(T, &[bool]) -> T,
    {
        let (t, _) = (0..self.get_cutoff()).fold((init, state), |(acc, state), p| {
            let t = fold_fn(acc, state);
            if let Some(op) = self.get_pth_op(p) {
                op.get_vars()
                    .iter()
                    .cloned()
                    .enumerate()
                    .for_each(|(relv, v)| {
                        debug_assert_eq!(op.get_inputs()[relv], state[v]);
                        state[v] = op.get_outputs()[relv];
                    });
            }
            (t, state)
        });
        t
    }
    fn get_op(&self, index: FastOpIndex) -> &Self::Op {
        &self.ops[index].1
    }

    fn get_op_mut(&mut self, index: FastOpIndex) -> &mut Self::Op {
        &mut self.ops[index].1
    }

    fn get_p_for_opindex(&self, index: FastOpIndex) -> usize {
        self.ops[index].0
    }

    fn get_opindex_for_p(&self, p: usize) -> Option<FastOpIndex> {
        self.op_indices[p]
    }

    fn opindex_to_usize(&self, index: Self::OpIndex) -> usize {
        index.index
    }

    fn usize_to_opindex(&self, index: usize) -> Self::OpIndex {
        FastOpIndex { index }
    }
}

impl<O: Op + Clone, ALLOC: FastOpAllocator> LoopUpdater for FastOpsTemplate<O, ALLOC> {
    type Node = FastOpNodeTemplate<O>;

    fn get_node_ref(&self, p: usize) -> Option<&Self::Node> {
        self.op_indices[p].map(|index| &self.op_indices[index].1)
    }

    fn get_node_mut(&mut self, p: usize) -> Option<&mut Self::Node> {
        self.op_indices[p].map(|index| &mut self.op_indices[index].1)
    }

    fn get_node_ref_loc(&self, loc: FastOpIndex) -> &Self::Node {
        todo!()
    }

    fn get_node_mut_loc(&mut self, loc: FastOpIndex) -> &mut Self::Node {
        todo!()
    }

    fn get_first_loc(&self) -> Option<FastOpIndex> {
        todo!()
    }

    fn get_last_loc(&self) -> Option<FastOpIndex> {
        todo!()
    }

    fn get_first_p(&self) -> Option<usize> {
        self.loc_ends.map(|(p, _)| p)
    }

    fn get_last_p(&self) -> Option<usize> {
        self.loc_ends.map(|(_, p)| p)
    }

    fn get_first_prel_for_var(&self, var: usize) -> Option<PRel<FastOpIndex>> {
        self.var_ends[var].map(|(start, _)| start)
    }

    fn get_last_prel_for_var(&self, var: usize) -> Option<PRel<FastOpIndex>> {
        self.var_ends[var].map(|(_, end)| end)
    }

    fn get_previous_loc(&self, node: &Self::Node) -> Option<FastOpIndex> {
        todo!()
    }

    fn get_next_loc(&self, node: &Self::Node) -> Option<FastOpIndex> {
        todo!()
    }

    fn get_previous_p(&self, node: &Self::Node) -> Option<usize> {
        node.previous_loc
    }

    fn get_next_p(&self, node: &Self::Node) -> Option<usize> {
        node.next_loc
    }

    fn get_previous_prel_for_rel_var(
        &self,
        relvar: usize,
        node: &Self::Node,
    ) -> Option<PRel<FastOpIndex>> {
        node.previous_for_vars[relvar]
    }

    fn get_next_prel_for_rel_var(
        &self,
        relvar: usize,
        node: &Self::Node,
    ) -> Option<PRel<FastOpIndex>> {
        node.next_for_vars[relvar]
    }

    // TODO review
    // fn get_nth_p(&self, n: usize) -> usize {
    //     let n = n % self.n;
    //     let init = self.p_ends.map(|(head, _)| head).unwrap();
    //     (0..n).fold(init, |p, _| self.ops[p].as_ref().unwrap().next_p.unwrap())
    // }
}

impl<O: Op + Clone, ALLOC: FastOpAllocator> Factory<Vec<bool>> for FastOpsTemplate<O, ALLOC> {
    fn get_instance(&mut self) -> Vec<bool> {
        self.alloc.get_instance()
    }

    fn return_instance(&mut self, t: Vec<bool>) {
        self.alloc.return_instance(t)
    }
}

impl<O: Op + Clone, ALLOC: FastOpAllocator> Factory<Vec<usize>> for FastOpsTemplate<O, ALLOC> {
    fn get_instance(&mut self) -> Vec<usize> {
        self.alloc.get_instance()
    }

    fn return_instance(&mut self, t: Vec<usize>) {
        self.alloc.return_instance(t)
    }
}

impl<O: Op + Clone, ALLOC: FastOpAllocator> Factory<Vec<Option<usize>>>
    for FastOpsTemplate<O, ALLOC>
{
    fn get_instance(&mut self) -> Vec<Option<usize>> {
        self.alloc.get_instance()
    }

    fn return_instance(&mut self, t: Vec<Option<usize>>) {
        self.alloc.return_instance(t)
    }
}

impl<O: Op + Clone, ALLOC: FastOpAllocator> Factory<Vec<OpSide>> for FastOpsTemplate<O, ALLOC> {
    fn get_instance(&mut self) -> Vec<OpSide> {
        self.alloc.get_instance()
    }

    fn return_instance(&mut self, t: Vec<OpSide>) {
        self.alloc.return_instance(t)
    }
}
impl<O: Op + Clone, ALLOC: FastOpAllocator> Factory<Vec<Leg>> for FastOpsTemplate<O, ALLOC> {
    fn get_instance(&mut self) -> Vec<Leg> {
        self.alloc.get_instance()
    }

    fn return_instance(&mut self, t: Vec<Leg>) {
        self.alloc.return_instance(t)
    }
}

impl<O: Op + Clone, ALLOC: FastOpAllocator> Factory<Vec<f64>> for FastOpsTemplate<O, ALLOC> {
    fn get_instance(&mut self) -> Vec<f64> {
        self.alloc.get_instance()
    }

    fn return_instance(&mut self, t: Vec<f64>) {
        self.alloc.return_instance(t)
    }
}
impl<O: Op + Clone, ALLOC: FastOpAllocator> Factory<BondContainer<usize>>
    for FastOpsTemplate<O, ALLOC>
{
    fn get_instance(&mut self) -> BondContainer<usize> {
        self.alloc.get_instance()
    }

    fn return_instance(&mut self, t: BondContainer<usize>) {
        self.alloc.return_instance(t)
    }
}
impl<O: Op + Clone, ALLOC: FastOpAllocator> Factory<BondContainer<VarPos>>
    for FastOpsTemplate<O, ALLOC>
{
    fn get_instance(&mut self) -> BondContainer<VarPos> {
        self.alloc.get_instance()
    }

    fn return_instance(&mut self, t: BondContainer<VarPos>) {
        self.alloc.return_instance(t)
    }
}

impl<O: Op + Clone, ALLOC: FastOpAllocator> Factory<BinaryHeap<CmpBy<Reverse<usize>, FastOpIndex>>>
    for FastOpsTemplate<O, ALLOC>
{
    fn get_instance(&mut self) -> BinaryHeap<CmpBy<Reverse<usize>, FastOpIndex>> {
        self.alloc.get_instance()
    }

    fn return_instance(&mut self, t: BinaryHeap<CmpBy<Reverse<usize>, FastOpIndex>>) {
        self.alloc.return_instance(t)
    }
}

impl<O: Op + Clone, ALLOC: FastOpAllocator> Factory<Vec<FastOpIndex>>
    for FastOpsTemplate<O, ALLOC>
{
    fn get_instance(&mut self) -> Vec<FastOpIndex> {
        self.alloc.get_instance()
    }

    fn return_instance(&mut self, t: Vec<FastOpIndex>) {
        self.alloc.return_instance(t)
    }
}
impl<O: Op + Clone, ALLOC: FastOpAllocator> Factory<Vec<Option<FastOpIndex>>>
    for FastOpsTemplate<O, ALLOC>
{
    fn get_instance(&mut self) -> Vec<Option<FastOpIndex>> {
        self.alloc.get_instance()
    }

    fn return_instance(&mut self, t: Vec<Option<FastOpIndex>>) {
        self.alloc.return_instance(t)
    }
}

impl<O: Op + Clone, ALLOC: FastOpAllocator> Factory<Vec<Option<PRel<FastOpIndex>>>>
    for FastOpsTemplate<O, ALLOC>
{
    fn get_instance(&mut self) -> Vec<Option<PRel<FastOpIndex>>> {
        self.alloc.get_instance()
    }

    fn return_instance(&mut self, t: Vec<Option<PRel<FastOpIndex>>>) {
        self.alloc.return_instance(t)
    }
}

impl<O: Op + Clone, ALLOC: FastOpAllocator> ClusterUpdater for FastOpsTemplate<O, ALLOC> {}

impl<O: Op + Clone, ALLOC: FastOpAllocator> RvbUpdater for FastOpsTemplate<O, ALLOC> {
    fn constant_ops_on_var(&self, var: usize, ps: &mut Vec<FastOpIndex>) {
        let mut p_and_rel = self.get_first_prel_for_var(var);
        while let Some(PRel {
            loc: node_loc,
            relv: node_relv,
        }) = p_and_rel
        {
            let node = self.get_node_ref_loc(node_loc);
            debug_assert_eq!(node.get_op_ref().get_vars()[node_relv], var);
            if node.get_op_ref().is_constant() {
                ps.push(node_loc);
            }
            p_and_rel = self.get_next_prel_for_rel_var(node_relv, node);
        }
    }

    fn spin_flips_on_var(&self, var: usize, ps: &mut Vec<FastOpIndex>) {
        let mut p_and_rel = self.get_first_prel_for_var(var);
        while let Some(PRel {
            loc: node_loc,
            relv: node_relv,
        }) = p_and_rel
        {
            let node = self.get_node_ref(node_loc);
            let op = node.get_op_ref();
            debug_assert_eq!(op.get_vars()[node_relv], var);
            if op.get_inputs()[node_relv] != op.get_outputs()[node_relv] {
                ps.push(node_loc)
            };
            p_and_rel = self.get_next_prel_for_rel_var(node_relv, node);
        }
    }
}

impl<O: Op, ALLOC: FastOpAllocator> IsingManager for FastOpsTemplate<O, ALLOC> {}
impl<O: Op, ALLOC: FastOpAllocator> QmcManager for FastOpsTemplate<O, ALLOC> {}
