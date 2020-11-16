use crate::sse::{DiagonalUpdater, LoopUpdater, OpContainer};

/// Args required to mutate, allows flexibility of implementations.
pub trait MutateArgs {
    /// Type for subvar indices, helps distinguish from variables.
    type SubvarIndex: Into<usize> + From<usize>;
    /// Number of subvars.
    fn n_subvars(&self) -> usize;
    /// Map subvars to variables.
    fn subvar_to_var(&self, index: Self::SubvarIndex) -> usize;
    /// Map variables to subvars.
    fn var_to_subvar(&self, var: usize) -> Option<Self::SubvarIndex>;
}

/// Ways to build args using subvars or existing args.
#[derive(Debug)]
pub enum SubvarAccess<'a, Args: MutateArgs> {
    /// Include all vars.
    ALL,
    /// Use list of vars.
    VARLIST(&'a [usize]),
    /// Get vars from existing args.
    ARGS(Args),
}

/// Allows for mutations on subsections of the data.
pub trait DiagonalSubsection: OpContainer + LoopUpdater + DiagonalUpdater {
    /// The type required for mutating p values.
    type Args: MutateArgs;

    /// Mutate a single p value.
    fn mutate_p<T, F>(&mut self, f: F, p: usize, t: T, args: Self::Args) -> (T, Self::Args)
    where
        F: Fn(&Self, Option<&Self::Op>, T) -> (Option<Option<Self::Op>>, T);

    /// Mutate using a restricted prange and args. Allows for running on smaller subsections of
    /// the variables.
    fn mutate_subsection<T, F>(
        &mut self,
        pstart: usize,
        pend: usize,
        t: T,
        f: F,
        args: Option<Self::Args>,
    ) -> T
    where
        F: Fn(&Self, Option<&Self::Op>, T) -> (Option<Option<Self::Op>>, T);

    /// Mutate ops using a restricted prange and args. Allows for running on smaller subsections of
    /// the variables.
    fn mutate_subsection_ops<T, F>(
        &mut self,
        pstart: usize,
        pend: usize,
        t: T,
        f: F,
        args: Option<Self::Args>,
    ) -> T
    where
        F: Fn(&Self, &Self::Op, usize, T) -> (Option<Option<Self::Op>>, T),
    {
        let (t, _) = self.mutate_subsection(
            pstart,
            pend,
            (t, 0),
            |s, op, (t, p)| {
                if let Some(op) = op {
                    let (new_op, t) = f(s, op, p, t);
                    (new_op, (t, p + 1))
                } else {
                    (None, (t, p + 1))
                }
            },
            args,
        );
        t
    }

    /// Get empty args for a subsection of variables, or from an existing set of args (does not clear).
    fn get_empty_args(&mut self, vars: SubvarAccess<Self::Args>) -> Self::Args;

    /// Get the args required for mutation.
    fn fill_args_at_p(&self, p: usize, empty_args: Self::Args) -> Self::Args;

    /// Use a list of ps to make args.
    /// Hint provides up to one ps per var in order.
    fn fill_args_at_p_with_hint<It>(
        &self,
        p: usize,
        args: &mut Self::Args,
        vars: &[usize],
        hint: It,
    ) where
        It: Iterator<Item = Option<usize>>;

    /// Returns arg made for the mutation
    fn return_args(&mut self, args: Self::Args);

    /// Get the substate at p using ps to set values.
    /// Hint provides up to one ps per var in order.
    /// substate should start in p=0 configuration.
    fn get_propagated_substate_with_hint<It>(
        &self,
        p: usize,
        substate: &mut [bool],
        vars: &[usize],
        hint: It,
    ) where
        It: Iterator<Item = Option<usize>>;

    /// Iterate over ops at indices less than or equal to p. Applies function `f_at_p` only to the
    /// op at p. Applies `f` to all other ops above p.
    fn iter_ops_above_p<T, F, G>(&self, p: usize, mut t: T, f: F, f_at_p: G) -> T
    where
        // Applied to each op at or above p until returns false
        // Takes p and op
        F: Fn(usize, &Self::Node, T) -> (T, bool),
        G: Fn(&Self::Node, T) -> (T, bool),
    {
        // Find the most recent node above p. Can set subbstate using inputs at node at p.
        let mut node_p = match self.get_node_ref(p) {
            None if p == 0 => None,
            None => {
                let mut sel_p = p - 1;
                let mut prev_node = self.get_node_ref(sel_p);
                while prev_node.is_none() && sel_p > 0 {
                    sel_p -= 1;
                    prev_node = self.get_node_ref(sel_p);
                }
                prev_node.map(|_| sel_p)
            }
            Some(node) => {
                let (tt, c) = f_at_p(node, t);
                t = tt;
                if c {
                    self.get_previous_p(node)
                } else {
                    None
                }
            }
        };

        // Find previous ops and use their outputs to set substate.
        while let Some(p) = node_p {
            let node = self.get_node_ref(p).unwrap();
            let c = f(p, node, t);
            t = c.0;
            if !c.1 {
                break;
            }
            node_p = self.get_previous_p(node);
        }
        t
    }
}
