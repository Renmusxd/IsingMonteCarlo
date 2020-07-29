use crate::parallel_tempering::StateGetter;
use crate::sse::fast_ops::*;
use crate::sse::qmc_graph::{DefaultQMCGraph, VecEdge};
use crate::sse::qmc_types::{Op, SubState, Vars};
use rand::Rng;
use serde::*;
use std::marker::PhantomData;

macro_rules! serialize_to_writer {
    ($graph:expr, $writer:expr) => {
        {
            let graph: $crate::sse::serialization::SerializeQMCGraph<_> = $graph.into();
            serde_cbor::to_writer($writer, &graph)
        }
    };
}

macro_rules! deserialize_from_reader {
    ($R:ident, $reader:expr) => {
        {
            let graph: serde_cbor::Result<$crate::sse::serialization::SerializeQMCGraph<$R>> = serde_cbor::from_reader($reader);
            graph.map(|g| g.into())
        }
    };
}

#[derive(Serialize, Deserialize)]
pub struct SerializeQMCGraph<R: Rng + Clone> {
    edges: Vec<(VecEdge, f64)>,
    transverse: f64,
    state: Option<Vec<bool>>,
    cutoff: usize,
    op_manager: Option<SerializeFastOps>,
    twosite_energy_offset: f64,
    singlesite_energy_offset: f64,

    #[serde(bound(deserialize = "Option<R>: Deserialize<'de>"))]
    #[serde(bound(serialize = "Option<R>: Serialize"))]
    rng: Option<R>,
    // We dont need to save the whole things
    nvars: usize,
    state_updates_size: usize,
}

#[derive(Serialize, Deserialize)]
pub struct SerializeFastOps {
    ops: Vec<Option<SerializeFastOpNode>>,
    n: usize,
    p_ends: Option<(usize, usize)>,
    var_ends: Vec<Option<(usize, usize)>>,
}

#[derive(Serialize, Deserialize)]
pub struct SerializeFastOpNode {
    op: SerializeOp,
    previous_p: Option<usize>,
    next_p: Option<usize>,
    previous_for_vars: LinkVars,
    next_for_vars: LinkVars,
}

#[derive(Serialize, Deserialize)]
pub struct SerializeOp {
    pub vars: Vars,
    pub bond: usize,
    pub inputs: SubState,
    pub outputs: SubState,
}

impl<R: Rng + Clone> Into<SerializeQMCGraph<R>> for &DefaultQMCGraph<R> {
    fn into(self) -> SerializeQMCGraph<R> {
        SerializeQMCGraph {
            edges: self.get_edges().clone(),
            transverse: self.get_transverse(),
            state: Some(self.get_state_ref().to_vec()),
            cutoff: self.get_cutoff(),
            op_manager: self.get_op_manager().as_ref().map(|m| m.into()),
            twosite_energy_offset: self.get_twosite_energy_offset(),
            singlesite_energy_offset: self.get_singlesite_energy_offset(),
            rng: self.get_rng().clone(),
            nvars: self.get_nvars(),
            state_updates_size: self.get_state_updates().len(),
        }
    }
}

impl<R: Rng + Clone> Into<DefaultQMCGraph<R>> for SerializeQMCGraph<R> {
    fn into(self) -> DefaultQMCGraph<R> {
        DefaultQMCGraph {
            edges: self.edges,
            transverse: self.transverse,
            state: self.state,
            cutoff: self.cutoff,
            op_manager: self.op_manager.map(|m| m.into()),
            twosite_energy_offset: self.twosite_energy_offset,
            singlesite_energy_offset: self.singlesite_energy_offset,
            rng: self.rng,
            phantom: PhantomData,
            vars: (0 .. self.nvars).collect(),
            state_updates: Vec::with_capacity(self.state_updates_size)
        }
    }
}

impl Into<SerializeFastOps> for &FastOps {
    fn into(self) -> SerializeFastOps {
        SerializeFastOps {
            ops: self
                .ops
                .iter()
                .map(|op| op.as_ref().map(|op| op.into()))
                .collect(),
            n: self.n,
            p_ends: self.p_ends,
            var_ends: self.var_ends.clone(),
        }
    }
}

impl Into<FastOps> for SerializeFastOps {
    fn into(self) -> FastOps {
        FastOps {
            ops: self
                .ops
                .into_iter()
                .map(|op| op.map(|op| op.into()))
                .collect(),
            n: self.n,
            p_ends: self.p_ends,
            var_ends: self.var_ends,
            frontier: Some(vec![]),
            interior_frontier: Some(vec![]),
            boundaries: Some(vec![]),
            flips: Some(vec![]),
            last_vars_alloc: Some(vec![]),
        }
    }
}

impl Into<SerializeFastOpNode> for &FastOpNode {
    fn into(self) -> SerializeFastOpNode {
        SerializeFastOpNode {
            op: (&self.op).into(),
            previous_p: self.previous_p,
            next_p: self.next_p,
            previous_for_vars: self.previous_for_vars.clone(),
            next_for_vars: self.next_for_vars.clone(),
        }
    }
}

impl Into<FastOpNode> for SerializeFastOpNode {
    fn into(self) -> FastOpNode {
        FastOpNode {
            op: self.op.into(),
            previous_p: self.previous_p,
            next_p: self.next_p,
            previous_for_vars: self.previous_for_vars,
            next_for_vars: self.next_for_vars,
        }
    }
}

impl Into<SerializeOp> for &Op {
    fn into(self) -> SerializeOp {
        SerializeOp {
            vars: self.vars.clone(),
            bond: self.bond,
            inputs: self.inputs.clone(),
            outputs: self.outputs.clone(),
        }
    }
}

impl Into<Op> for SerializeOp {
    fn into(self) -> Op {
        Op {
            vars: self.vars,
            bond: self.bond,
            inputs: self.inputs,
            outputs: self.outputs,
        }
    }
}

#[cfg(test)]
mod serialize_tests {
    use super::*;
    use rand::SeedableRng;
    use rand_isaac::IsaacRng;

    #[test]
    fn test_ser_de() {
        let edges = vec![
            ((0, 1), 1.0),
            ((1, 2), 1.0),
            ((0, 2), 1.0),
        ];
        let transverse = 1.0;
        let cutoff = 3;
        let rng = IsaacRng::seed_from_u64(1234);
        let mut graph = DefaultQMCGraph::<IsaacRng>::new_with_rng(edges, transverse, cutoff, rng, None);
        graph.timesteps(100, 1.0);

        let mut buf : Vec<u8> = vec![];
        serialize_to_writer!(&graph, &mut buf).unwrap();

        println!("{:?}", buf);

        let _new_graph = deserialize_from_reader!(IsaacRng, &*buf).unwrap();
    }
}