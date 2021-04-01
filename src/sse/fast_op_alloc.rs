use crate::sse::qmc_types::{Leg, OpSide};
use crate::sse::VarPos;
use crate::util::allocator::{Allocator, Factory};
use crate::util::bondcontainer::BondContainer;
#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};
use std::cmp::Reverse;
use std::collections::BinaryHeap;

/// An allocator for the FastOp group of structs.
pub trait FastOpAllocator:
    Default
    + Factory<Vec<usize>>
    + Factory<Vec<bool>>
    + Factory<Vec<OpSide>>
    + Factory<Vec<Leg>>
    + Factory<Vec<Option<usize>>>
    + Factory<Vec<f64>>
    + Factory<BondContainer<usize>>
    + Factory<BondContainer<VarPos>>
    + Factory<BinaryHeap<Reverse<usize>>>
    + Clone
{
}

/// An allocator for the FastOps which saves memory requested for later.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct DefaultFastOpAllocator {
    usize_alloc: Allocator<Vec<usize>>,
    bool_alloc: Allocator<Vec<bool>>,
    opside_alloc: Allocator<Vec<OpSide>>,
    leg_alloc: Allocator<Vec<Leg>>,
    option_usize_alloc: Allocator<Vec<Option<usize>>>,
    f64_alloc: Allocator<Vec<f64>>,
    bond_container_alloc: Allocator<BondContainer<usize>>,
    bond_container_varpos_alloc: Allocator<BondContainer<VarPos>>,
    binary_heap_alloc: Allocator<BinaryHeap<Reverse<usize>>>,
}

impl Default for DefaultFastOpAllocator {
    fn default() -> Self {
        // Set bounds to make sure there are not "leaks"
        Self {
            usize_alloc: Allocator::new_with_max_in_flight(10),
            bool_alloc: Allocator::new_with_max_in_flight(2),
            opside_alloc: Allocator::new_with_max_in_flight(1),
            leg_alloc: Allocator::new_with_max_in_flight(1),
            option_usize_alloc: Allocator::new_with_max_in_flight(4),
            f64_alloc: Allocator::new_with_max_in_flight(1),
            bond_container_alloc: Allocator::new_with_max_in_flight(2),
            bond_container_varpos_alloc: Allocator::new_with_max_in_flight(2),
            binary_heap_alloc: Allocator::new_with_max_in_flight(1),
        }
    }
}

impl Factory<Vec<bool>> for DefaultFastOpAllocator {
    fn get_instance(&mut self) -> Vec<bool> {
        self.bool_alloc.get_instance()
    }

    fn return_instance(&mut self, t: Vec<bool>) {
        self.bool_alloc.return_instance(t)
    }
}

impl Factory<Vec<usize>> for DefaultFastOpAllocator {
    fn get_instance(&mut self) -> Vec<usize> {
        self.usize_alloc.get_instance()
    }

    fn return_instance(&mut self, t: Vec<usize>) {
        self.usize_alloc.return_instance(t)
    }
}

impl Factory<Vec<Option<usize>>> for DefaultFastOpAllocator {
    fn get_instance(&mut self) -> Vec<Option<usize>> {
        self.option_usize_alloc.get_instance()
    }

    fn return_instance(&mut self, t: Vec<Option<usize>>) {
        self.option_usize_alloc.return_instance(t)
    }
}

impl Factory<Vec<OpSide>> for DefaultFastOpAllocator {
    fn get_instance(&mut self) -> Vec<OpSide> {
        self.opside_alloc.get_instance()
    }

    fn return_instance(&mut self, t: Vec<OpSide>) {
        self.opside_alloc.return_instance(t)
    }
}
impl Factory<Vec<Leg>> for DefaultFastOpAllocator {
    fn get_instance(&mut self) -> Vec<Leg> {
        self.leg_alloc.get_instance()
    }

    fn return_instance(&mut self, t: Vec<Leg>) {
        self.leg_alloc.return_instance(t)
    }
}

impl Factory<Vec<f64>> for DefaultFastOpAllocator {
    fn get_instance(&mut self) -> Vec<f64> {
        self.f64_alloc.get_instance()
    }

    fn return_instance(&mut self, t: Vec<f64>) {
        self.f64_alloc.return_instance(t)
    }
}
impl Factory<BondContainer<usize>> for DefaultFastOpAllocator {
    fn get_instance(&mut self) -> BondContainer<usize> {
        self.bond_container_alloc.get_instance()
    }

    fn return_instance(&mut self, t: BondContainer<usize>) {
        self.bond_container_alloc.return_instance(t)
    }
}
impl Factory<BondContainer<VarPos>> for DefaultFastOpAllocator {
    fn get_instance(&mut self) -> BondContainer<VarPos> {
        self.bond_container_varpos_alloc.get_instance()
    }

    fn return_instance(&mut self, t: BondContainer<VarPos>) {
        self.bond_container_varpos_alloc.return_instance(t)
    }
}

impl Factory<BinaryHeap<Reverse<usize>>> for DefaultFastOpAllocator {
    fn get_instance(&mut self) -> BinaryHeap<Reverse<usize>> {
        self.binary_heap_alloc.get_instance()
    }

    fn return_instance(&mut self, t: BinaryHeap<Reverse<usize>>) {
        self.binary_heap_alloc.return_instance(t)
    }
}

impl FastOpAllocator for DefaultFastOpAllocator {}

/// An allocator which can either forward to another allocator or provide os memory management.
#[derive(Debug, Default, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct SwitchableFastOpAllocator<ALLOC: FastOpAllocator = DefaultFastOpAllocator> {
    alloc: Option<ALLOC>,
}

impl<ALLOC: FastOpAllocator> SwitchableFastOpAllocator<ALLOC> {
    /// Construct a new allocator with an optional forward to another allocator.
    pub fn new(alloc: Option<ALLOC>) -> Self {
        Self { alloc }
    }
}

impl<ALLOC: FastOpAllocator> Factory<Vec<bool>> for SwitchableFastOpAllocator<ALLOC> {
    fn get_instance(&mut self) -> Vec<bool> {
        self.alloc
            .as_mut()
            .map(|a| a.get_instance())
            .unwrap_or_else(Default::default)
    }

    fn return_instance(&mut self, t: Vec<bool>) {
        if let Some(a) = self.alloc.as_mut() {
            a.return_instance(t)
        }
    }
}

impl<ALLOC: FastOpAllocator> Factory<Vec<usize>> for SwitchableFastOpAllocator<ALLOC> {
    fn get_instance(&mut self) -> Vec<usize> {
        self.alloc
            .as_mut()
            .map(|a| a.get_instance())
            .unwrap_or_else(Default::default)
    }

    fn return_instance(&mut self, t: Vec<usize>) {
        if let Some(a) = self.alloc.as_mut() {
            a.return_instance(t)
        }
    }
}

impl<ALLOC: FastOpAllocator> Factory<Vec<Option<usize>>> for SwitchableFastOpAllocator<ALLOC> {
    fn get_instance(&mut self) -> Vec<Option<usize>> {
        self.alloc
            .as_mut()
            .map(|a| a.get_instance())
            .unwrap_or_else(Default::default)
    }

    fn return_instance(&mut self, t: Vec<Option<usize>>) {
        if let Some(a) = self.alloc.as_mut() {
            a.return_instance(t)
        }
    }
}

impl<ALLOC: FastOpAllocator> Factory<Vec<OpSide>> for SwitchableFastOpAllocator<ALLOC> {
    fn get_instance(&mut self) -> Vec<OpSide> {
        self.alloc
            .as_mut()
            .map(|a| a.get_instance())
            .unwrap_or_else(Default::default)
    }

    fn return_instance(&mut self, t: Vec<OpSide>) {
        if let Some(a) = self.alloc.as_mut() {
            a.return_instance(t)
        }
    }
}
impl<ALLOC: FastOpAllocator> Factory<Vec<Leg>> for SwitchableFastOpAllocator<ALLOC> {
    fn get_instance(&mut self) -> Vec<Leg> {
        self.alloc
            .as_mut()
            .map(|a| a.get_instance())
            .unwrap_or_else(Default::default)
    }

    fn return_instance(&mut self, t: Vec<Leg>) {
        if let Some(a) = self.alloc.as_mut() {
            a.return_instance(t)
        }
    }
}

impl<ALLOC: FastOpAllocator> Factory<Vec<f64>> for SwitchableFastOpAllocator<ALLOC> {
    fn get_instance(&mut self) -> Vec<f64> {
        self.alloc
            .as_mut()
            .map(|a| a.get_instance())
            .unwrap_or_else(Default::default)
    }

    fn return_instance(&mut self, t: Vec<f64>) {
        if let Some(a) = self.alloc.as_mut() {
            a.return_instance(t)
        }
    }
}
impl<ALLOC: FastOpAllocator> Factory<BondContainer<usize>> for SwitchableFastOpAllocator<ALLOC> {
    fn get_instance(&mut self) -> BondContainer<usize> {
        self.alloc
            .as_mut()
            .map(|a| a.get_instance())
            .unwrap_or_else(Default::default)
    }

    fn return_instance(&mut self, t: BondContainer<usize>) {
        if let Some(a) = self.alloc.as_mut() {
            a.return_instance(t)
        }
    }
}
impl<ALLOC: FastOpAllocator> Factory<BondContainer<VarPos>> for SwitchableFastOpAllocator<ALLOC> {
    fn get_instance(&mut self) -> BondContainer<VarPos> {
        self.alloc
            .as_mut()
            .map(|a| a.get_instance())
            .unwrap_or_else(Default::default)
    }

    fn return_instance(&mut self, t: BondContainer<VarPos>) {
        if let Some(a) = self.alloc.as_mut() {
            a.return_instance(t)
        }
    }
}

impl<ALLOC: FastOpAllocator> Factory<BinaryHeap<Reverse<usize>>>
    for SwitchableFastOpAllocator<ALLOC>
{
    fn get_instance(&mut self) -> BinaryHeap<Reverse<usize>> {
        self.alloc
            .as_mut()
            .map(|a| a.get_instance())
            .unwrap_or_else(Default::default)
    }

    fn return_instance(&mut self, t: BinaryHeap<Reverse<usize>>) {
        if let Some(a) = self.alloc.as_mut() {
            a.return_instance(t)
        }
    }
}

impl<ALLOC: FastOpAllocator> FastOpAllocator for SwitchableFastOpAllocator<ALLOC> {}
