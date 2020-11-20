use crate::util::allocator::Reset;
use rand::prelude::*;
#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};

/// A HashSet with random sampling.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct BondContainer<T: Clone + Into<usize>> {
    map: Vec<Option<usize>>,
    keys: Option<Vec<T>>,
}

impl<T: Clone + Into<usize>> Reset for BondContainer<T> {
    fn reset(&mut self) {
        self.clear();
    }
}

impl<T: Clone + Into<usize>> Default for BondContainer<T> {
    fn default() -> Self {
        Self {
            map: Vec::default(),
            keys: Some(Vec::default()),
        }
    }
}

impl<T: Clone + Into<usize>> BondContainer<T> {
    /// Get a random entry from the HashSampler
    pub fn get_random<R: Rng>(&self, mut r: R) -> Option<&T> {
        let keys = self.keys.as_ref().unwrap();
        if keys.is_empty() {
            None
        } else {
            // Choose a key to remove
            let index = r.gen_range(0, keys.len());
            Some(&keys[index])
        }
    }

    /// Pop a random element.
    pub fn pop_random<R: Rng>(&mut self, mut r: R) -> Option<T> {
        let keys = self.keys.as_ref().unwrap();
        if keys.is_empty() {
            None
        } else {
            // Choose a key to remove
            let keys_index = r.gen_range(0, keys.len());
            Some(self.remove_index(keys_index))
        }
    }

    /// Remove a given value.
    pub fn remove(&mut self, value: &T) -> bool {
        let bond_number = value.clone().into();
        let address = self.map[bond_number];
        if let Some(address) = address {
            self.remove_index(address);
            true
        } else {
            false
        }
    }

    fn remove_index(&mut self, keys_index: usize) -> T {
        // Move key to last position.
        let keys = self.keys.as_mut().unwrap();
        let last_indx = keys.len() - 1;
        keys.swap(keys_index, last_indx);
        // Update address
        let bond_number = keys[keys_index].clone().into();
        let old_indx = self.map[bond_number].as_mut().unwrap();
        *old_indx = keys_index;
        // Remove key
        let out = self.keys.as_mut().unwrap().pop().unwrap();
        self.map[out.clone().into()] = None;
        out
    }

    /// Check if a given element has been inserted.
    pub fn contains(&self, value: &T) -> bool {
        let t = value.clone().into();
        if t >= self.map.len() {
            false
        } else {
            self.map[t].is_some()
        }
    }

    /// Insert an element.
    pub fn insert(&mut self, value: T) -> bool {
        let entry_index = value.clone().into();
        if entry_index >= self.map.len() {
            self.map.resize(entry_index + 1, None);
        }
        match self.map[entry_index] {
            Some(_) => false,
            None => {
                let keys = self.keys.as_mut().unwrap();
                self.map[entry_index] = Some(keys.len());
                keys.push(value);
                true
            }
        }
    }

    /// Clear the set
    pub fn clear(&mut self) {
        let mut keys = self.keys.take().unwrap();
        keys.iter().for_each(|k| {
            let bond = k.clone().into();
            self.map[bond] = None
        });
        keys.clear();
        self.keys = Some(keys);
    }

    /// Get number of elements in set.
    pub fn len(&self) -> usize {
        self.keys.as_ref().unwrap().len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.keys.as_ref().unwrap().is_empty()
    }

    /// Iterate through items.
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.keys.as_ref().unwrap().iter()
    }
}
