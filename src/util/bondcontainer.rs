use crate::util::allocator::Reset;
use rand::prelude::*;
#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

/// A HashSet with random sampling.
#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct BondContainer<T: Clone + Into<usize>> {
    map: Vec<Option<usize>>,
    keys: Vec<(T, f64)>,
    total_weight: f64,
}

impl<T: Clone + Into<usize>> Reset for BondContainer<T> {
    fn reset(&mut self) {
        self.clear();
    }
}

impl<T: Clone + Into<usize>> BondContainer<T> {
    /// Return sum of all weights in BondContainer.
    pub fn get_total_weight(&self) -> f64 {
        self.total_weight
    }

    /// Get a random weighted entry.
    pub fn get_random<R: Rng>(&self, mut r: R) -> Option<&(T, f64)> {
        if self.keys.is_empty() {
            None
        } else {
            // Choose a key to remove
            let mut p = r.gen_range(0. ..self.total_weight);
            let mut i = 0;
            while i < self.keys.len() {
                p -= self.keys[i].1;
                if p <= 0. {
                    break;
                }
                i += 1
            }
            Some(&self.keys[i])
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

    fn remove_index(&mut self, keys_index: usize) -> (T, f64) {
        // Move key to last position.
        let last_indx = self.keys.len() - 1;
        self.keys.swap(keys_index, last_indx);
        // Update address
        let bond_number = self.keys[keys_index].0.clone().into();
        let old_indx = self.map[bond_number].as_mut().unwrap();
        *old_indx = keys_index;
        // Remove key
        let (out, weight) = self.keys.pop().unwrap();
        self.map[out.clone().into()] = None;
        self.total_weight -= weight;
        self.correct_total_weight();

        (out, weight)
    }

    fn correct_total_weight(&mut self) {
        if self.total_weight < 0. {
            self.total_weight = 0.;
            debug_assert!({
                let weight_sum = self.keys.iter().map(|(_, w)| *w).sum::<f64>();
                let valid = weight_sum.abs() <= std::f64::EPSILON * self.keys.len() as f64;
                if !valid {
                    println!("Weight sum: {}", weight_sum);
                }
                valid
            });
        }
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

    /// Get the weight of the value if it's in the container.
    pub fn get_weight(&self, value: &T) -> Option<f64> {
        let t = value.clone().into();
        if t >= self.map.len() {
            None
        } else {
            self.map[t].map(|indx| self.keys[indx].1)
        }
    }

    /// Insert an element. Returns true if the element is new.
    pub fn insert(&mut self, value: T, weight: f64) -> bool {
        let entry_index = value.clone().into();
        if entry_index >= self.map.len() {
            self.map.resize(entry_index + 1, None);
        }
        match self.map[entry_index] {
            Some(index) => {
                let old_weight = self.keys[index].1;
                self.keys[index].1 = weight;
                self.total_weight += weight - old_weight;
                self.correct_total_weight();
                false
            }
            None => {
                self.map[entry_index] = Some(self.keys.len());
                self.keys.push((value, weight));
                self.total_weight += weight;
                true
            }
        }
    }

    /// Clear the set
    pub fn clear(&mut self) {
        let keys = &mut self.keys;
        let map = &mut self.map;
        keys.iter().map(|(t, _)| t).for_each(|k| {
            let bond = k.clone().into();
            map[bond] = None
        });
        keys.clear();
        self.total_weight = 0.;
    }

    /// Get number of elements in set.
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// Iterate through items.
    pub fn iter(&self) -> impl Iterator<Item = &(T, f64)> {
        self.keys.iter()
    }
}
