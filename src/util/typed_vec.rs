#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};
use std::ops::{Index, IndexMut};

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub(crate) struct TypedVec<K, V>
where
    K: From<usize> + Into<usize>,
{
    data: Vec<V>,
    k: std::marker::PhantomData<K>,
}

impl<K, V> Default for TypedVec<K, V>
where
    K: From<usize> + Into<usize>,
{
    fn default() -> Self {
        Self {
            data: Default::default(),
            k: Default::default(),
        }
    }
}

impl<K, V> TypedVec<K, V>
where
    K: From<usize> + Into<usize>,
{
    pub(crate) fn push(&mut self, v: V) -> K {
        let index = self.data.len().into();
        self.data.push(v);
        index
    }
    pub(crate) fn pop(&mut self) -> Option<V> {
        self.data.pop()
    }
    pub(crate) fn clear(&mut self) {
        self.data.clear()
    }
}

impl<K, V> Index<K> for TypedVec<K, V>
where
    K: From<usize> + Into<usize>,
{
    type Output = V;

    fn index(&self, index: K) -> &Self::Output {
        &self.data[index.into()]
    }
}

impl<K, V> IndexMut<K> for TypedVec<K, V>
where
    K: From<usize> + Into<usize>,
{
    fn index_mut(&mut self, index: K) -> &mut Self::Output {
        &mut self.data[index.into()]
    }
}
