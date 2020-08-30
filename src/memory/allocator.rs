#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};

/// A factory which produces Ts.
pub trait Factory<T> {
    /// Get an instance of T
    fn get_instance(&mut self) -> T;
    /// Return an instance of T
    fn return_instance(&mut self, t: T);
}

/// Reset an instance while preserving its memory.
pub trait Reset {
    /// Reset the instance, keep the memory location.
    fn reset(&mut self);
}

impl<T> Reset for Vec<T> {
    fn reset(&mut self) {
        self.clear()
    }
}

#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub(crate) struct Allocator<T: Default + Reset> {
    instances: Vec<T>,
}

impl<T: Default + Reset> Factory<T> for Allocator<T> {
    fn get_instance(&mut self) -> T {
        match self.instances.pop() {
            None => T::default(),
            Some(t) => t,
        }
    }

    fn return_instance(&mut self, mut t: T) {
        t.reset();
        self.instances.push(t)
    }
}

#[derive(Debug)]
pub(crate) struct StackTuplizer<A, B> {
    a_vec: Vec<A>,
    b_vec: Vec<B>,
}

impl<A, B> Extend<(A, B)> for StackTuplizer<A, B> {
    fn extend<T: IntoIterator<Item = (A, B)>>(&mut self, iter: T) {
        iter.into_iter().for_each(|(a, b)| {
            self.a_vec.push(a);
            self.b_vec.push(b);
        })
    }
}

impl<A, B> StackTuplizer<A, B> {
    pub(crate) fn new<T: ?Sized + Factory<Vec<A>> + Factory<Vec<B>>>(t: &mut T) -> Self {
        let a_vec: Vec<A> = t.get_instance();
        let b_vec: Vec<B> = t.get_instance();
        Self { a_vec, b_vec }
    }

    pub(crate) fn dissolve<T: ?Sized + Factory<Vec<A>> + Factory<Vec<B>>>(self, t: &mut T) {
        t.return_instance(self.a_vec);
        t.return_instance(self.b_vec);
    }

    pub(crate) fn push(&mut self, tup: (A, B)) {
        let (a, b) = tup;
        self.a_vec.push(a);
        self.b_vec.push(b);
    }

    pub(crate) fn pop(&mut self) -> Option<(A, B)> {
        self.a_vec.pop().zip(self.b_vec.pop())
    }

    pub(crate) fn get(&self, index: usize) -> Option<(&A, &B)> {
        self.a_vec.get(index).zip(self.b_vec.get(index))
    }

    pub(crate) fn at(&self, index: usize) -> (&A, &B) {
        (&self.a_vec[index], &self.b_vec[index])
    }

    pub(crate) fn set(&mut self, index: usize, tup: (A, B)) {
        let (a, b) = tup;
        self.a_vec[index] = a;
        self.b_vec[index] = b;
    }

    pub(crate) fn resize_each<F, G>(&mut self, size: usize, f: F, g: G)
    where
        F: Fn() -> A,
        G: Fn() -> B,
    {
        self.a_vec.resize_with(size, f);
        self.b_vec.resize_with(size, g);
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = (&A, &B)> {
        self.a_vec.iter().zip(self.b_vec.iter())
    }

    pub(crate) fn iter_mut(&mut self) -> impl Iterator<Item = (&mut A, &mut B)> {
        self.a_vec.iter_mut().zip(self.b_vec.iter_mut())
    }
}
