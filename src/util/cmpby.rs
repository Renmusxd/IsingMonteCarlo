use std::cmp::Ordering;
use std::fmt::{Debug, Formatter};

pub struct CmpBy<T, V>
where
    T: PartialEq + Eq + PartialOrd + Ord,
{
    t: T,
    v: V,
}

impl<T, V> Debug for CmpBy<T, V>
where
    T: PartialEq + Eq + PartialOrd + Ord + Debug,
    V: Debug,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CmpBy")
            .field("t", &self.t)
            .field("v", &self.v)
            .finish()
    }
}

impl<T, V> CmpBy<T, V>
where
    T: PartialEq + Eq + PartialOrd + Ord,
{
    pub fn new(t: T, v: V) -> Self {
        Self { t, v }
    }
}

impl<T, V> PartialEq for CmpBy<T, V>
where
    T: PartialEq + Eq + PartialOrd + Ord,
{
    fn eq(&self, other: &Self) -> bool {
        T::eq(&self.t, &other.t)
    }
}

impl<T, V> PartialOrd for CmpBy<T, V>
where
    T: PartialEq + Eq + PartialOrd + Ord,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        T::partial_cmp(&self.t, &other.t)
    }
}

impl<T, V> Eq for CmpBy<T, V> where T: PartialEq + Eq + PartialOrd + Ord {}

impl<T, V> Ord for CmpBy<T, V>
where
    T: PartialEq + Eq + PartialOrd + Ord,
{
    fn cmp(&self, other: &Self) -> Ordering {
        T::cmp(&self.t, &other.t)
    }
}
