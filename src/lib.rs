mod error;

pub use crate::error::Error;

use std::{
    cmp::Ordering,
    collections::{vec_deque::Iter, HashSet, TryReserveError, VecDeque},
    fmt,
    hash::{Hash, Hasher},
    ops::{Index, RangeBounds},
};

#[derive(Clone, Default)]
pub struct UniqueVecDeque<T> {
    deque: VecDeque<T>,
    set: HashSet<T>,
}

impl<T> UniqueVecDeque<T> {
    #[inline]
    pub fn new() -> Self {
        Self {
            deque: VecDeque::new(),
            set: HashSet::new(),
        }
    }

    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            deque: VecDeque::with_capacity(capacity),
            set: HashSet::with_capacity(capacity),
        }
    }

    #[inline]
    pub fn get(&self, index: usize) -> Option<&T> {
        self.deque.get(index)
    }

    #[inline]
    pub fn swap(&mut self, i: usize, j: usize) {
        self.deque.swap(i, j)
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.deque.capacity()
    }

    #[inline]
    pub fn reserve_exact(&mut self, additional: usize) {
        self.deque.reserve_exact(additional);
    }

    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        self.deque.reserve(additional);
    }

    #[inline]
    pub fn try_reserve_exact(&mut self, additional: usize) -> Result<(), TryReserveError> {
        self.deque.try_reserve_exact(additional)
    }

    #[inline]
    pub fn try_reserve(&mut self, additional: usize) -> Result<(), TryReserveError> {
        self.deque.try_reserve(additional)
    }

    #[inline]
    pub fn shrink_to_fit(&mut self) {
        self.deque.shrink_to_fit();
    }

    #[inline]
    pub fn shrink_to(&mut self, min_capacity: usize) {
        self.deque.shrink_to(min_capacity);
    }

    #[inline]
    pub fn truncate(&mut self, len: usize)
    where
        T: Eq + Hash + PartialEq,
    {
        for v in self.deque.range(len..) {
            self.set.remove(v);
        }
        self.deque.truncate(len);
    }

    #[inline]
    pub fn iter(&self) -> Iter<'_, T> {
        self.deque.iter()
    }

    #[inline]
    pub fn as_slices(&self) -> (&[T], &[T]) {
        self.deque.as_slices()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.deque.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.deque.is_empty()
    }

    #[inline]
    pub fn range<R>(&self, range: R) -> Iter<'_, T>
    where
        R: RangeBounds<usize>,
    {
        self.deque.range(range)
    }

    #[inline]
    pub fn clear(&mut self) {
        self.deque.clear();
        self.set.clear();
    }

    #[inline]
    pub fn contains(&self, x: &T) -> bool
    where
        T: Eq + Hash + PartialEq<T>,
    {
        self.set.contains(x)
    }

    #[inline]
    pub fn front(&self) -> Option<&T> {
        self.deque.front()
    }

    #[inline]
    pub fn back(&self) -> Option<&T> {
        self.deque.back()
    }

    pub fn pop_front(&mut self) -> Option<T>
    where
        T: Eq + Hash + PartialEq<T>,
    {
        let v = self.deque.pop_front()?;
        self.set.remove(&v);
        Some(v)
    }

    pub fn pop_back(&mut self) -> Option<T>
    where
        T: Eq + Hash + PartialEq<T>,
    {
        let v = self.deque.pop_back()?;
        self.set.remove(&v);
        Some(v)
    }

    pub fn push_front(&mut self, value: T) -> Result<(), Error>
    where
        T: Clone + Eq + Hash + PartialEq<T>,
    {
        if !self.set.insert(value.clone()) {
            return Err(Error::Duplicated);
        }
        self.deque.push_front(value);
        Ok(())
    }

    pub fn push_back(&mut self, value: T) -> Result<(), Error>
    where
        T: Clone + Eq + Hash + PartialEq<T>,
    {
        if !self.set.insert(value.clone()) {
            return Err(Error::Duplicated);
        }
        self.deque.push_back(value);
        Ok(())
    }

    pub fn insert(&mut self, index: usize, value: T) -> Result<(), Error>
    where
        T: Clone + Eq + Hash + PartialEq<T>,
    {
        if !self.set.insert(value.clone()) {
            return Err(Error::Duplicated);
        }
        self.deque.insert(index, value);
        Ok(())
    }

    pub fn remove(&mut self, index: usize) -> Option<T>
    where
        T: Clone + Eq + Hash + PartialEq<T>,
    {
        let v = self.deque.remove(index)?;
        self.set.remove(&v);
        Some(v)
    }

    #[inline]
    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&T) -> bool,
    {
        self.deque.retain(|elem| f(elem));
        self.set.retain(f);
    }

    #[inline]
    pub fn resize_with(&mut self, new_len: usize, generator: impl FnMut() -> T) {
        self.deque.resize_with(new_len, generator);
    }

    #[inline]
    pub fn rotate_left(&mut self, mid: usize) {
        self.deque.rotate_left(mid);
    }

    #[inline]
    pub fn rotate_right(&mut self, mid: usize) {
        self.deque.rotate_right(mid);
    }

    #[inline]
    pub fn binary_search(&self, x: &T) -> Result<usize, usize>
    where
        T: Ord,
    {
        self.deque.binary_search(x)
    }

    #[inline]
    pub fn binary_search_by<'a, F>(&'a self, f: F) -> Result<usize, usize>
    where
        F: FnMut(&'a T) -> Ordering,
    {
        self.deque.binary_search_by(f)
    }

    #[inline]
    pub fn binary_search_by_key<'a, B, F>(&'a self, b: &B, f: F) -> Result<usize, usize>
    where
        F: FnMut(&'a T) -> B,
        B: Ord,
    {
        self.deque.binary_search_by_key(b, f)
    }

    #[inline]
    pub fn partition_point<P>(&self, pred: P) -> usize
    where
        P: FnMut(&T) -> bool,
    {
        self.deque.partition_point(pred)
    }
}

impl<T: PartialEq> PartialEq for UniqueVecDeque<T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.deque.eq(&other.deque)
    }
}

impl<T: Eq> Eq for UniqueVecDeque<T> {}

impl<T: PartialOrd> PartialOrd for UniqueVecDeque<T> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.deque.partial_cmp(&other.deque)
    }
}

impl<T: Ord> Ord for UniqueVecDeque<T> {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.deque.cmp(&other.deque)
    }
}

impl<T: Hash> Hash for UniqueVecDeque<T> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.deque.hash(state)
    }
}

impl<T> Index<usize> for UniqueVecDeque<T> {
    type Output = T;

    #[inline]
    fn index(&self, index: usize) -> &T {
        self.deque.index(index)
    }
}

impl<T: fmt::Debug> fmt::Debug for UniqueVecDeque<T> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.deque.fmt(f)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::hash_map::DefaultHasher;

    use maplit::hashset;

    use super::*;

    #[test]
    fn unique_vec_deque_new() {
        let deque = UniqueVecDeque::<u8>::new();
        assert_eq!(deque.deque.len(), 0);
        assert_eq!(deque.set.len(), 0);
    }

    #[test]
    fn unique_vec_deque_with_capacity() {
        const CAPACITY: usize = 256;
        let deque = UniqueVecDeque::<u8>::with_capacity(CAPACITY);
        assert_eq!(deque.deque.capacity(), CAPACITY);
        assert!(CAPACITY <= deque.set.capacity());
    }

    #[test]
    fn unique_vec_deque_with_get() {
        const VALUE: usize = 127;
        let deque = UniqueVecDeque {
            deque: [VALUE].into(),
            set: hashset! { VALUE },
        };
        assert_eq!(deque.get(0), Some(&VALUE));
        assert_eq!(deque.get(1), None);
    }

    #[test]
    fn unique_vec_deque_with_swap() {
        let mut deque = UniqueVecDeque {
            deque: [0, 1, 2, 3].into(),
            set: hashset! { 0, 1, 2, 3 },
        };
        deque.swap(0, 1);
        assert_eq!(
            UniqueVecDeque {
                deque: [1, 0, 2, 3].into(),
                set: hashset! { 0, 1, 2, 3 },
            },
            deque
        );
    }

    #[test]
    #[should_panic]
    fn unique_vec_deque_with_swap_should_panic() {
        let mut deque = UniqueVecDeque {
            deque: [0, 1, 2, 3].into(),
            set: hashset! { 0, 1, 2, 3 },
        };
        deque.swap(0, 4);
    }

    #[test]
    fn unique_vec_deque_capacity() {
        const CAPACITY: usize = 256;
        let deque = UniqueVecDeque {
            deque: VecDeque::<u8>::with_capacity(CAPACITY),
            set: HashSet::with_capacity(CAPACITY),
        };
        assert_eq!(deque.capacity(), CAPACITY)
    }

    #[test]
    fn unique_vec_deque_reserve_exact() {
        const ADDITIONAL: usize = 10;
        let mut deque = UniqueVecDeque {
            deque: [0, 1, 2, 3].into(),
            set: hashset! { 0, 1, 2, 3 },
        };
        deque.reserve_exact(ADDITIONAL);
        assert!(4 + ADDITIONAL <= deque.capacity())
    }

    #[test]
    fn unique_vec_deque_reserve() {
        const ADDITIONAL: usize = 10;
        let mut deque = UniqueVecDeque {
            deque: [0, 1, 2, 3].into(),
            set: hashset! { 0, 1, 2, 3 },
        };
        deque.reserve(ADDITIONAL);
        assert_eq!(deque.capacity(), 4 + ADDITIONAL)
    }

    #[test]
    fn unique_vec_deque_try_reserve_exact() {
        const ADDITIONAL: usize = 10;
        let mut deque = UniqueVecDeque {
            deque: [0, 1, 2, 3].into(),
            set: hashset! { 0, 1, 2, 3 },
        };
        assert!(deque.try_reserve_exact(ADDITIONAL).is_ok());
        assert!(4 + ADDITIONAL <= deque.capacity())
    }

    #[test]
    fn unique_vec_deque_try_reserve() {
        const ADDITIONAL: usize = 10;
        let mut deque = UniqueVecDeque {
            deque: [0, 1, 2, 3].into(),
            set: hashset! { 0, 1, 2, 3 },
        };
        assert!(deque.try_reserve_exact(ADDITIONAL).is_ok());
        assert_eq!(deque.capacity(), 4 + ADDITIONAL)
    }

    #[test]
    fn unique_vec_deque_shrink_to_fit() {
        let mut deque = UniqueVecDeque {
            deque: {
                let mut deque = VecDeque::with_capacity(10);
                deque.push_back(0);
                deque.push_back(1);
                deque.push_back(2);
                deque.push_back(3);
                deque
            },
            set: hashset! { 0, 1, 2, 3 },
        };
        deque.shrink_to_fit();
        assert_eq!(deque.capacity(), 4);
    }

    #[test]
    fn unique_vec_deque_shrink_to() {
        let mut deque = UniqueVecDeque {
            deque: {
                let mut deque = VecDeque::with_capacity(10);
                deque.push_back(0);
                deque.push_back(1);
                deque.push_back(2);
                deque.push_back(3);
                deque
            },
            set: hashset! { 0, 1, 2, 3 },
        };
        deque.shrink_to(5);
        assert_eq!(deque.capacity(), 5);
        deque.shrink_to(3);
        assert_eq!(deque.capacity(), 4);
    }

    #[test]
    fn unique_vec_deque_truncate() {
        let mut deque = UniqueVecDeque {
            deque: [0, 1, 2, 3].into(),
            set: hashset! { 0, 1, 2, 3 },
        };
        deque.truncate(2);
        assert_eq!(deque.deque, [0, 1]);
        assert_eq!(deque.set, hashset! { 0, 1 });
    }

    #[test]
    fn unique_vec_deque_iter() {
        let deque = UniqueVecDeque {
            deque: [0, 1, 2, 3].into(),
            set: hashset! { 0, 1, 2, 3 },
        };
        let mut iter = deque.iter();
        assert_eq!(iter.next(), Some(&0));
        assert_eq!(iter.next(), Some(&1));
        assert_eq!(iter.next(), Some(&2));
        assert_eq!(iter.next(), Some(&3));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn unique_vec_deque_as_slices() {
        let deque = UniqueVecDeque {
            deque: {
                let mut deque = VecDeque::new();
                deque.push_front(1);
                deque.push_front(0);
                deque.push_back(2);
                deque.push_back(3);
                deque
            },
            set: hashset! { 0, 1, 2, 3 },
        };
        assert_eq!(deque.as_slices(), (&[0, 1][..], &[2, 3][..]));
    }

    #[test]
    fn unique_vec_deque_len() {
        let deque = UniqueVecDeque {
            deque: [0, 1, 2, 3].into(),
            set: hashset! { 0, 1, 2, 3 },
        };
        assert_eq!(deque.len(), 4);
    }

    #[test]
    fn unique_vec_deque_is_empty() {
        let deque = UniqueVecDeque::<u8>::new();
        assert!(deque.is_empty());

        let deque = UniqueVecDeque {
            deque: [0, 1, 2, 3].into(),
            set: hashset! { 0, 1, 2, 3 },
        };
        assert!(!deque.is_empty());
    }

    #[test]
    fn unique_vec_deque_range() {
        let deque = UniqueVecDeque {
            deque: [0, 1, 2, 3].into(),
            set: hashset! { 0, 1, 2, 3 },
        };
        let mut range = deque.range(1..=2);
        assert_eq!(range.next(), Some(&1));
        assert_eq!(range.next(), Some(&2));
        assert_eq!(range.next(), None);
    }

    #[test]
    fn unique_vec_deque_clear() {
        let mut deque = UniqueVecDeque {
            deque: [0, 1, 2, 3].into(),
            set: hashset! { 0, 1, 2, 3 },
        };
        deque.clear();
        assert_eq!(deque.deque, []);
        assert_eq!(deque.set, hashset! {});
    }

    #[test]
    fn unique_vec_deque_contains() {
        let deque = UniqueVecDeque {
            deque: [0, 1, 2, 3].into(),
            set: hashset! { 0, 1, 2, 3 },
        };
        assert!(deque.contains(&0));
        assert!(deque.contains(&1));
        assert!(deque.contains(&2));
        assert!(deque.contains(&3));
        assert!(!deque.contains(&4));
    }

    #[test]
    fn unique_vec_deque_front() {
        let deque = UniqueVecDeque {
            deque: [0, 1, 2, 3].into(),
            set: hashset! { 0, 1, 2, 3 },
        };
        assert_eq!(deque.front(), Some(&0));
    }

    #[test]
    fn unique_vec_deque_back() {
        let deque = UniqueVecDeque {
            deque: [0, 1, 2, 3].into(),
            set: hashset! { 0, 1, 2, 3 },
        };
        assert_eq!(deque.back(), Some(&3));
    }

    #[test]
    fn unique_vec_deque_pop_front() {
        let mut deque = UniqueVecDeque {
            deque: [0, 1, 2, 3].into(),
            set: hashset! { 0, 1, 2, 3 },
        };
        assert_eq!(deque.pop_front(), Some(0));
    }

    #[test]
    fn unique_vec_deque_pop_back() {
        let mut deque = UniqueVecDeque {
            deque: [0, 1, 2, 3].into(),
            set: hashset! { 0, 1, 2, 3 },
        };
        assert_eq!(deque.pop_back(), Some(3));
    }

    #[test]
    fn unique_vec_deque_push_front() {
        let mut deque = UniqueVecDeque {
            deque: [0, 1, 2, 3].into(),
            set: hashset! { 0, 1, 2, 3 },
        };
        assert!(deque.push_front(0).is_err());
        assert!(deque.push_front(-1).is_ok());
        assert_eq!(deque.deque, [-1, 0, 1, 2, 3]);
        assert_eq!(deque.set, hashset! { -1, 0, 1, 2, 3 });
    }

    #[test]
    fn unique_vec_deque_push_back() {
        let mut deque = UniqueVecDeque {
            deque: [0, 1, 2, 3].into(),
            set: hashset! { 0, 1, 2, 3 },
        };
        assert!(deque.push_back(3).is_err());
        assert!(deque.push_back(4).is_ok());
        assert_eq!(deque.deque, [0, 1, 2, 3, 4]);
        assert_eq!(deque.set, hashset! { 0, 1, 2, 3, 4 });
    }

    #[test]
    fn unique_vec_deque_insert() {
        let mut deque = UniqueVecDeque {
            deque: [0, 1, 2, 3].into(),
            set: hashset! { 0, 1, 2, 3 },
        };
        assert!(deque.insert(1, 3).is_err());
        assert!(deque.insert(1, 4).is_ok());
        assert_eq!(deque.deque, [0, 4, 1, 2, 3]);
        assert_eq!(deque.set, hashset! { 0, 4, 1, 2, 3 });
    }

    #[test]
    fn unique_vec_deque_remove() {
        let mut deque = UniqueVecDeque {
            deque: [0, 1, 2, 3].into(),
            set: hashset! { 0, 1, 2, 3 },
        };
        assert!(deque.remove(4).is_none());
        assert_eq!(deque.remove(3), Some(3));
    }

    #[test]
    fn unique_vec_deque_retain() {
        let mut deque = UniqueVecDeque {
            deque: [0, 1, 2, 3].into(),
            set: hashset! { 0, 1, 2, 3 },
        };
        deque.retain(|v| v % 2 == 0);
        assert_eq!(deque.deque, [0, 2]);
        assert_eq!(deque.set, hashset! { 0, 2 });
    }

    #[test]
    fn unique_vec_deque_index() {
        let deque = UniqueVecDeque {
            deque: [0, 1, 2, 3].into(),
            set: hashset! { 0, 1, 2, 3 },
        };
        assert_eq!(deque.index(0), &0);
        assert_eq!(deque.index(1), &1);
        assert_eq!(deque.index(2), &2);
        assert_eq!(deque.index(3), &3);
    }

    #[test]
    #[should_panic]
    fn unique_vec_deque_index_should_panic() {
        let deque = UniqueVecDeque::<u8>::new();
        deque.index(0);
    }

    #[test]
    fn unique_vec_deque_partition_point() {
        let deque = UniqueVecDeque {
            deque: [0, 1, 2, 3].into(),
            set: hashset! { 0, 1, 2, 3 },
        };

        assert_eq!(deque.partition_point(|&x| x < 2), 2);
    }

    #[test]
    fn unique_vec_deque_eq() {
        let deque = UniqueVecDeque {
            deque: [0, 1, 2, 3].into(),
            set: hashset! { 0, 1, 2, 3 },
        };

        assert!(!deque.eq(&UniqueVecDeque {
            deque: [1, 2, 3, 4].into(),
            set: hashset! { 1, 2, 3, 4 },
        }),);

        assert!(deque.eq(&UniqueVecDeque {
            deque: [0, 1, 2, 3].into(),
            set: hashset! { 0, 1, 2, 3 },
        }),);
    }

    #[test]
    fn unique_vec_deque_cmp() {
        let deque = UniqueVecDeque {
            deque: [0, 1, 2, 3].into(),
            set: hashset! { 0, 1, 2, 3 },
        };

        assert_eq!(
            deque.cmp(&UniqueVecDeque {
                deque: [1, 2, 3, 4].into(),
                set: hashset! { 1, 2, 3, 4 },
            }),
            Ordering::Less
        );

        assert_eq!(
            deque.cmp(&UniqueVecDeque {
                deque: [0, 1, 2, 3].into(),
                set: hashset! { 0, 1, 2, 3 },
            }),
            Ordering::Equal
        );

        assert_eq!(
            deque.cmp(&UniqueVecDeque {
                deque: [-1, 0, 1, 2].into(),
                set: hashset! { -1, 0, 1, 2 },
            }),
            Ordering::Greater
        );
    }

    #[test]
    fn unique_vec_deque_hash() {
        let deque = UniqueVecDeque {
            deque: [0, 1, 2, 3].into(),
            set: hashset! { 0, 1, 2, 3 },
        };
        assert_eq!(
            {
                let mut hasher = DefaultHasher::new();
                deque.hash(&mut hasher);
                hasher.finish()
            },
            {
                let mut hasher = DefaultHasher::new();
                [0, 1, 2, 3].hash(&mut hasher);
                hasher.finish()
            }
        );
    }

    #[test]
    fn unique_vec_deque_debug() {
        let deque = UniqueVecDeque {
            deque: [0, 1, 2, 3].into(),
            set: hashset! { 0, 1, 2, 3 },
        };
        assert_eq!(format!("{deque:?}"), format!("{:?}", [0, 1, 2, 3]));
    }
}
