//! Deferred resource release, shared by both backends.
//!
//! `Device::destroy` may be called while submitted work still references the resource, so each is
//! tagged with the submission value that must complete before it is freed.

use std::cell::RefCell;
use std::collections::VecDeque;

/// Resources waiting on a submission value, in issue order.
pub(crate) struct RetirementQueue<R> {
    /// `(value that must complete, resource)`. Values are pushed in issue order and therefore
    /// non-decreasing, so draining a prefix is enough to find everything that has retired.
    entries: RefCell<VecDeque<(u64, R)>>,
}

impl<R> Default for RetirementQueue<R> {
    fn default() -> Self {
        Self {
            entries: RefCell::new(VecDeque::new()),
        }
    }
}

impl<R> RetirementQueue<R> {
    /// Hold `resource` until `pending_until` has completed.
    ///
    /// `pending_until` is the last value issued, so nothing submitted so far can still reference
    /// the resource once it is reached. Zero means nothing has ever been submitted, in which case
    /// nothing can reference it and it is freed at once.
    pub(crate) fn release(&self, pending_until: u64, resource: R, free: impl FnOnce(R)) {
        if pending_until == 0 {
            free(resource);
            return;
        }
        self.entries
            .borrow_mut()
            .push_back((pending_until, resource));
    }

    /// Free everything whose value is at or below `completed`.
    ///
    /// The queue is unborrowed while `free` runs, because freeing a resource can push another
    /// onto this same queue.
    pub(crate) fn collect(&self, completed: u64, mut free: impl FnMut(R)) {
        while let Some(resource) = self.pop_retired(completed) {
            free(resource);
        }
    }

    fn pop_retired(&self, completed: u64) -> Option<R> {
        let mut entries = self.entries.borrow_mut();
        match entries.front() {
            Some(&(value, _)) if value <= completed => {
                entries.pop_front().map(|(_, resource)| resource)
            }
            _ => None,
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.entries.borrow().is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_resource_is_held_until_its_value_completes() {
        let queue = RetirementQueue::default();
        let freed = RefCell::new(Vec::new());
        // Nothing submitted yet, so nothing can reference it.
        queue.release(0, "now", |r| freed.borrow_mut().push(r));
        assert_eq!(*freed.borrow(), ["now"]);
        freed.borrow_mut().clear();

        queue.release(5, "a", |_| unreachable!("held, not freed"));
        queue.release(9, "b", |_| unreachable!("held, not freed"));

        queue.collect(4, |r| freed.borrow_mut().push(r));
        assert!(freed.borrow().is_empty(), "nothing has completed yet");

        queue.collect(5, |r| freed.borrow_mut().push(r));
        assert_eq!(*freed.borrow(), ["a"], "only the first value has completed");

        queue.collect(100, |r| freed.borrow_mut().push(r));
        assert_eq!(*freed.borrow(), ["a", "b"]);
        assert!(queue.is_empty());
    }

    /// Freeing a resource can hand another one back to the same queue, so `collect` must not be
    /// holding the borrow while it runs.
    #[test]
    fn freeing_may_release_onto_the_same_queue() {
        let queue = RetirementQueue::default();
        let freed = RefCell::new(Vec::new());
        queue.release(1, 1, |_| unreachable!("held"));
        queue.collect(1, |r| {
            freed.borrow_mut().push(r);
            queue.release(2, r + 10, |_| unreachable!("held"));
        });
        assert_eq!(*freed.borrow(), [1]);
        queue.collect(2, |r| freed.borrow_mut().push(r));
        assert_eq!(*freed.borrow(), [1, 11]);
    }
}
