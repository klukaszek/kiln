use crate::render::{Error, Result};

#[derive(Clone, Copy, Debug)]
pub(super) struct SpatialSchedule {
    pub(super) target_spp: u32,
    pub(super) passes_per_frame: u32,
    pub(super) pixel_stride: u32,
    phase_count: u32,
    target_passes: u32,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct TraceBatch {
    pub(super) start: u32,
    pub(super) count: u32,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct SpatialProgress {
    pub(super) completed_samples: u32,
    pub(super) remaining_phases: u32,
}

impl SpatialSchedule {
    pub(super) fn new(target_spp: u32, passes_per_frame: u32, pixel_stride: u32) -> Result<Self> {
        if target_spp == 0 || passes_per_frame == 0 || pixel_stride == 0 {
            return Err(Error::Settings(
                "sample count, passes per frame, and pixel stride must be non-zero",
            ));
        }
        let phase_count = pixel_stride
            .checked_mul(pixel_stride)
            .ok_or(Error::Settings("pixel stride is too large"))?;
        let target_passes = target_spp
            .checked_mul(phase_count)
            .ok_or(Error::Settings("sample schedule exceeds u32 capacity"))?;
        Ok(Self {
            target_spp,
            passes_per_frame,
            pixel_stride,
            phase_count,
            target_passes,
        })
    }

    pub(super) fn next_batch(self, completed_passes: u32) -> Option<TraceBatch> {
        let target = self.target_passes;
        (completed_passes < target).then(|| TraceBatch {
            start: completed_passes,
            count: self.passes_per_frame.min(target - completed_passes),
        })
    }

    pub(super) fn progress(self, completed_passes: u32) -> SpatialProgress {
        SpatialProgress {
            completed_samples: completed_passes / self.phase_count,
            remaining_phases: completed_passes % self.phase_count,
        }
    }

    pub(super) fn sample_count(self, completed_passes: u32) -> u32 {
        self.progress(completed_passes).completed_samples
    }

    pub(super) fn sample_count_for_pixel(self, completed_passes: u32, x: u32, y: u32) -> u32 {
        let progress = self.progress(completed_passes);
        let phase = (y % self.pixel_stride) * self.pixel_stride + x % self.pixel_stride;
        progress.completed_samples + u32::from(phase < progress.remaining_phases)
    }

    pub(super) fn is_complete(self, completed_passes: u32) -> bool {
        completed_passes >= self.target_passes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batches_stop_at_the_exact_target() {
        let schedule = SpatialSchedule::new(2, 3, 2).unwrap();

        let first = schedule.next_batch(0).unwrap();
        assert_eq!((first.start, first.count), (0, 3));
        let last = schedule.next_batch(6).unwrap();
        assert_eq!((last.start, last.count), (6, 2));
        assert!(schedule.next_batch(8).is_none());
        assert!(schedule.is_complete(8));
    }

    #[test]
    fn partial_phase_counts_only_pixels_already_visited() {
        let schedule = SpatialSchedule::new(2, 1, 2).unwrap();

        assert_eq!(schedule.sample_count_for_pixel(5, 0, 0), 2);
        assert_eq!(schedule.sample_count_for_pixel(5, 1, 0), 1);
        assert_eq!(schedule.sample_count_for_pixel(5, 0, 1), 1);
        assert_eq!(schedule.sample_count_for_pixel(5, 1, 1), 1);
    }

    #[test]
    fn rejects_zero_and_overflowing_inputs() {
        assert!(SpatialSchedule::new(0, 1, 1).is_err());
        assert!(SpatialSchedule::new(1, 0, 1).is_err());
        assert!(SpatialSchedule::new(1, 1, 0).is_err());
        assert!(SpatialSchedule::new(u32::MAX, 1, 2).is_err());
    }
}
