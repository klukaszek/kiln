#[derive(Clone, Copy, Debug)]
pub struct SpatialSchedule {
    target_spp: u32,
    passes_per_frame: u32,
    pixel_stride: u32,
    phase_count: u32,
    target_passes: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct TraceBatch {
    pub start: u32,
    pub count: u32,
    pub target: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct SpatialProgress {
    pub completed_samples: u32,
    pub remaining_phases: u32,
}

impl SpatialSchedule {
    pub fn new(target_spp: u32, passes_per_frame: u32, pixel_stride: u32) -> anyhow::Result<Self> {
        anyhow::ensure!(target_spp > 0, "target spp must be greater than zero");
        anyhow::ensure!(
            passes_per_frame > 0,
            "passes per frame must be greater than zero"
        );
        anyhow::ensure!(pixel_stride > 0, "pixel stride must be greater than zero");
        let phase_count = pixel_stride
            .checked_mul(pixel_stride)
            .ok_or_else(|| anyhow::anyhow!("pixel stride {pixel_stride} is too large"))?;
        let target_passes = target_spp.checked_mul(phase_count).ok_or_else(|| {
            anyhow::anyhow!("target spp and pixel stride produce too many passes")
        })?;
        Ok(Self {
            target_spp,
            passes_per_frame,
            pixel_stride,
            phase_count,
            target_passes,
        })
    }

    pub fn target_spp(self) -> u32 {
        self.target_spp
    }

    pub fn passes_per_frame(self) -> u32 {
        self.passes_per_frame
    }

    pub fn pixel_stride(self) -> u32 {
        self.pixel_stride
    }

    pub fn phase_count(self) -> u32 {
        self.phase_count
    }

    pub fn target_passes(self) -> u32 {
        self.target_passes
    }

    pub fn next_batch(self, completed_passes: u32) -> Option<TraceBatch> {
        let target = self.target_passes();
        let remaining = target.checked_sub(completed_passes)?;
        (remaining > 0).then_some(TraceBatch {
            start: completed_passes,
            count: self.passes_per_frame.min(remaining),
            target,
        })
    }

    pub fn progress(self, completed_passes: u32) -> SpatialProgress {
        SpatialProgress {
            completed_samples: completed_passes / self.phase_count,
            remaining_phases: completed_passes % self.phase_count,
        }
    }

    pub fn sample_count(self, completed_passes: u32) -> u32 {
        self.progress(completed_passes).completed_samples
    }

    pub fn sample_count_for_pixel(self, completed_passes: u32, x: u32, y: u32) -> u32 {
        let progress = self.progress(completed_passes);
        let phase = (y % self.pixel_stride) * self.pixel_stride + x % self.pixel_stride;
        progress.completed_samples + u32::from(phase < progress.remaining_phases)
    }

    pub fn is_complete(self, completed_passes: u32) -> bool {
        completed_passes >= self.target_passes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batches_stop_at_the_exact_target() {
        let schedule = SpatialSchedule::new(2, 3, 2).unwrap();

        let first = schedule.next_batch(0).unwrap();
        assert_eq!((first.start, first.count, first.target), (0, 3, 8));
        let last = schedule.next_batch(6).unwrap();
        assert_eq!((last.start, last.count, last.target), (6, 2, 8));
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
