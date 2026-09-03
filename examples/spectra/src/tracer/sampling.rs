//! How samples are placed: the progressive schedule that decides which pixels a pass touches, and
//! the Owen-scrambled Sobol' sequence each path draws from.
//!
//! Hash-based Owen scrambling per Burley, "Practical Hash-based Owen Scrambling" (JCGT 2020). Each
//! (pixel, dimension group) gets an independently shuffled copy of the same 4D sequence: stratified
//! per pixel, decorrelated across pixels so padding introduces no structured aliasing.

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

/// Sobol' sequence dimensions per padded point. The integrator never consumes more
/// than four dimensions at once, so every decision draws one 4D point from its own
/// dimension group and the table stays tiny.
const SOBOL_DIMS: usize = 4;

/// Bytes of `index` folded per lookup, one table each.
pub(super) const SOBOL_BYTES: usize = 4;

/// Values a folded byte can take.
pub(super) const SOBOL_VALUES: usize = 256;

/// The byte-folded direction table, flattened to `[table][byte][value]` for the shader.
///
/// Dimension 0 is the plain radical inverse and needs no table, so only dimensions 2..=4 are
/// stored.
pub(super) fn table() -> Vec<u32> {
    sobol_byte_lut()
        .into_iter()
        .flatten()
        .flatten()
        .collect::<Vec<_>>()
}

fn sobol_byte_lut() -> [[[u32; SOBOL_VALUES]; SOBOL_BYTES]; SOBOL_DIMS - 1] {
    let mut lut = [[[0u32; SOBOL_VALUES]; SOBOL_BYTES]; SOBOL_DIMS - 1];
    for (dimension, directions) in lut.iter_mut().zip(sobol_direction_vectors()) {
        for (byte_index, byte) in dimension.iter_mut().enumerate() {
            for (value, folded) in byte.iter_mut().enumerate() {
                for bit in 0..8 {
                    if value & (1 << bit) != 0 {
                        *folded ^= directions[byte_index * 8 + bit];
                    }
                }
            }
        }
    }
    lut
}

/// Direction vectors for Sobol' dimensions 2..=4, computed from the Joe–Kuo
/// `new-joe-kuo-6` parameters: (degree of the primitive polynomial, its interior
/// coefficient bits a_1..a_{s-1} packed MSB-first, the first s values of m).
fn sobol_direction_vectors() -> [[u32; 32]; SOBOL_DIMS - 1] {
    const PARAMS: [(usize, u32, [u32; 3]); SOBOL_DIMS - 1] =
        [(1, 0, [1, 0, 0]), (2, 1, [1, 3, 0]), (3, 1, [1, 3, 1])];

    let mut all = [[0u32; 32]; SOBOL_DIMS - 1];
    for (directions, &(s, a, m_init)) in all.iter_mut().zip(PARAMS.iter()) {
        let mut m = [0u32; 32];
        m[..s].copy_from_slice(&m_init[..s]);
        for k in s..32 {
            // m_k = 2^s m_{k-s} ^ m_{k-s} ^ XOR_i (2^i a_i m_{k-i})
            let mut v = m[k - s] ^ (m[k - s] << s);
            for i in 1..s {
                if (a >> (s - 1 - i)) & 1 == 1 {
                    v ^= m[k - i] << i;
                }
            }
            m[k] = v;
        }
        for k in 0..32 {
            directions[k] = m[k] << (31 - k);
        }
    }
    all
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
