//! Sample generation: hash-based Owen-scrambled Sobol' (Burley, "Practical
//! Hash-based Owen Scrambling", JCGT 2020).
//!
//! Each (pixel, dimension group) gets an independently shuffled + scrambled copy of
//! the same 4D Sobol' sequence indexed by sample number: stratified per pixel for
//! fast convergence, decorrelated across pixels and across dimension groups so
//! padding introduces no structured aliasing. The direction-vector table is computed
//! here on the host from the Joe–Kuo parameters and injected into the Slang source.

use std::fmt::Write;

/// Sobol' sequence dimensions per padded point. The integrator never consumes more
/// than four dimensions at once, so every decision draws one 4D point from its own
/// dimension group and the table stays tiny.
const SOBOL_DIMS: usize = 4;

const RNG: &str = include_str!("integrator/sampler.slang");

/// The sampler's full Slang source: direction table first, then the functions.
pub fn source() -> String {
    let mut source = sobol_byte_lut_slang();
    source.push_str(RNG);
    source
}

/// Emit byte-folded Sobol direction tables for dimensions 2..=4.
fn sobol_byte_lut_slang() -> String {
    let mut out = String::from("static const uint SOBOL_BYTE_LUT[3][4][256] = {\n");
    for dimension in sobol_byte_lut() {
        out.push_str("    {\n");
        for byte in dimension {
            out.push_str("        {");
            for (value, folded) in byte.into_iter().enumerate() {
                if value % 8 == 0 {
                    out.push_str("\n            ");
                }
                write!(out, "0x{folded:08x}u, ").expect("writing to a String cannot fail");
            }
            out.push_str("\n        },\n");
        }
        out.push_str("    },\n");
    }
    out.push_str("};\n");
    out
}

fn sobol_byte_lut() -> [[[u32; 256]; 4]; SOBOL_DIMS - 1] {
    let mut lut = [[[0u32; 256]; 4]; SOBOL_DIMS - 1];
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
