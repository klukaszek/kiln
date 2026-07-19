//! Sample generation: hash-based Owen-scrambled Sobol' (Burley, "Practical
//! Hash-based Owen Scrambling", JCGT 2020).
//!
//! Each (pixel, dimension group) gets an independently shuffled + scrambled copy of
//! the same 4D Sobol' sequence indexed by sample number: stratified per pixel for
//! fast convergence, decorrelated across pixels and across dimension groups so
//! padding introduces no structured aliasing. The direction-vector table is computed
//! here on the host from the Joe–Kuo parameters and injected into the Slang source.

/// Sobol' sequence dimensions per padded point. The integrator never consumes more
/// than four dimensions at once, so every decision draws one 4D point from its own
/// dimension group and the table stays tiny.
const SOBOL_DIMS: usize = 4;

const RNG: &str = /*slang*/
    r#"
uint hash_u32(uint x)
{
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

uint hash_combine(uint seed, uint v)
{
    return hash_u32(seed ^ (v + 0x9e3779b9u + (seed << 6) + (seed >> 2)));
}

uint laine_karras_permutation(uint x, uint seed)
{
    x += seed;
    x ^= x * 0x6c50b47cu;
    x ^= x * 0xb82f1e52u;
    x ^= x * 0xc7afe638u;
    x ^= x * 0x8d22f6e6u;
    return x;
}

// Owen scramble of the binary radical-inverse tree: each bit is flipped based
// only on the bits above it, so power-of-two sample prefixes stay stratified.
uint nested_uniform_scramble(uint x, uint seed)
{
    x = reversebits(x);
    x = laine_karras_permutation(x, seed);
    return reversebits(x);
}

uint sobol_u32(uint dim, uint index)
{
    if (dim == 0u) {
        return reversebits(index);
    }
    uint table = dim - 1u;
    // Four byte lookups replace the 32-bit direction scan.
    return SOBOL_BYTE_LUT[table][0][index & 0xffu]
        ^ SOBOL_BYTE_LUT[table][1][(index >> 8) & 0xffu]
        ^ SOBOL_BYTE_LUT[table][2][(index >> 16) & 0xffu]
        ^ SOBOL_BYTE_LUT[table][3][index >> 24];
}

float scrambled_to_unit(uint x, uint seed)
{
    return (float)(nested_uniform_scramble(x, seed) >> 8) * (1.0 / 16777216.0);
}

// One 4D point per pixel and dimension group.
float4 sample_4d(uint pixelSeed, uint sampleIndex, uint group)
{
    uint seed = hash_combine(pixelSeed, group);
    uint index = sampleIndex;
    return float4(
        scrambled_to_unit(sobol_u32(0u, index), hash_combine(seed, 1u)),
        scrambled_to_unit(sobol_u32(1u, index), hash_combine(seed, 2u)),
        scrambled_to_unit(sobol_u32(2u, index), hash_combine(seed, 3u)),
        scrambled_to_unit(sobol_u32(3u, index), hash_combine(seed, 4u)));
}
"#;

/// The sampler's full Slang source: direction table first, then the functions.
pub fn source() -> String {
    [sobol_byte_lut_slang().as_str(), RNG].concat()
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
                out.push_str(&format!("0x{folded:08x}u, "));
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
    const PARAMS: [(usize, u32, [u32; 3]); SOBOL_DIMS - 1] = [
        (1, 0, [1, 0, 0]),
        (2, 1, [1, 3, 0]),
        (3, 1, [1, 3, 1]),
    ];

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
