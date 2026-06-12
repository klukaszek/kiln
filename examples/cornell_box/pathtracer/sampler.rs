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
    uint result = 0u;
    for (uint bit = 0u; bit < 32u; bit++) {
        if ((index & (1u << bit)) != 0u) {
            result ^= SOBOL_DIRECTIONS[dim - 1u][bit];
        }
    }
    return result;
}

float scrambled_to_unit(uint x, uint seed)
{
    return (float)(nested_uniform_scramble(x, seed) >> 8) * (1.0 / 16777216.0);
}

// One 4D low-discrepancy point. `group` selects a dimension group: every
// distinct (pixel, group) pair sees its own shuffled, scrambled sequence, while
// the four dimensions inside a group share one shuffled index so their joint
// stratification survives.
float4 sample_4d(uint pixel, uint sampleIndex, uint group)
{
    uint seed = hash_combine(hash_u32(pixel), group);
    uint index = nested_uniform_scramble(sampleIndex, hash_combine(seed, 0xa511e9b3u));
    return float4(
        scrambled_to_unit(sobol_u32(0u, index), hash_combine(seed, 1u)),
        scrambled_to_unit(sobol_u32(1u, index), hash_combine(seed, 2u)),
        scrambled_to_unit(sobol_u32(2u, index), hash_combine(seed, 3u)),
        scrambled_to_unit(sobol_u32(3u, index), hash_combine(seed, 4u)));
}
"#;

/// The sampler's full Slang source: direction table first, then the functions.
pub fn source() -> String {
    [sobol_directions_slang().as_str(), RNG].concat()
}

/// Emit the Sobol' direction-vector table for dimensions 2..=4 as a Slang constant
/// (dimension 1 needs no table: its point is the bit-reversed sample index).
fn sobol_directions_slang() -> String {
    let mut out = String::from("static const uint SOBOL_DIRECTIONS[3][32] = {\n");
    for directions in sobol_direction_vectors() {
        out.push_str("    {");
        for (i, v) in directions.iter().enumerate() {
            if i % 8 == 0 {
                out.push_str("\n        ");
            }
            out.push_str(&format!("0x{v:08x}u, "));
        }
        out.push_str("\n    },\n");
    }
    out.push_str("};\n");
    out
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
