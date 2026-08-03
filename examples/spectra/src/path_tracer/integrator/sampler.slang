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

// Next-event estimation consumes only xyz. Keep the fourth Sobol dimension
// out of the hot path instead of computing and discarding it every bounce.
float3 sample_3d(uint pixelSeed, uint sampleIndex, uint group)
{
    uint seed = hash_combine(pixelSeed, group);
    uint index = sampleIndex;
    return float3(
        scrambled_to_unit(sobol_u32(0u, index), hash_combine(seed, 1u)),
        scrambled_to_unit(sobol_u32(1u, index), hash_combine(seed, 2u)),
        scrambled_to_unit(sobol_u32(2u, index), hash_combine(seed, 3u)));
}
