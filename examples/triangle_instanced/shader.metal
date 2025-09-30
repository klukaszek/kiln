// Instanced triangle with sunflower distribution
#include <metal_stdlib>
using namespace metal;

struct SceneProperties { float time; };
struct VertexInput { packed_float3 position; packed_float3 color; };
struct VertexOutput { float4 position [[position]]; float4 color; };

vertex VertexOutput vertex_main(
    device const SceneProperties& properties [[buffer(0)]],
    device const VertexInput* vertices [[buffer(1)]],
    uint vertex_idx [[vertex_id]],
    uint instance_id [[instance_id]])
{
    VertexOutput out;
    VertexInput in = vertices[vertex_idx];
    float2 p = float2(in.position.x, in.position.y);
    float c = cos(properties.time);
    float s = sin(properties.time);
    float2 r = float2x2(c, -s, s, c) * p;
    const float golden = 2.39996323;
    float iid = float(instance_id);
    float angle = golden * iid;
    float t = tanh(0.06 * iid);
    float radius = 0.9 * t;
    float2 offset = radius * float2(cos(angle), sin(angle));
    float near_factor = 1.0 - t;
    float scale = mix(0.25, 1.15, near_factor);
    float2 rp = r * scale;
    out.position = float4(rp.x + offset.x, rp.y + offset.y, in.position.z, 1.0);
    out.color = float4(in.color, 1.0);
    return out;
}

fragment float4 fragment_main(VertexOutput in [[stage_in]]) { return in.color; }

