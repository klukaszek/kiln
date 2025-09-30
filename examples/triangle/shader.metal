// Basic rotating triangle
#include <metal_stdlib>
using namespace metal;

struct SceneProperties { float time; };
struct VertexInput { packed_float3 position; packed_float3 color; };
struct VertexOutput { float4 position [[position]]; float4 color; };

vertex VertexOutput vertex_main(
    device const SceneProperties& properties [[buffer(0)]],
    device const VertexInput* vertices [[buffer(1)]],
    uint vertex_idx [[vertex_id]])
{
    VertexOutput out;
    VertexInput in = vertices[vertex_idx];
    float2 p = float2(in.position.x, in.position.y);
    float c = cos(properties.time);
    float s = sin(properties.time);
    float2 r = float2x2(c, -s, s, c) * p;
    out.position = float4(r.x, r.y, in.position.z, 1.0);
    out.color = float4(in.color, 1.0);
    return out;
}

fragment float4 fragment_main(VertexOutput in [[stage_in]])
{ return in.color; }

