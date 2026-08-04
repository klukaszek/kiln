//! Raster-preview shader interface and pipeline state.

use glam::Vec4;
use kiln_rhi::{
    ColorTarget, CommandBuffer, CompareOp, Cull, DepthFlags, DepthStencilState, Device, Format,
    GpuAddress, MeshletPso, MeshletPsoDesc, SampleCount, ShaderStage, Topology, gpu_struct,
};

use super::scene::{GpuRasterMaterial, RasterVertex};
use crate::base::gpu::GpuTextureBinding;
use crate::base::renderer;

const TRIANGLES_PER_MESHLET: u32 = 64;

gpu_struct! {
    pub(super) struct Root {
        vp0: Vec4,
        vp1: Vec4,
        vp2: Vec4,
        vp3: Vec4,
        cam_pos: Vec4,
        tri_count: u32,
        _pad: u32,
        verts: GpuAddress as "RasterVertex*",
        materials: GpuAddress as "GpuRasterMaterial*",
        texture_bindings: GpuAddress as "GpuTextureBinding*",
    }
}

const SHADER_BODY: &str = /*slang*/
    r#"
static const uint NO_TEXTURE = 0xffffffffu;

struct VOut {
    float4 pos : SV_Position;
    float3 world : WORLD;
    float3 nrm : NORMAL;
    float2 uv : TEXCOORD0;
    nointerpolation uint material_id : MATERIAL;
};

[shader("mesh")]
[numthreads(1, 1, 1)]
[outputtopology("triangle")]
void msMain(uint3 gid : SV_GroupID, out vertices VOut verts[192], out indices uint3 tris[64], uniform Root* r)
{
    uint base = gid.x * 64u;
    uint count = min(64u, r.tri_count - base);
    SetMeshOutputCounts(count * 3u, count);
    for (uint t = 0u; t < count; t++) {
        for (uint k = 0u; k < 3u; k++) {
            RasterVertex v = r.verts[(base + t) * 3u + k];
            VOut o;
            o.pos = v.pos.x * r.vp0 + v.pos.y * r.vp1 + v.pos.z * r.vp2 + v.pos.w * r.vp3;
            o.world = v.pos.xyz;
            o.nrm = v.normal.xyz;
            o.uv = v.uv.xy;
            o.material_id = v.material_id;
            verts[t * 3u + k] = o;
        }
        tris[t] = uint3(t * 3u, t * 3u + 1u, t * 3u + 2u);
    }
}

[shader("fragment")]
float4 fsMain(VOut i, uniform Root* r) : SV_Target
{
    float3 n = normalize(i.nrm);
    float3 l = normalize(r.cam_pos.xyz - i.world);
    GpuRasterMaterial material = r.materials[i.material_id];
    float3 color = material.color.rgb;
    if (material.texture_id != NO_TEXTURE) {
        GpuTextureBinding binding = r.texture_bindings[material.texture_id];
        Texture2D texture = binding.image;
        SamplerState sampler = binding.sampler;
        color *= texture.Sample(sampler, float2(i.uv.x, 1.0 - i.uv.y)).rgb;
    }
    return float4(color * (0.2 + 0.8 * abs(dot(n, l))), 1.0);
}
"#;

pub(super) struct Pipeline(MeshletPso);

impl Pipeline {
    pub(super) fn new(device: &Device, color_format: Format) -> renderer::Result<Self> {
        let source = format!(
            "{}{}{}{}{}",
            RasterVertex::SLANG,
            GpuRasterMaterial::SLANG,
            GpuTextureBinding::SLANG,
            Root::SLANG,
            SHADER_BODY
        );
        let mesh_shader = kiln_rhi::compiler::compile(device, &source, "msMain", ShaderStage::Mesh);
        let fragment_shader =
            kiln_rhi::compiler::compile(device, &source, "fsMain", ShaderStage::Pixel);
        let pipeline = device.create_meshlet_pso(
            &MeshletPsoDesc {
                topology: Topology::TriangleList,
                color_targets: vec![ColorTarget::new(color_format)],
                depth_format: Some(Format::D32Float),
                stencil_format: None,
                sample_count: SampleCount::S1,
                alpha_to_coverage: false,
                cull: Cull::None,
                support_dual_source_blending: false,
                blendstate: None,
                label: Some("spectra-raster".into()),
            },
            &mesh_shader,
            &fragment_shader,
        )?;
        Ok(Self(pipeline))
    }

    pub(super) fn record(&self, commands: &mut CommandBuffer, root: GpuAddress, triangles: u32) {
        commands.set_meshlet_pipeline(&self.0);
        commands.set_depth_stencil_state(&DepthStencilState {
            depth_mode: DepthFlags::READ | DepthFlags::WRITE,
            depth_test: CompareOp::Less,
            stencil_read_mask: 0,
            stencil_write_mask: 0,
            ..Default::default()
        });
        commands.draw_meshlets(root, triangles.div_ceil(TRIANGLES_PER_MESHLET), 1, 1);
    }
}
