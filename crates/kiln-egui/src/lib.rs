//! egui painter for the Kiln RHI.
//!
//! Renders [`egui`]'s tessellated output through Kiln's pointer-first model: each
//! [`egui::ClippedPrimitive`] mesh is uploaded into per-frame vertex/index buffers and drawn with
//! [`kiln_rhi::CommandBuffer::draw_indexed`], reading its vertices through a root pointer and
//! sampling the font/image atlas through the bindless heap ([`kiln_rhi::TextureHandle`] /
//! [`kiln_rhi::SamplerHandle`], obtained directly from texture and sampler objects). This is the
//! Kiln analogue of `egui_wgpu` /
//! `egui_glow`; the windowed input glue (winit) lives in the caller (see the `egui_demo` example).
//!
//! Usage per frame, around the swapchain render pass:
//! ```ignore
//! renderer.update_textures(device, &full_output.textures_delta)?;   // before the pass
//! cmd.begin_render_pass(&pass);
//! renderer.paint(device, &mut cmd, slot, pixels_per_point, [w, h], &primitives)?;
//! cmd.end_render_pass();
//! renderer.free_textures(device, &full_output.textures_delta.free); // after the pass
//! ```
//!
//! Requires `slangc` on `PATH` (the painter compiles its shader through `kiln_rhi::compiler`).

use std::collections::HashMap;
use std::mem::size_of;

use kiln_rhi::{
    AddressMode, Allocation, AllocationDesc, BlendAttachment, BlendFactor, BlendOp, BlendState,
    ColorTarget, CommandBuffer, Cull, Device, FilterMode, Format, GraphicsPso, GraphicsPsoDesc,
    MAX_FRAMES_IN_FLIGHT, MemoryType, RhiError, RhiResult, SampleCount, Sampler, SamplerDesc,
    SamplerHandle, ShaderStage, StageFlags, Texture, TextureDesc, TextureDimension, TextureHandle,
    TextureUsage, Topology, gpu_struct,
};

gpu_struct! {
    /// One egui vertex. Byte-identical to `egui::epaint::Vertex` (`{ pos, uv, color }`, 20 bytes),
    /// so meshes upload verbatim. `color` is packed sRGB premultiplied RGBA8.
    struct EguiVertex {
        pos: [f32; 2],
        uv: [f32; 2],
        color: u32,
    }
}

gpu_struct! {
    /// Per-mesh draw root. `verts` points at this mesh's vertex sub-array (egui indices are
    /// per-mesh 0-based, so `r.verts[vid]` resolves directly). `flags` bit 0 = sRGB target.
    struct EguiRoot {
        verts: GpuPtr<EguiVertex>,
        screen_size: [f32; 2],
        flags: u32,
        _pad: u32,
        tex: TextureHandle,
        smp: SamplerHandle,
    }
}

/// Root stride, rounded up to 16 bytes so each per-mesh root stays aligned in the ring buffer.
const ROOT_STRIDE: u64 = (std::mem::size_of::<EguiRoot>() as u64 + 15) & !15;

// One Slang source: `EguiVertex` + `EguiRoot` declarations are prepended so the host/device
// layouts stay locked. Colours follow the canonical egui pipeline: vertex colours and texels are
// sRGB premultiplied; convert both to linear, multiply, then either output linear (sRGB target,
// the GPU encodes) or re-encode to gamma (UNORM target).
const SHADER_BODY: &str = /* slang */
    r#"
struct VOut {
    float4 pos   : SV_Position;
    float2 uv    : TEXCOORD0;
    float4 color : COLOR0;
};

float lin1(float c) { return c <= 0.04045 ? c / 12.92 : pow((c + 0.055) / 1.055, 2.4); }
float3 to_linear(float3 c) { return float3(lin1(c.x), lin1(c.y), lin1(c.z)); }
float gam1(float c) { return c <= 0.0031308 ? c * 12.92 : 1.055 * pow(c, 1.0 / 2.4) - 0.055; }
float3 to_gamma(float3 c) { return float3(gam1(c.x), gam1(c.y), gam1(c.z)); }

[shader("vertex")]
VOut vsMain(uint vid : SV_VertexID, uniform EguiRoot* r)
{
    EguiVertex v = r.verts[vid];
    VOut o;
    // egui points (origin top-left, y-down) -> Y-up NDC (Kiln normalizes every backend to Y-up).
    float2 p = v.pos / r.screen_size;
    o.pos = float4(p.x * 2.0 - 1.0, 1.0 - p.y * 2.0, 0.0, 1.0);
    o.uv = v.uv;
    float4 c = float4(
        float(v.color & 0xFFu),
        float((v.color >> 8) & 0xFFu),
        float((v.color >> 16) & 0xFFu),
        float((v.color >> 24) & 0xFFu)) / 255.0;
    o.color = float4(to_linear(c.rgb), c.a); // premultiplied; gamma applied to rgb (egui's approx)
    return o;
}

[shader("fragment")]
float4 fsMain(VOut i, uniform EguiRoot* r) : SV_Target
{
    Texture2D tex = r.tex;
    SamplerState smp = r.smp;
    float4 t = tex.Sample(smp, i.uv);
    float4 lin = i.color * float4(to_linear(t.rgb), t.a);
    if ((r.flags & 1u) != 0u) {
        return lin;                                  // sRGB target: hardware encodes on store
    }
    return float4(to_gamma(lin.rgb), lin.a);         // UNORM target: encode in-shader
}
"#;

/// A GPU texture egui asked us to manage (the font atlas, or a user image), plus a CPU shadow
/// copy so egui's sub-region patch updates can be applied and the whole texture re-uploaded
/// (the RHI's texture copy has no sub-rect form).
struct ManagedTexture {
    texture: Texture,
    mem: Allocation,
    /// Value for the root's [`TextureHandle`] field (heap index on Vulkan, `gpuResourceID` on Metal).
    handle: TextureHandle,
    width: u32,
    height: u32,
    shadow: Vec<u8>, // RGBA8, width*height*4
}

impl ManagedTexture {
    fn destroy(self, device: &Device) {
        device.destroy(self.texture);
        device.destroy(self.mem);
    }
}

/// Per-frame-in-flight geometry buffers. Reused (and grown) once their slot's prior frame has
/// retired, which the swapchain's per-slot fence guarantees before recording begins.
#[derive(Default)]
struct FrameBuffers {
    vtx: Option<Allocation>,
    idx: Option<Allocation>,
    root: Option<Allocation>,
}

/// egui renderer/painter built on the Kiln RHI. Create once with the swapchain's colour format;
/// drive it each frame with [`update_textures`](Self::update_textures) + [`paint`](Self::paint).
pub struct EguiRenderer {
    pso: GraphicsPso,
    _sampler: Sampler,
    sampler_handle: SamplerHandle,
    textures: HashMap<egui::TextureId, ManagedTexture>,
    frames: [FrameBuffers; MAX_FRAMES_IN_FLIGHT],
    srgb_target: bool,
}

impl EguiRenderer {
    /// Build the egui pipeline for a render target of `color_format` (typically the swapchain
    /// format). Compiles the painter shader via `slangc` (must be on `PATH`).
    pub fn new(device: &Device, color_format: Format) -> RhiResult<Self> {
        let src = format!("{}{}{}", EguiVertex::SLANG, EguiRoot::SLANG, SHADER_BODY);
        let vs = kiln_rhi::compiler::compile(device, &src, "vsMain", ShaderStage::Vertex, &[])?;
        let fs = kiln_rhi::compiler::compile(device, &src, "fsMain", ShaderStage::Pixel, &[])?;

        // Premultiplied-alpha blending: out = src + dst*(1-src.a); alpha accumulates so the
        // result composites correctly even when rendering egui into an offscreen target.
        let blend = BlendState {
            attachments: vec![BlendAttachment {
                blend_enable: true,
                src_color: BlendFactor::One,
                dst_color: BlendFactor::OneMinusSrcAlpha,
                color_op: BlendOp::Add,
                src_alpha: BlendFactor::OneMinusDstAlpha,
                dst_alpha: BlendFactor::One,
                alpha_op: BlendOp::Add,
            }],
        };

        let pso = device.create_graphics_pso(
            &GraphicsPsoDesc {
                topology: Topology::TriangleList,
                color_targets: vec![ColorTarget::new(color_format)],
                depth_format: None,
                sample_count: SampleCount::S1,
                // Two root pointers (vertex + pixel share one EguiRoot): 2 * 8 bytes.
                // egui is not consistent about winding; never cull.
                cull: Cull::None,
                blendstate: Some(blend),
                label: Some("egui".into()),
                ..Default::default()
            },
            &vs,
            &fs,
        )?;

        // egui samples its atlas with linear filtering and clamps at the edges.
        let sampler = device.create_sampler(&SamplerDesc {
            min_filter: FilterMode::Linear,
            mag_filter: FilterMode::Linear,
            mip_filter: FilterMode::Linear,
            address_u: AddressMode::ClampToEdge,
            address_v: AddressMode::ClampToEdge,
            address_w: AddressMode::ClampToEdge,
            label: Some("egui-sampler".into()),
            ..Default::default()
        })?;
        let sampler_handle = sampler.gpu();

        Ok(Self {
            pso,
            sampler_handle,
            _sampler: sampler,
            textures: HashMap::new(),
            frames: Default::default(),
            srgb_target: is_srgb(color_format),
        })
    }

    /// Apply egui's texture *set* deltas (call before the render pass). Newly created and updated
    /// textures are uploaded synchronously; egui only emits these when the atlas changes (e.g. a
    /// new glyph), so steady-state frames do no work here.
    pub fn update_textures(
        &mut self,
        device: &Device,
        delta: &egui::TexturesDelta,
    ) -> RhiResult<()> {
        for (id, image_delta) in &delta.set {
            self.set_texture(device, *id, image_delta)?;
        }
        Ok(())
    }

    /// Free textures egui dropped. Drains the GPU first: destruction is immediate and the frame
    /// just submitted may still be sampling them. `free` is empty on almost every frame.
    pub fn free_textures(&mut self, device: &Device, free: &[egui::TextureId]) {
        let doomed: Vec<_> = free
            .iter()
            .filter_map(|id| self.textures.remove(id))
            .collect();
        if doomed.is_empty() {
            return;
        }
        device.wait_idle();
        for m in doomed {
            m.destroy(device);
        }
    }

    /// Record draws for one frame's tessellated `primitives` into an already-begun render pass.
    /// `slot` is the frame-in-flight index (its buffers are reused), `framebuffer_px` the target
    /// size in physical pixels, and `pixels_per_point` egui's current scale factor.
    pub fn paint(
        &mut self,
        device: &Device,
        cmd: &mut CommandBuffer,
        slot: usize,
        pixels_per_point: f32,
        framebuffer_px: [u32; 2],
        primitives: &[egui::ClippedPrimitive],
    ) -> RhiResult<()> {
        // Sum up the geometry so the per-slot buffers can be sized in one pass. Only meshes whose
        // texture we actually hold are drawn, but sizing for all meshes is simpler and harmless.
        let mut total_verts = 0u64;
        let mut total_indices = 0u64;
        let mut mesh_count = 0u64;
        for prim in primitives {
            if let egui::epaint::Primitive::Mesh(mesh) = &prim.primitive {
                if mesh.indices.is_empty() {
                    continue;
                }
                total_verts += mesh.vertices.len() as u64;
                total_indices += mesh.indices.len() as u64;
                mesh_count += 1;
            }
        }
        if mesh_count == 0 {
            return Ok(());
        }

        // Grow (reuse) this slot's buffers to fit. Safe to free the old ones: the slot's previous
        // frame has retired (the swapchain fence was waited at acquire time).
        let frame = &mut self.frames[slot];
        grow(
            device,
            &mut frame.vtx,
            total_verts * size_of::<EguiVertex>() as u64,
            "egui-vtx",
        )?;
        grow(
            device,
            &mut frame.idx,
            total_indices * size_of::<u32>() as u64,
            "egui-idx",
        )?;
        grow(
            device,
            &mut frame.root,
            mesh_count * ROOT_STRIDE,
            "egui-root",
        )?;

        let vtx = frame.vtx.as_ref().unwrap();
        let idx = frame.idx.as_ref().unwrap();
        let root = frame.root.as_ref().unwrap();
        let vtx = mapped::<EguiVertex>(vtx);
        let idx = mapped::<u32>(idx);
        let root = mapped::<u8>(root);

        let [fb_w, fb_h] = framebuffer_px;
        let screen_size = [
            fb_w as f32 / pixels_per_point,
            fb_h as f32 / pixels_per_point,
        ];
        let flags = self.srgb_target as u32;

        cmd.set_pipeline(&self.pso);
        cmd.set_viewport(0.0, 0.0, fb_w as f32, fb_h as f32, 0.0, 1.0);

        let mut v_off = 0u64; // vertices written so far
        let mut i_off = 0u64; // indices written so far
        let mut m_off = 0u64; // meshes drawn so far
        for prim in primitives {
            let egui::epaint::Primitive::Mesh(mesh) = &prim.primitive else {
                // Paint callbacks (custom GPU passes) are not supported by this painter yet.
                continue;
            };
            if mesh.indices.is_empty() {
                continue;
            }
            let Some(managed) = self.textures.get(&mesh.texture_id) else {
                // Texture not registered (e.g. a user texture we don't manage): skip its draw.
                continue;
            };

            // Scissor from the clip rect (points -> physical pixels), clamped to the target.
            let Some((sx, sy, sw, sh)) = scissor(prim.clip_rect, pixels_per_point, fb_w, fb_h)
            else {
                continue; // fully clipped
            };

            // One offset per buffer moves the CPU and GPU addresses together, so the bytes
            // written here and the address handed to the draw cannot disagree.
            let verts = vtx.offset(v_off);
            let indices = idx.offset(i_off);
            // egui's `Vertex` is byte-identical to `EguiVertex`, so blit it.
            verts
                .cast::<u8>()
                .write_slice(bytemuck::cast_slice(&mesh.vertices))?;
            indices.write_slice(&mesh.indices)?;

            let slot = root.byte_offset(m_off * ROOT_STRIDE);
            slot.cast::<EguiRoot>().write(&EguiRoot {
                verts: verts.gpu(),
                screen_size,
                flags,
                _pad: 0,
                tex: managed.handle,
                smp: self.sampler_handle,
            })?;

            cmd.set_scissor(sx, sy, sw, sh);
            cmd.draw_indexed(slot.gpu(), indices.gpu(), mesh.indices.len() as u32, 1);

            v_off += mesh.vertices.len() as u64;
            i_off += mesh.indices.len() as u64;
            m_off += 1;
        }
        Ok(())
    }

    /// Release all GPU resources. Dropping leaks textures/buffers (RHI handles are not RAII for
    /// device-owned storage), so call this before the device is destroyed.
    pub fn destroy(self, device: &Device) {
        let EguiRenderer {
            textures,
            frames,
            _sampler,
            ..
        } = self;
        for (_, m) in textures {
            m.destroy(device);
        }
        for f in frames {
            for b in [f.vtx, f.idx, f.root].into_iter().flatten() {
                device.destroy(b);
            }
        }
        device.destroy(_sampler);
    }

    /// Apply one egui texture delta (create / full update / sub-region patch), then upload the
    /// whole texture synchronously.
    fn set_texture(
        &mut self,
        device: &Device,
        id: egui::TextureId,
        delta: &egui::epaint::ImageDelta,
    ) -> RhiResult<()> {
        let egui::epaint::ImageData::Color(image) = &delta.image;
        let [pw, ph] = image.size;
        // egui's Color32 is premultiplied sRGB RGBA8; bytes upload directly into an UNORM texture.
        let patch: &[u8] = bytemuck::cast_slice(image.pixels.as_slice());

        if let Some([px, py]) = delta.pos {
            // Sub-region update of an existing texture: patch the shadow, re-upload the whole image.
            let m = self.textures.get_mut(&id).ok_or_else(|| {
                RhiError::Backend("egui patched a texture that was never created".into())
            })?;
            blit(&mut m.shadow, m.width as usize, [px, py], [pw, ph], patch);
        } else {
            // Full image. (Re)create if the size changed or it's new; otherwise overwrite shadow.
            let recreate = self
                .textures
                .get(&id)
                .is_none_or(|m| m.width as usize != pw || m.height as usize != ph);
            if recreate {
                if let Some(old) = self.textures.remove(&id) {
                    // An in-flight frame may still be sampling the old atlas.
                    device.wait_idle();
                    old.destroy(device);
                }
                let m = create_texture(device, pw as u32, ph as u32, patch.to_vec())?;
                self.textures.insert(id, m);
            } else {
                self.textures
                    .get_mut(&id)
                    .unwrap()
                    .shadow
                    .copy_from_slice(patch);
            }
        }

        upload_full(device, self.textures.get(&id).unwrap())
    }
}

/// True for the sRGB colour formats, where the GPU encodes linear->sRGB on attachment store.
fn is_srgb(format: Format) -> bool {
    matches!(format, Format::R8G8B8A8Srgb | Format::B8G8R8A8Srgb)
}

/// CPU-mapped base pointer of a `Default` buffer (always mapped; panics otherwise — a bug).
fn mapped<T>(buf: &Allocation) -> kiln_rhi::Mapped<'_, T> {
    buf.mapped()
        .expect("egui geometry buffer must be CPU-mapped")
}

/// Ensure `buf` exists and holds at least `need` bytes, reallocating (and freeing the old) on
/// growth. Grows in powers of two to amortize reallocation as egui's geometry fluctuates.
fn grow(device: &Device, buf: &mut Option<Allocation>, need: u64, label: &str) -> RhiResult<()> {
    let have = buf.as_ref().map_or(0, |b| b.size());
    if have >= need {
        return Ok(());
    }
    if let Some(old) = buf.take() {
        device.destroy(old);
    }
    let size = need.next_power_of_two().max(4096);
    *buf = Some(device.create_allocation(&AllocationDesc {
        size,
        memory: MemoryType::Upload,
        label: Some(label.into()),
        ..Default::default()
    })?);
    Ok(())
}

/// Convert an egui clip rect (points) to a physical-pixel scissor clamped to the target, or
/// `None` if it is empty after clamping.
fn scissor(clip: egui::Rect, ppp: f32, fb_w: u32, fb_h: u32) -> Option<(i32, i32, u32, u32)> {
    let min_x = (clip.min.x * ppp).floor().clamp(0.0, fb_w as f32) as u32;
    let min_y = (clip.min.y * ppp).floor().clamp(0.0, fb_h as f32) as u32;
    let max_x = (clip.max.x * ppp).ceil().clamp(0.0, fb_w as f32) as u32;
    let max_y = (clip.max.y * ppp).ceil().clamp(0.0, fb_h as f32) as u32;
    if max_x <= min_x || max_y <= min_y {
        return None;
    }
    Some((min_x as i32, min_y as i32, max_x - min_x, max_y - min_y))
}

/// Copy a `[pw, ph]` RGBA8 patch into `shadow` (a `stride`-wide RGBA8 image) at `[px, py]`.
fn blit(
    shadow: &mut [u8],
    stride: usize,
    [px, py]: [usize; 2],
    [pw, ph]: [usize; 2],
    patch: &[u8],
) {
    for row in 0..ph {
        let src = &patch[row * pw * 4..(row + 1) * pw * 4];
        let dst_start = ((py + row) * stride + px) * 4;
        shadow[dst_start..dst_start + pw * 4].copy_from_slice(src);
    }
}

/// Allocate a sampled RGBA8 texture and register a bindless view. `pixels` becomes the CPU shadow;
/// the caller uploads it via [`upload_full`].
fn create_texture(
    device: &Device,
    width: u32,
    height: u32,
    pixels: Vec<u8>,
) -> RhiResult<ManagedTexture> {
    let desc = TextureDesc {
        width,
        height,
        depth: 1,
        mip_levels: 1,
        array_layers: 1,
        format: Format::R8G8B8A8Unorm,
        dimension: TextureDimension::D2,
        sample_count: SampleCount::S1,
        usage: TextureUsage::SAMPLED | TextureUsage::TRANSFER_DST,
        label: Some("egui-texture".into()),
    };
    let sa = device.texture_size_align(&desc)?;
    let mem = device.allocate_aligned(sa.size, sa.align, MemoryType::GpuOnly)?;
    let texture = device.create_texture(&desc, mem.gpu())?;
    let handle = texture.gpu();
    Ok(ManagedTexture {
        texture,
        mem,
        handle,
        width,
        height,
        shadow: pixels,
    })
}

/// Upload a managed texture's full CPU shadow to the GPU synchronously. egui texture changes are
/// infrequent, so a submit+wait here keeps the painter simple without stalling steady-state frames.
fn upload_full(device: &Device, m: &ManagedTexture) -> RhiResult<()> {
    let staging = device.upload_slice(&m.shadow)?;
    let mut cmd = device.create_command_buffer()?;
    cmd.copy_buffer_to_texture(staging.gpu(), &m.texture);
    cmd.barrier(StageFlags::TRANSFER, StageFlags::ALL_COMMANDS);
    cmd.end();
    let queue = device.queue();
    queue.submit(cmd)?;
    queue.wait_idle();
    device.destroy(staging);
    Ok(())
}
