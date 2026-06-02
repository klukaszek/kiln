use kiln_rhi::{
    AccelerationStructure, BlasDesc, BlasMeshDesc, BlendState, BufferDesc, BuildAccelFlags,
    BumpAllocator, ColorTarget, CommandBuffer, CompareOp, ComputePso, ComputePsoDesc, DepthFlags,
    DepthStencilState, Device, Format, GeometryFlags, GeometryType, GpuAddress, GpuAllocation,
    GraphicsPso, GraphicsPsoDesc, MemoryType, SampleCount, ShaderStage, StageFlags, TlasDesc,
    TlasInstance, Topology, gpu_struct,
};
use std::sync::atomic::{AtomicU32, Ordering};

use crate::Vertex;
use crate::common;
use crate::scene::{Material, Scene};

mod display;
mod integrator;

static TARGET_SPP_OVERRIDE: AtomicU32 = AtomicU32::new(integrator::DEFAULT_TARGET_SPP);
static SAMPLES_PER_FRAME_OVERRIDE: AtomicU32 =
    AtomicU32::new(integrator::DEFAULT_SAMPLES_PER_FRAME);

pub fn default_target_spp() -> u32 {
    integrator::DEFAULT_TARGET_SPP
}

pub fn default_samples_per_frame() -> u32 {
    integrator::DEFAULT_SAMPLES_PER_FRAME
}

pub fn set_target_spp(spp: u32) {
    TARGET_SPP_OVERRIDE.store(spp, Ordering::Relaxed);
}

pub fn set_samples_per_frame(samples_per_frame: u32) {
    SAMPLES_PER_FRAME_OVERRIDE.store(samples_per_frame, Ordering::Relaxed);
}

fn configured_target_spp() -> u32 {
    TARGET_SPP_OVERRIDE.load(Ordering::Relaxed)
}

fn configured_samples_per_frame() -> u32 {
    SAMPLES_PER_FRAME_OVERRIDE.load(Ordering::Relaxed)
}

gpu_struct! {
    pub struct GpuMaterial {
        base_roughness: [f32; 4] as "float4",
        emission_metallic: [f32; 4] as "float4",
        specular_ior: [f32; 4] as "float4",
        coat_opacity: [f32; 4] as "float4",
        flags: [u32; 4] as "uint4",
    }
}

gpu_struct! {
    struct TraceRoot {
        cam_pos: [f32; 4] as "float4",
        cam_right: [f32; 4] as "float4",
        cam_up: [f32; 4] as "float4",
        cam_forward: [f32; 4] as "float4",
        lens: [f32; 4] as "float4",
        accum: GpuAddress as "float4*",
        verts: GpuAddress as "Vertex*",
        triangle_materials: GpuAddress as "uint*",
        materials: GpuAddress as "GpuMaterial*",
        light_triangles: GpuAddress as "uint*",
        _ptr_pad: [u32; 2] as "uint2",
        dims0: [u32; 4] as "uint4", // width, height, sample_index, max_spp
        dims1: [u32; 4] as "uint4", // tri_count, light_count, samples_per_frame, unused
    }
}

gpu_struct! {
    struct DisplayRoot {
        dims: [u32; 4] as "uint4", // width, height, sample_count, target_is_srgb
        accum: GpuAddress as "float4*",
    }
}

pub struct RayScene {
    triangle_material_buffer: GpuAllocation,
    material_buffer: GpuAllocation,
    light_triangle_buffer: GpuAllocation,
    _instance_buffer: GpuAllocation,
    _blas: AccelerationStructure,
    tlas: AccelerationStructure,
    trace_pso: ComputePso,
    display_pso: GraphicsPso,
    trace_root: GpuAllocation,
    display_bump: BumpAllocator,
    accum: Option<GpuAllocation>,
    extent: [u32; 2],
    target_spp: u32,
    samples_per_frame: u32,
    display_target_is_srgb: bool,
    sample_count: u32,
    triangle_count: u32,
    light_count: u32,
    material_count: u32,
}

impl RayScene {
    pub fn build(
        device: &Device,
        color_format: Format,
        scene: &Scene,
        vertex_buffer: &GpuAllocation,
    ) -> anyhow::Result<Self> {
        let triangle_count = scene.triangle_count();
        if triangle_count == 0 {
            anyhow::bail!("scene has no triangles");
        }
        let target_spp = configured_target_spp();
        let samples_per_frame = configured_samples_per_frame();
        let display_target_is_srgb = format_is_srgb(color_format);

        let trace_src = format!(
            "{}{}{}",
            Vertex::SLANG,
            GpuMaterial::SLANG,
            TraceRoot::SLANG
        ) + &integrator::source();
        let trace_shader = common::compile_with_caps(
            device,
            &trace_src,
            "traceMain",
            ShaderStage::Compute,
            &["spvRayQueryKHR"],
        );
        let trace_pso = device.create_compute_pso(
            &ComputePsoDesc {
                root_constant_size: std::mem::size_of::<GpuAddress>() as u32,
                threads_per_threadgroup: [integrator::THREADS_X, integrator::THREADS_Y, 1],
                label: Some("cornell-trace".into()),
            },
            &trace_shader,
        )?;

        let display_src = format!("{}{}", DisplayRoot::SLANG, display::SOURCE);
        let display_vs = common::compile(device, &display_src, "displayVs", ShaderStage::Vertex);
        let display_fs = common::compile(device, &display_src, "displayFs", ShaderStage::Pixel);
        let display_pso = device.create_graphics_pso(
            &GraphicsPsoDesc {
                topology: Topology::TriangleList,
                color_targets: vec![ColorTarget::new(color_format)],
                depth_format: Some(Format::D32Float),
                sample_count: SampleCount::S1,
                root_constant_size: 16,
                cull: kiln_rhi::Cull::None,
                blendstate: Some(BlendState::default()),
                label: Some("cornell-display".into()),
                ..Default::default()
            },
            &display_vs,
            &display_fs,
        )?;

        let triangle_material_buffer = upload_slice(
            device,
            &scene.triangle_materials,
            "triangle material buffer",
        );

        let gpu_materials: Vec<GpuMaterial> = scene
            .materials
            .iter()
            .copied()
            .map(material_to_gpu)
            .collect();
        let material_buffer = upload_slice(device, &gpu_materials, "material buffer");

        let light_triangles: Vec<u32> = scene
            .triangle_materials
            .iter()
            .enumerate()
            .filter_map(|(tri, &mat)| {
                scene
                    .materials
                    .get(mat as usize)
                    .is_some_and(|m| m.is_emissive())
                    .then_some(tri as u32)
            })
            .collect();
        let light_triangle_buffer = upload_slice(device, &light_triangles, "light triangle buffer");

        let blas_desc = BlasDesc {
            meshes: vec![BlasMeshDesc {
                geometry_type: GeometryType::Triangles,
                flags: GeometryFlags::OPAQUE,
                vertex_buffer: vertex_buffer.gpu(),
                vertex_stride: std::mem::size_of::<Vertex>() as u64,
                vertex_count: scene.vertices.len() as u32,
                index_buffer: GpuAddress(0),
                index_count: 0,
                aabb_buffer: GpuAddress(0),
                aabb_count: 0,
            }],
            flags: BuildAccelFlags::PREFER_FAST_TRACE,
        };
        let blas = device.create_blas(&blas_desc)?;
        {
            let mut cmd = device.create_command_buffer()?;
            cmd.build_blas(&blas, &blas_desc);
            cmd.end();
            let queue = device.queue();
            queue.submit(cmd)?;
            queue.wait_idle();
        }

        let instance_buffer = device
            .malloc(device.tlas_instance_stride() as u64, MemoryType::Default)
            .expect("alloc TLAS instance buffer");
        device.write_tlas_instance(
            &instance_buffer,
            0,
            &TlasInstance {
                transform: [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                ],
                instance_custom_index_and_mask: 0xFF << 24,
                instance_sbt_offset_and_flags: 0,
                acceleration_structure_reference: blas.gpu(),
            },
        )?;

        let tlas_desc = TlasDesc {
            instance_buffer: instance_buffer.gpu(),
            instance_count: 1,
            flags: BuildAccelFlags::PREFER_FAST_TRACE,
        };
        let tlas = device.create_tlas(&tlas_desc)?;
        {
            let mut cmd = device.create_command_buffer()?;
            cmd.build_tlas(&tlas, &tlas_desc);
            cmd.end();
            let queue = device.queue();
            queue.submit(cmd)?;
            queue.wait_idle();
        }

        let trace_root = device
            .malloc(std::mem::size_of::<TraceRoot>() as u64, MemoryType::Default)
            .expect("alloc trace root");
        let display_bump = BumpAllocator::new(
            device
                .create_buffer(&BufferDesc {
                    size: 4096,
                    memory: MemoryType::Default,
                    label: Some("cornell-display-bump".into()),
                })
                .expect("create display bump"),
        );

        eprintln!(
            "cornell ray scene: {triangle_count} triangles, {} materials, {} emissive triangles, target spp={target_spp}, samples/frame={samples_per_frame}",
            gpu_materials.len(),
            light_triangles.len()
        );

        Ok(Self {
            triangle_material_buffer,
            material_buffer,
            light_triangle_buffer,
            _instance_buffer: instance_buffer,
            _blas: blas,
            tlas,
            trace_pso,
            display_pso,
            trace_root,
            display_bump,
            accum: None,
            extent: [0, 0],
            target_spp,
            samples_per_frame,
            display_target_is_srgb,
            sample_count: 0,
            triangle_count,
            light_count: light_triangles.len() as u32,
            material_count: gpu_materials.len() as u32,
        })
    }

    pub fn pre_render(
        &mut self,
        device: &Device,
        cmd: &mut CommandBuffer,
        scene: &Scene,
        vertex_buffer: &GpuAllocation,
        extent: [u32; 2],
    ) {
        self.ensure_extent(device, extent);
        if self.sample_count >= self.target_spp || self.light_count == 0 {
            return;
        }

        let samples = self
            .samples_per_frame
            .min(self.target_spp - self.sample_count);
        let accum = self.accum.as_ref().expect("accum buffer");
        let camera = CameraGpu::from_scene(scene, extent);
        self.trace_root
            .upload(&TraceRoot {
                cam_pos: camera.pos,
                cam_right: camera.right,
                cam_up: camera.up,
                cam_forward: camera.forward,
                lens: camera.lens,
                accum: accum.gpu(),
                verts: vertex_buffer.gpu(),
                triangle_materials: self.triangle_material_buffer.gpu(),
                materials: self.material_buffer.gpu(),
                light_triangles: self.light_triangle_buffer.gpu(),
                _ptr_pad: [0; 2],
                dims0: [extent[0], extent[1], self.sample_count, self.target_spp],
                dims1: [self.triangle_count, self.light_count, samples, 0],
            })
            .expect("upload trace root");

        cmd.set_compute_pipeline(&self.trace_pso);
        cmd.bind_acceleration_structure(1, &self.tlas);
        cmd.dispatch(
            self.trace_root.gpu(),
            extent[0].div_ceil(integrator::THREADS_X),
            extent[1].div_ceil(integrator::THREADS_Y),
            1,
        );
        self.sample_count += samples;
        self.log_progress();
        cmd.barrier(StageFlags::COMPUTE, StageFlags::PIXEL_SHADER);
    }

    pub fn render(&mut self, cmd: &mut CommandBuffer, extent: [u32; 2]) {
        let Some(accum) = self.accum.as_ref() else {
            return;
        };
        if self.extent != extent {
            return;
        }

        self.display_bump.reset();
        let root = self
            .display_bump
            .alloc(std::mem::size_of::<DisplayRoot>() as u64, 16)
            .expect("display root bump");
        root.upload(&DisplayRoot {
            dims: [
                extent[0],
                extent[1],
                self.sample_count.max(1),
                self.display_target_is_srgb as u32,
            ],
            accum: accum.gpu(),
        })
        .expect("upload display root");

        cmd.set_graphics_pipeline(&self.display_pso);
        cmd.set_depth_stencil_state(&DepthStencilState {
            depth_mode: DepthFlags::empty(),
            depth_test: CompareOp::Always,
            stencil_read_mask: 0,
            stencil_write_mask: 0,
            ..Default::default()
        });
        cmd.draw(None, root.gpu, 3, 1, 0, 0);
    }

    pub fn is_complete(&self) -> bool {
        self.sample_count >= self.target_spp
    }

    pub fn sample_count(&self) -> u32 {
        self.sample_count
    }

    pub fn target_spp(&self) -> u32 {
        self.target_spp
    }

    pub fn samples_per_frame(&self) -> u32 {
        self.samples_per_frame
    }

    pub fn light_count(&self) -> u32 {
        self.light_count
    }

    pub fn extent(&self) -> [u32; 2] {
        self.extent
    }

    pub fn tonemapped_rgba8(&self) -> anyhow::Result<Vec<u8>> {
        let accum = self
            .accum
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("path tracer has no accumulation buffer"))?;
        let pixels = accum.as_slice::<[f32; 4]>()?;
        let sample_count = self.sample_count.max(1) as f32;
        let mut rgba = Vec::with_capacity(pixels.len() * 4);

        for pixel in pixels {
            let mapped = [
                tonemap_channel(pixel[0], sample_count),
                tonemap_channel(pixel[1], sample_count),
                tonemap_channel(pixel[2], sample_count),
            ];
            rgba.extend_from_slice(&[mapped[0], mapped[1], mapped[2], 255]);
        }

        Ok(rgba)
    }

    fn ensure_extent(&mut self, device: &Device, extent: [u32; 2]) {
        if self.extent == extent && self.accum.is_some() {
            return;
        }

        let pixels = (extent[0] as u64) * (extent[1] as u64);
        let accum = device
            .malloc(
                pixels * std::mem::size_of::<[f32; 4]>() as u64,
                MemoryType::Default,
            )
            .expect("alloc accumulation buffer");
        let zeros = vec![[0.0f32; 4]; pixels as usize];
        accum.upload_slice(&zeros).expect("clear accumulation");

        self.accum = Some(accum);
        self.extent = extent;
        self.sample_count = 0;
        eprintln!(
            "cornell path tracer reset: {}x{}, target spp={}, samples/frame={}, materials={}, lights={}",
            extent[0],
            extent[1],
            self.target_spp,
            self.samples_per_frame,
            self.material_count,
            self.light_count
        );
    }

    fn log_progress(&self) {
        if self.sample_count == self.target_spp
            || (self.sample_count >= 64 && self.sample_count.is_power_of_two())
        {
            eprintln!(
                "cornell path tracer progress: {}/{} spp",
                self.sample_count, self.target_spp
            );
        }
    }
}

struct CameraGpu {
    pos: [f32; 4],
    right: [f32; 4],
    up: [f32; 4],
    forward: [f32; 4],
    lens: [f32; 4],
}

impl CameraGpu {
    fn from_scene(scene: &Scene, extent: [u32; 2]) -> Self {
        let m = &scene.camera.world;
        let aspect = extent[0] as f32 / extent[1].max(1) as f32;
        let tan_half_fovy = (scene.camera.usd.vertical_fov_rad() * 0.5).tan();
        let forward = normalize3([-(m[8] as f32), -(m[9] as f32), -(m[10] as f32)]);

        Self {
            pos: [m[12] as f32, m[13] as f32, m[14] as f32, 1.0],
            right: normalize4([m[0] as f32, m[1] as f32, m[2] as f32, 0.0]),
            up: normalize4([m[4] as f32, m[5] as f32, m[6] as f32, 0.0]),
            forward: [forward[0], forward[1], forward[2], 0.0],
            lens: [aspect, tan_half_fovy, 0.0, 0.0],
        }
    }
}

fn upload_slice<T: kiln_rhi::GpuPod>(device: &Device, data: &[T], label: &str) -> GpuAllocation {
    let size = (data.len() * std::mem::size_of::<T>()).max(1) as u64;
    let buffer = device.malloc(size, MemoryType::Default).expect(label);
    if !data.is_empty() {
        buffer.upload_slice(data).expect(label);
    }
    buffer
}

fn material_to_gpu(material: Material) -> GpuMaterial {
    GpuMaterial {
        base_roughness: [
            material.base_color[0],
            material.base_color[1],
            material.base_color[2],
            material.roughness,
        ],
        emission_metallic: [
            material.emission[0],
            material.emission[1],
            material.emission[2],
            material.metallic,
        ],
        specular_ior: [
            material.specular_color[0],
            material.specular_color[1],
            material.specular_color[2],
            material.ior,
        ],
        coat_opacity: [
            material.clearcoat,
            material.clearcoat_roughness,
            material.opacity,
            material.opacity_threshold,
        ],
        flags: [material.use_specular_workflow as u32, 0, 0, 0],
    }
}

fn tonemap_channel(value: f32, sample_count: f32) -> u8 {
    let linear = (value / sample_count) * 0.25;
    let reinhard = linear / (linear + 1.0);
    let srgb = reinhard.max(0.0).powf(1.0 / 2.2);
    (srgb.clamp(0.0, 1.0) * 255.0).round() as u8
}

fn format_is_srgb(format: Format) -> bool {
    matches!(format, Format::R8G8B8A8Srgb | Format::B8G8R8A8Srgb)
}

fn normalize4(v: [f32; 4]) -> [f32; 4] {
    let n = normalize3([v[0], v[1], v[2]]);
    [n[0], n[1], n[2], v[3]]
}

fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len > 1e-8 {
        [v[0] / len, v[1] / len, v[2] / len]
    } else {
        [0.0, 0.0, -1.0]
    }
}
