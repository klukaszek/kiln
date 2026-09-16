//! Holographic Radiance Cascades: allocation, the pass schedule, and the per-dispatch roots.
//!
//! Standard radiance cascades halve probe resolution in *both* axes per level, so a distant thin
//! occluder is resolved on a grid coarser than the occluder itself. HRC reduces resolution only
//! *along* the gathered direction. Fluence splits into four quadrants, each solved in a canonical
//! frame whose +x axis is the gather direction, then summed.
//!
//! Quadrants 0/2 and 1/3 gather along opposite axes, so they need differently shaped buffers. The
//! paper notes a square domain lets all four share one set; on a non-square window that pads every
//! cascade out to `max(w, h)^2`, which is wasted bandwidth in the passes that dominate the frame.
//! Two sets cost about 1.5x the memory and save that work, so the cascades are stored per parity.

mod program;
mod resources;

use glam::{IVec2, UVec2, Vec4};
use kiln_rhi::{
    AllocationDesc, BumpAllocator, CommandBuffer, Device, Format, GpuPtr, MemoryType, RhiResult,
    StageFlags, MAX_FRAMES_IN_FLIGHT,
};

use crate::scene::{self, Prim};

use program::{
    ClearRoot, Pipelines, ReferenceRoot, Root, CASCADE_THREADS, CLEAR_THREADS, CONE_STRIDE,
    EDGES_PER_PRIM, FIELD_THREADS, GRID_CELLS, REFERENCE_EPSILON, REFERENCE_MAX_STEPS,
    SEGMENT_STRIDE,
};
pub use program::{DIRECT_TRACE_LEVELS, REFERENCE_DIRECTIONS};
use resources::SceneResources;

/// Ceiling on cascade storage. The domain grows as `N * X^2`, so a large window with a generous
/// probe budget can ask for more than a GPU has; [`plan_resolution`] backs the probe grid off until
/// the field fits.
pub const MEMORY_BUDGET_BYTES: u64 = 1024 * 1024 * 1024;

/// Smallest probe grid the budget is allowed to back off to.
const MIN_RESOLUTION: u32 = 64;

/// Per-frame root data. About a hundred dispatches at 256 bytes each, so this is generous.
const ARENA_BYTES: u64 = 128 * 1024;

/// Where one frame's work goes: the device, the frame-in-flight slot whose transients it may
/// overwrite, the probe grid it solves on, and the resolution it resolves to.
pub struct Frame<'a> {
    pub device: &'a Device,
    pub slot: usize,
    pub res: UVec2,
    pub out_res: UVec2,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ViewMode {
    Final,
    LightField,
}

#[derive(Clone, Copy, Debug)]
pub struct Settings {
    /// Probe budget in thousands. Cost is set by how many probes exist, so budgeting by count
    /// rather than by a fixed side keeps frame time from swinging with the window's aspect.
    pub probes_thousands: u32,
    pub blur_passes: u32,
    pub surface_falloff: f32,
    pub bounce: f32,
    pub exposure: f32,
    pub sky: f32,
    pub view_mode: ViewMode,
    pub surface_shading: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            // The knee of the quality curve, measured against the brute-force reference at fixed
            // output resolution: past here the error is dominated by the method rather than by
            // probe density, and doubling to 315k costs 75% more frame time to take relative L1
            // from 0.054 to 0.043. The original's 315k default was chosen to fill a 120 Hz budget
            // rather than because the solver needed it.
            probes_thousands: 150,
            blur_passes: 2,
            surface_falloff: 0.02,
            bounce: 1.0,
            exposure: 1.0,
            sky: 0.0,
            view_mode: ViewMode::Final,
            surface_shading: true,
        }
    }
}

/// What a light field of this resolution costs.
#[derive(Clone, Copy, Debug)]
pub struct Plan {
    /// `T` and `R` storage, at half precision.
    pub cascade_bytes: u64,
    /// The resolved field and its two blur scratch buffers, at full precision.
    pub field_bytes: u64,
}

impl Plan {
    pub fn bytes(&self) -> u64 {
        self.cascade_bytes + self.field_bytes
    }
}

/// Probe columns *stored* at level `n`. The construction reads one past the end; that column lies
/// outside the domain and is evaluated in the shader rather than stored. See `levelWidth` there.
fn level_width(side: u32, n: u32) -> u32 {
    (side + (1 << n) - 1) >> n
}

/// Levels needed to reach across `canon_x` probes: `ceil(log2(canon_x))`, at least one.
fn levels_for(canon_x: u32) -> u32 {
    let v = canon_x.max(2);
    (u32::BITS - (v - 1).leading_zeros()).max(1)
}

/// (X, Y) of the canonical frame for a parity; X is that quadrant pair's gather axis.
fn canon(res: UVec2, parity: usize) -> UVec2 {
    if parity == 0 {
        res
    } else {
        UVec2::new(res.y, res.x)
    }
}

pub fn plan(res: UVec2) -> Plan {
    // Mirrors the layout `configure` builds, padding included, so the budget and the allocation
    // cannot disagree.
    let level = |width: u32, height: u32, dirs: u32, stride: u64| {
        (u64::from(width) * u64::from(height) * u64::from(dirs) * stride).next_multiple_of(16)
    };
    let mut cascade_bytes: u64 = 0;
    for parity in 0..2 {
        let c = canon(res, parity);
        let levels = levels_for(c.x);
        for n in 0..=levels {
            cascade_bytes += level(level_width(c.x, n), c.y, (1 << n) + 1, SEGMENT_STRIDE);
            // R_0 is stored per quadrant, and R_N is uniform and evaluated in the shader.
            if n > 0 && n < levels {
                cascade_bytes += level(level_width(c.x, n), c.y, 1 << n, CONE_STRIDE);
            }
        }
    }
    // R_0, one buffer per quadrant.
    for quadrant in 0..4 {
        let c = canon(res, quadrant & 1);
        cascade_bytes += level(level_width(c.x, 0), c.y, 1, CONE_STRIDE);
    }
    Plan {
        cascade_bytes,
        field_bytes: 3 * u64::from(res.x) * u64::from(res.y) * 16,
    }
}

/// Probe grid for a window of `extent`, honouring the budget and backing off until the cascades
/// fit in [`MEMORY_BUDGET_BYTES`]. Returns the grid and the probe count it actually represents.
pub fn plan_resolution(extent: UVec2, probes_thousands: u32) -> (UVec2, u32) {
    let pixels = f64::from(extent.x.max(1)) * f64::from(extent.y.max(1));
    let scale = (f64::from(probes_thousands.max(1)) * 1000.0 / pixels).sqrt();
    let axis = |v: u32| ((f64::from(v) * scale) as u32).max(MIN_RESOLUTION);
    let mut res = UVec2::new(axis(extent.x), axis(extent.y));
    while plan(res).bytes() > MEMORY_BUDGET_BYTES && res.min_element() > MIN_RESOLUTION {
        res = (res / 2).max(UVec2::splat(MIN_RESOLUTION));
    }
    (res, (res.x * res.y / 1000).max(1))
}

/// One cascade level's storage: where it starts and the shape the index arithmetic assumes.
#[derive(Clone, Copy)]
struct Level {
    base: GpuPtr<u8>,
    width: u32,
    /// Directions for a `T` level, cones for an `R` level.
    dirs: u32,
}

impl Level {
    fn entries(&self, height: u32) -> u64 {
        u64::from(self.width) * u64::from(height) * u64::from(self.dirs)
    }
}

/// The cascades for one gather parity.
#[derive(Default)]
struct Parity {
    levels: u32,
    canon: UVec2,
    segments: Vec<Level>,
    /// Indexed by level; entry 0 is unused, because R_0 is stored per quadrant instead.
    cones: Vec<Level>,
}

pub struct HrcRenderer {
    pipelines: Pipelines,
    resources: SceneResources,
    arenas: Vec<BumpAllocator>,
    /// One allocation behind every cascade level; sub-buffers are offsets into it.
    cascades: Option<kiln_rhi::Allocation>,
    /// The resolved field, plus the two blur scratch buffers.
    field: Option<kiln_rhi::Allocation>,
    parities: [Parity; 2],
    quadrant_cones: [GpuPtr<u8>; 4],
    field_buffers: [GpuPtr<Vec4>; 3],
    /// The buffer holding the field as it will be displayed, which is also what the next frame's
    /// surfaces re-emit.
    resolved: GpuPtr<Vec4>,
    res: UVec2,
    out_res: UVec2,
    target_is_srgb: bool,
    packed: Vec<Vec4>,
    /// Root for the resolve draw, uploaded during `record` and consumed in the render pass.
    resolve_root: GpuPtr<Root>,
    pub plan: Plan,
    pub traced_rays: u64,
}

impl HrcRenderer {
    pub fn new(device: &Device, color_format: Format) -> RhiResult<Self> {
        let pipelines = Pipelines::new(device, color_format)?;
        let resources = SceneResources::new(device)?;
        let mut arenas = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        for slot in 0..MAX_FRAMES_IN_FLIGHT {
            arenas.push(BumpAllocator::new(device.create_allocation(
                &AllocationDesc {
                    size: ARENA_BYTES,
                    memory: MemoryType::Upload,
                    label: Some(format!("hrc-frame-roots-{slot}")),
                    ..Default::default()
                },
            )?));
        }

        Ok(Self {
            pipelines,
            resources,
            arenas,
            cascades: None,
            field: None,
            parities: [Parity::default(), Parity::default()],
            quadrant_cones: [GpuPtr::NULL; 4],
            field_buffers: [GpuPtr::NULL; 3],
            resolved: GpuPtr::NULL,
            res: UVec2::ZERO,
            out_res: UVec2::ZERO,
            target_is_srgb: matches!(color_format, Format::R8G8B8A8Srgb | Format::B8G8R8A8Srgb),
            packed: Vec::new(),
            resolve_root: GpuPtr::NULL,
            plan: plan(UVec2::splat(MIN_RESOLUTION)),
            traced_rays: 0,
        })
    }

    pub fn resolution(&self) -> UVec2 {
        self.res
    }

    pub fn output_resolution(&self) -> UVec2 {
        self.out_res
    }

    pub fn levels_per_axis(&self) -> (u32, u32) {
        (self.parities[0].levels + 1, self.parities[1].levels + 1)
    }

    // --- allocation ---------------------------------------------------------

    /// Lay the cascades out in one allocation. Every level is fully written before it is read each
    /// frame, so only the fluence field needs clearing.
    fn configure(
        &mut self,
        device: &Device,
        cmd: &mut CommandBuffer,
        slot: usize,
        res: UVec2,
    ) -> RhiResult<()> {
        if res == self.res {
            return Ok(());
        }

        self.plan = plan(res);
        let field_entries = u64::from(res.x) * u64::from(res.y);

        let cascades = device.create_allocation(&AllocationDesc {
            size: self.plan.cascade_bytes,
            memory: MemoryType::GpuOnly,
            label: Some("hrc-cascades".into()),
            ..Default::default()
        })?;
        let field = device.create_allocation(&AllocationDesc {
            size: self.plan.field_bytes,
            memory: MemoryType::GpuOnly,
            label: Some("hrc-fluence-field".into()),
            ..Default::default()
        })?;

        let base = cascades.gpu();
        let mut offset = 0u64;
        // Each level is padded to sixteen bytes so a six-byte `R` element never leaves the next
        // level's base misaligned.
        let mut take = |width: u32, height: u32, dirs: u32, stride: u64| {
            let level = Level {
                base: base.byte_add(offset),
                width,
                dirs,
            };
            offset += (level.entries(height) * stride).next_multiple_of(16);
            level
        };

        for parity in 0..2 {
            let c = canon(res, parity);
            let levels = levels_for(c.x);
            let mut segments = Vec::with_capacity(levels as usize + 1);
            let mut cones = Vec::with_capacity(levels as usize + 1);
            for n in 0..=levels {
                segments.push(take(level_width(c.x, n), c.y, (1 << n) + 1, SEGMENT_STRIDE));
            }
            // Two levels have no storage: R_0 lives in the per-quadrant buffers below, and R_N is
            // uniform so the sweep evaluates it. Both keep a slot, so `cones[n]` still indexes by
            // level everywhere else.
            let placeholder = Level {
                base: GpuPtr::NULL,
                width: 0,
                dirs: 0,
            };
            cones.push(placeholder);
            for n in 1..levels {
                cones.push(take(level_width(c.x, n), c.y, 1 << n, CONE_STRIDE));
            }
            cones.push(placeholder);
            self.parities[parity] = Parity {
                levels,
                canon: c,
                segments,
                cones,
            };
        }
        for quadrant in 0..4 {
            let c = canon(res, quadrant & 1);
            self.quadrant_cones[quadrant] = take(level_width(c.x, 0), c.y, 1, CONE_STRIDE).base;
        }
        debug_assert!(offset <= cascades.size());

        let field_base = field.gpu().cast::<Vec4>();
        for (i, slot) in self.field_buffers.iter_mut().enumerate() {
            *slot = field_base.byte_add(field_entries * 16 * i as u64);
        }
        self.resolved = self.field_buffers[0];

        if let Some(previous) = self.cascades.replace(cascades) {
            device.destroy(previous);
        }
        if let Some(previous) = self.field.replace(field) {
            device.destroy(previous);
        }
        self.res = res;

        // The field is read a frame before it is first written: surfaces re-emit whatever reached
        // them last frame, and on the first frame that is this buffer's uninitialised contents.
        self.clear(cmd, slot, field_base, self.plan.field_bytes / 4);
        cmd.barrier(StageFlags::COMPUTE, StageFlags::COMPUTE);
        Ok(())
    }

    fn clear(&self, cmd: &mut CommandBuffer, slot: usize, target: GpuPtr<Vec4>, u32_count: u64) {
        let count = u32_count as u32;
        let root = self.arenas[slot]
            .alloc(size_of::<ClearRoot>() as u64, 16)
            .expect("frame arena exhausted")
            .cast::<ClearRoot>();
        root.write(&ClearRoot {
            target: target.cast(),
            count,
            pad: 0,
        })
        .expect("write clear root");
        cmd.set_pipeline(&self.pipelines.clear);
        cmd.dispatch(root.gpu(), count.div_ceil(CLEAR_THREADS), 1, 1);
    }

    // --- dispatch -----------------------------------------------------------

    /// Record one frame: scene upload, tessellation, the acceleration build, the cascade sweep, and
    /// the resolve into the fluence field. The image itself is drawn by [`Self::record_resolve`]
    /// inside the harness's render pass.
    pub fn record(
        &mut self,
        frame: &Frame<'_>,
        cmd: &mut CommandBuffer,
        prims: &[Prim],
        settings: Settings,
    ) -> RhiResult<()> {
        let Frame {
            device,
            slot,
            res,
            out_res,
        } = *frame;
        self.arenas[slot].reset();
        self.out_res = out_res;

        // Order this frame's writes after the previous in-flight frame's reads. Every cascade and
        // field buffer is shared across frames, and recorded outside an encoder this lands as a
        // queue-scoped barrier at the head of the first one.
        cmd.barrier(StageFlags::ALL_COMMANDS, StageFlags::ALL_COMMANDS);

        self.configure(device, cmd, slot, res)?;

        self.upload_scene(cmd, slot, prims);
        let base = self.base_root(settings);
        self.record_scene_build(cmd, slot, &base);
        self.traced_rays = self.record_cascades(cmd, slot, &base);
        self.record_field(cmd, slot, &base, settings);
        Ok(())
    }

    /// Draw the resolved field. Runs inside the harness's swapchain render pass.
    pub fn record_resolve(&self, cmd: &mut CommandBuffer) {
        cmd.set_pipeline(&self.pipelines.resolve);
        cmd.draw(self.resolve_root, 3, 1, 0, 0);
    }

    /// Pack the scene into the frame arena and stage it into device memory. Reading primitives
    /// straight out of upload memory would put a host-visible fetch in the inner loop of every
    /// distance query.
    fn upload_scene(&mut self, cmd: &mut CommandBuffer, slot: usize, prims: &[Prim]) {
        scene::pack(prims, (self.res.x, self.res.y), &mut self.packed);
        self.resources.prim_count = prims.len() as u32;

        let bytes = std::mem::size_of_val(self.packed.as_slice()) as u64;
        let staging = self.arenas[slot]
            .alloc(bytes, 16)
            .expect("frame arena exhausted")
            .cast::<Vec4>();
        staging.write_slice(&self.packed).expect("write scene");
        cmd.memcpy(self.resources.scene.gpu(), staging.gpu(), bytes);
        cmd.barrier(StageFlags::TRANSFER, StageFlags::COMPUTE);
    }

    fn record_scene_build(&self, cmd: &mut CommandBuffer, slot: usize, base: &Root) {
        let root = self.upload(slot, base);
        cmd.set_pipeline(&self.pipelines.tessellate);
        cmd.dispatch(
            root,
            (self.resources.prim_count * EDGES_PER_PRIM).div_ceil(CASCADE_THREADS),
            1,
            1,
        );
        cmd.set_pipeline(&self.pipelines.build_grid);
        cmd.dispatch(
            root,
            GRID_CELLS.div_ceil(FIELD_THREADS[0]),
            GRID_CELLS.div_ceil(FIELD_THREADS[1]),
            1,
        );

        // The acceleration build reads what the tessellation just wrote, from its own encoder, so
        // this has to be queue-scoped rather than encoder-scoped.
        cmd.barrier(StageFlags::COMPUTE, StageFlags::ALL_COMMANDS);
        cmd.build_blas(self.resources.blas(), &self.resources.blas_desc());
        cmd.build_tlas(&self.resources.tlas, &self.resources.tlas_desc());
        cmd.barrier(StageFlags::ALL_COMMANDS, StageFlags::COMPUTE);
    }

    /// `T` bottom-up then `R` top-down, once per quadrant. The two quadrants of a parity share the
    /// same cascade buffers and recompute them, which is why each quadrant's sweep runs to
    /// completion before the next one starts.
    fn record_cascades(&self, cmd: &mut CommandBuffer, slot: usize, base: &Root) -> u64 {
        let mut traced_rays = 0;

        for quadrant in 0..4usize {
            let p = &self.parities[quadrant & 1];
            let levels = p.levels;
            let quad = Root {
                quadrant: quadrant as i32,
                canon_res: p.canon.as_ivec2(),
                levels: levels as i32,
                ..*base
            };

            // T, bottom-up. Only the first few levels touch the scene, and those segments span at
            // most four probes; the levels are independent, so one barrier covers them all.
            let traced = DIRECT_TRACE_LEVELS.min(levels + 1);
            cmd.set_pipeline(&self.pipelines.trace_segments);
            for n in 0..traced {
                let level = p.segments[n as usize];
                traced_rays += level.entries(p.canon.y);
                let root = self.upload(
                    slot,
                    &Root {
                        level: n as i32,
                        segments_out: level.base.cast(),
                        ..quad
                    },
                );
                self.dispatch_level(cmd, root, level, p.canon.y);
            }
            cmd.barrier(StageFlags::COMPUTE, StageFlags::COMPUTE);

            cmd.set_pipeline(&self.pipelines.build_segments);
            for n in traced..=levels {
                let level = p.segments[n as usize];
                let root = self.upload(
                    slot,
                    &Root {
                        level: n as i32,
                        segments_out: level.base.cast(),
                        segments_prev: p.segments[n as usize - 1].base.cast(),
                        ..quad
                    },
                );
                self.dispatch_level(cmd, root, level, p.canon.y);
                cmd.barrier(StageFlags::COMPUTE, StageFlags::COMPUTE);
            }

            // R, top-down. No rays. Level 0 lands in this quadrant's own buffer, because that is
            // the one the resolve gathers from.
            cmd.set_pipeline(&self.pipelines.build_fluence);
            for n in (0..levels).rev() {
                let level = if n == 0 {
                    Level {
                        base: self.quadrant_cones[quadrant],
                        width: level_width(p.canon.x, 0),
                        dirs: 1,
                    }
                } else {
                    p.cones[n as usize]
                };
                let root = self.upload(
                    slot,
                    &Root {
                        level: n as i32,
                        cones_out: level.base.cast(),
                        cones_next: p.cones[n as usize + 1].base.cast(),
                        segments_cur: p.segments[n as usize].base.cast(),
                        segments_next: p.segments[n as usize + 1].base.cast(),
                        ..quad
                    },
                );
                self.dispatch_level(cmd, root, level, p.canon.y);
                cmd.barrier(StageFlags::COMPUTE, StageFlags::COMPUTE);
            }
        }
        traced_rays
    }

    /// Gather the four quadrants into the fluence field, blur out the checkerboard, and upload the
    /// root the resolve draw will read.
    fn record_field(
        &mut self,
        cmd: &mut CommandBuffer,
        slot: usize,
        base: &Root,
        settings: Settings,
    ) {
        let groups = (
            self.res.x.div_ceil(FIELD_THREADS[0]),
            self.res.y.div_ceil(FIELD_THREADS[1]),
        );

        let root = self.upload(
            slot,
            &Root {
                fluence_out: self.field_buffers[0],
                ..*base
            },
        );
        cmd.set_pipeline(&self.pipelines.resolve_fluence);
        cmd.dispatch(root, groups.0, groups.1, 1);
        cmd.barrier(StageFlags::COMPUTE, StageFlags::COMPUTE);

        // The checkerboard sits at Nyquist and a single 1-4-1 cross only takes it to a third; the
        // resolve then magnifies whatever is left.
        let mut src = self.field_buffers[0];
        let mut dst = self.field_buffers[1];
        cmd.set_pipeline(&self.pipelines.blur_fluence);
        for pass in 0..settings.blur_passes {
            let root = self.upload(
                slot,
                &Root {
                    fluence_in: src,
                    fluence_out: dst,
                    ..*base
                },
            );
            cmd.dispatch(root, groups.0, groups.1, 1);
            cmd.barrier(StageFlags::COMPUTE, StageFlags::COMPUTE);
            src = dst;
            dst = self.field_buffers[1 + (pass as usize + 1) % 2];
        }
        self.resolved = src;

        self.resolve_root = self.upload(
            slot,
            &Root {
                fluence_in: src,
                view_mode: u32::from(settings.view_mode == ViewMode::LightField),
                ..*base
            },
        );
        cmd.barrier(StageFlags::COMPUTE, StageFlags::PIXEL_SHADER);
    }

    /// Cascade passes flatten (direction, column) across the threadgroup and take the row from the
    /// dispatch's second axis, so threads adjacent in x touch adjacent addresses.
    fn dispatch_level(
        &self,
        cmd: &mut CommandBuffer,
        root: GpuPtr<Root>,
        level: Level,
        height: u32,
    ) {
        cmd.dispatch(
            root,
            (level.dirs * level.width).div_ceil(CASCADE_THREADS),
            height,
            1,
        );
    }

    /// The fields every pass shares. Per-dispatch roots are this with three or four fields changed.
    fn base_root(&self, settings: Settings) -> Root {
        let sky = settings.sky;
        Root {
            sky: Vec4::new(sky * 0.6, sky * 0.75, sky, 0.0),
            scene: self.resources.scene.gpu().cast(),
            cell_count: self.resources.cell_count.gpu().cast(),
            cell_prims: self.resources.cell_prims.gpu().cast(),
            cell_clear: self.resources.cell_clear.gpu().cast(),
            tri_prim: self.resources.tri_prim.gpu().cast(),
            vertices: self.resources.vertices.gpu().cast(),
            indices: self.resources.indices.gpu().cast(),
            // Last frame's resolved field, which surfaces re-emit.
            history: self.resolved,
            fluence_in: self.resolved,
            fluence_out: self.field_buffers[0],
            segments_out: GpuPtr::NULL,
            segments_prev: GpuPtr::NULL,
            segments_cur: GpuPtr::NULL,
            segments_next: GpuPtr::NULL,
            cones_out: GpuPtr::NULL,
            cones_next: GpuPtr::NULL,
            cones0_q0: self.quadrant_cones[0].cast(),
            cones0_q1: self.quadrant_cones[1].cast(),
            cones0_q2: self.quadrant_cones[2].cast(),
            cones0_q3: self.quadrant_cones[3].cast(),
            tlas: self.resources.tlas.gpu(),
            res: self.res.as_ivec2(),
            out_res: self.out_res.as_ivec2(),
            canon_res: self.res.as_ivec2(),
            cells: IVec2::splat(GRID_CELLS as i32),
            exposure: settings.exposure,
            surface_offset: 1.5,
            surface_falloff: settings.surface_falloff,
            bounce: settings.bounce,
            quadrant: 0,
            level: 0,
            levels: self.parities[0].levels as i32,
            prim_count: self.resources.prim_count as i32,
            surface_shading: i32::from(settings.surface_shading),
            view_mode: 0,
            target_is_srgb: u32::from(self.target_is_srgb),
            pad0: 0,
            pad1: 0,
            pad2: 0,
        }
    }

    /// Write the solved field into `dst` as linear mean radiance, for comparison against the
    /// reference below. Sized by the output resolution, which verification sets equal to the probe
    /// grid.
    pub fn record_light_field_linear(
        &self,
        cmd: &mut CommandBuffer,
        slot: usize,
        settings: Settings,
        dst: GpuPtr<Vec4>,
    ) {
        let root = self.upload(slot, &self.reference_root(settings, dst, 0, 0..0));
        cmd.set_pipeline(&self.pipelines.light_field_linear);
        self.dispatch_field(cmd, root);
    }

    /// Accumulate one slice of the brute-force angular integral into `dst`. The caller clears `dst`,
    /// walks `dirs` in slices, and divides by `dirs`.
    pub fn record_reference(
        &self,
        cmd: &mut CommandBuffer,
        slot: usize,
        settings: Settings,
        dst: GpuPtr<Vec4>,
        dirs: u32,
        slice: std::ops::Range<u32>,
    ) {
        let root = self.upload(slot, &self.reference_root(settings, dst, dirs, slice));
        cmd.set_pipeline(&self.pipelines.reference_linear);
        self.dispatch_field(cmd, root);
    }

    fn reference_root(
        &self,
        settings: Settings,
        dst: GpuPtr<Vec4>,
        dirs: u32,
        slice: std::ops::Range<u32>,
    ) -> ReferenceRoot {
        let sky = settings.sky;
        ReferenceRoot {
            sky: Vec4::new(sky * 0.6, sky * 0.75, sky, 0.0),
            scene: self.resources.scene.gpu().cast(),
            fluence_in: self.resolved,
            fluence_out: dst,
            res: self.res.as_ivec2(),
            out_res: self.out_res.as_ivec2(),
            prim_count: self.resources.prim_count as i32,
            ref_dir_begin: slice.start,
            ref_dir_count: slice.len() as u32,
            ref_dirs: dirs,
            ref_max_steps: REFERENCE_MAX_STEPS,
            // Long enough to leave the domain from any point inside it.
            ref_max_dist: self.res.as_vec2().length() * 1.5,
            ref_eps: REFERENCE_EPSILON,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        }
    }

    /// Clear `count` floats of `dst`, for the reference accumulator.
    pub fn record_clear(
        &self,
        cmd: &mut CommandBuffer,
        slot: usize,
        dst: GpuPtr<Vec4>,
        floats: u64,
    ) {
        self.clear(cmd, slot, dst, floats);
    }

    pub fn reset_arena(&mut self, slot: usize) {
        self.arenas[slot].reset();
    }

    fn dispatch_field(&self, cmd: &mut CommandBuffer, root: GpuPtr<ReferenceRoot>) {
        cmd.dispatch(
            root,
            self.out_res.x.div_ceil(FIELD_THREADS[0]),
            self.out_res.y.div_ceil(FIELD_THREADS[1]),
            1,
        );
    }

    fn upload<T: kiln_rhi::GpuPod>(&self, slot: usize, root: &T) -> GpuPtr<T> {
        let allocation = self.arenas[slot]
            .alloc(size_of::<T>() as u64, 16)
            .expect("frame arena exhausted")
            .cast::<T>();
        allocation.write(root).expect("write root");
        allocation.gpu()
    }

    pub fn destroy(self, device: &Device) {
        self.resources.destroy(device);
        for arena in self.arenas {
            device.destroy(arena.into_allocation());
        }
        if let Some(cascades) = self.cascades {
            device.destroy(cascades);
        }
        if let Some(field) = self.field {
            device.destroy(field);
        }
    }
}
