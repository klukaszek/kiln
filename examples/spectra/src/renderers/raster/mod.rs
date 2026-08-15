//! Simple raster renderer backend.

mod pipeline;
mod scene;

use glam::Vec4;
use kiln_rhi::{CommandBuffer, Device, Format};

use crate::base::gpu::FrameArenas;
use crate::base::renderer::{self as render, PresentRenderer, RenderFrame, Renderer};
use crate::base::scene::{Camera, Scene, SceneStorage};

use pipeline::{Pipeline, Root};
use scene::Storage;

pub struct RasterRenderer {
    storage: Storage,
    pipeline: Pipeline,
    frame_arenas: FrameArenas,
    vp: [Vec4; 4],
    cam_pos: Vec4,
}

impl RasterRenderer {
    pub fn new(device: &Device, color_format: Format, source: &Scene) -> render::Result<Self> {
        let pipeline = Pipeline::new(device, color_format)?;
        let storage = source.prepare::<Storage>(device, &())?;
        let frame_arenas = match FrameArenas::new(device, 4096, "spectra-raster-arena") {
            Ok(value) => value,
            Err(error) => {
                storage.destroy(device);
                return Err(error);
            }
        };
        Ok(Self {
            storage,
            pipeline,
            frame_arenas,
            vp: [Vec4::ZERO; 4],
            cam_pos: Vec4::ZERO,
        })
    }

    pub fn update_scene(&mut self, device: &Device, source: &Scene) -> render::Result<()> {
        let new_storage = source.prepare::<Storage>(device, &())?;
        let old_storage = std::mem::replace(&mut self.storage, new_storage);
        old_storage.destroy(device);
        Ok(())
    }
}

impl Renderer for RasterRenderer {
    fn encode(
        &mut self,
        frame: &RenderFrame<'_>,
        _cmd: &mut CommandBuffer,
        camera: &Camera,
    ) -> render::Result<()> {
        self.vp = camera_vp_rows(camera, frame.extent);
        self.cam_pos = camera.position().extend(1.0);
        Ok(())
    }

    fn destroy(self: Box<Self>, device: &Device) {
        let Self {
            storage,
            frame_arenas,
            ..
        } = *self;
        frame_arenas.destroy(device);
        storage.destroy(device);
    }
}

impl PresentRenderer for RasterRenderer {
    fn depth_format(&self) -> Option<Format> {
        Some(Format::D32Float)
    }

    fn encode_present(&mut self, frame: &RenderFrame<'_>, cmd: &mut CommandBuffer) {
        self.frame_arenas.reset(frame.slot);
        let root = self.frame_arenas.upload(
            frame.slot,
            &Root {
                vp0: self.vp[0],
                vp1: self.vp[1],
                vp2: self.vp[2],
                vp3: self.vp[3],
                cam_pos: self.cam_pos,
                verts: self.storage.vertices.gpu(),
                materials: self.storage.materials.gpu(),
                texture_bindings: self.storage.texture_bindings.gpu(),
                tri_count: self.storage.triangle_count,
                _pad: 0,
            },
        );
        self.pipeline.record(cmd, root, self.storage.triangle_count);
    }
}

fn camera_vp_rows(camera: &Camera, extent: glam::UVec2) -> [Vec4; 4] {
    let view = camera.world.inverse();
    let [near, far] = camera.projection.clipping_range;
    let projection = glam::DMat4::perspective_rh(
        camera.projection.vertical_fov_rad as f64,
        extent.x as f64 / extent.y.max(1) as f64,
        near.max(1e-3) as f64,
        far as f64,
    );
    let vp = projection * view;
    [
        vp.x_axis.as_vec4(),
        vp.y_axis.as_vec4(),
        vp.z_axis.as_vec4(),
        vp.w_axis.as_vec4(),
    ]
}
