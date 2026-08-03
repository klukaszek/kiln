//! Simple raster preview fallback backend.

mod pipeline;
mod scene;

use glam::Vec4;
use kiln_rhi::{CommandBuffer, Device, Format};

use crate::render::{self, FrameArenas, PresentRenderer, RenderFrame, Renderer};
use crate::scene::{Camera, Scene};

use pipeline::{Pipeline, Root};
use scene::PreviewScene;

pub struct PreviewRenderer {
    scene: PreviewScene,
    pipeline: Pipeline,
    frame_arenas: FrameArenas,
    vp: [Vec4; 4],
    cam_pos: Vec4,
}

impl PreviewRenderer {
    pub fn new(device: &Device, color_format: Format, scene: &Scene) -> render::Result<Self> {
        let pipeline = Pipeline::new(device, color_format)?;
        let scene = PreviewScene::build(device, scene)?;
        let frame_arenas = match FrameArenas::new(device, 4096, "spectra-raster-arena") {
            Ok(value) => value,
            Err(error) => {
                scene.destroy(device);
                return Err(error);
            }
        };
        Ok(Self {
            scene,
            pipeline,
            frame_arenas,
            vp: [Vec4::ZERO; 4],
            cam_pos: Vec4::ZERO,
        })
    }
}

impl Renderer for PreviewRenderer {
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
            scene,
            frame_arenas,
            ..
        } = *self;
        frame_arenas.destroy(device);
        scene.destroy(device);
    }
}

impl PresentRenderer for PreviewRenderer {
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
                verts: self.scene.vertices.gpu(),
                materials: self.scene.materials.gpu(),
                texture_bindings: self.scene.texture_bindings.gpu(),
                tri_count: self.scene.triangle_count,
                _pad: 0,
            },
        );
        self.pipeline.record(cmd, root, self.scene.triangle_count);
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
