//! One module per inspector domain.
//!
//! Every panel is a `show` function over [`super::Inspector`]: it reads the frame's view, draws its
//! own controls, and reports what it wants changed. Panels never reach into each other's state, so
//! adding a domain means adding a file and a tab.

pub mod camera;
pub mod illuminant;
pub mod light;
pub mod material;
pub mod object;
pub mod render;
pub mod scene_tree;
