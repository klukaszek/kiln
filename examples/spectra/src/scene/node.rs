//! Stable scene-graph objects used by the editor and importers.

use std::collections::HashMap;

use super::{Instance, Light};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct NodeId(pub usize);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NodeObject {
    Group,
    Mesh { instance: usize },
    Light { light: usize },
}

#[derive(Clone, Debug)]
pub struct SceneNode {
    pub id: NodeId,
    pub name: String,
    pub path: String,
    pub parent: Option<NodeId>,
    pub children: Vec<NodeId>,
    pub object: NodeObject,
}

impl SceneNode {
    pub fn is_group(&self) -> bool {
        matches!(self.object, NodeObject::Group)
    }
}

/// Build a hierarchical view from USD paths while keeping mesh and light ownership in their
/// canonical scene arrays. Parent nodes are editor-only grouping nodes; object transforms remain
/// owned by mesh instances and analytic lights.
pub fn build_scene_nodes(instances: &[Instance], lights: &[Light]) -> Vec<SceneNode> {
    let mut nodes = vec![SceneNode {
        id: NodeId(0),
        name: "Scene".into(),
        path: "/".into(),
        parent: None,
        children: Vec::new(),
        object: NodeObject::Group,
    }];
    let mut by_path = HashMap::from([(String::from("/"), NodeId(0))]);

    for (instance, value) in instances.iter().enumerate() {
        insert_path(
            &mut nodes,
            &mut by_path,
            &value.path,
            NodeObject::Mesh { instance },
        );
    }
    for (light, value) in lights.iter().enumerate() {
        insert_path(
            &mut nodes,
            &mut by_path,
            &value.path,
            NodeObject::Light { light },
        );
    }
    nodes
}

fn insert_path(
    nodes: &mut Vec<SceneNode>,
    by_path: &mut HashMap<String, NodeId>,
    path: &str,
    object: NodeObject,
) {
    let mut parent = NodeId(0);
    let mut current_path = String::new();
    for part in path.split('/').filter(|part| !part.is_empty()) {
        current_path.push('/');
        current_path.push_str(part);
        let node = if let Some(&node) = by_path.get(&current_path) {
            node
        } else {
            let id = NodeId(nodes.len());
            nodes.push(SceneNode {
                id,
                name: part.to_owned(),
                path: current_path.clone(),
                parent: Some(parent),
                children: Vec::new(),
                object: NodeObject::Group,
            });
            nodes[parent.0].children.push(id);
            by_path.insert(current_path.clone(), id);
            id
        };
        parent = node;
    }
    if let Some(node) = nodes.get_mut(parent.0) {
        node.object = object;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_parent_nodes_from_paths() {
        let nodes = build_scene_nodes(
            &[Instance {
                mesh: super::super::MeshId(0),
                path: "/World/Room/Emitter".into(),
                transform: glam::DMat4::IDENTITY,
            }],
            &[],
        );
        assert_eq!(nodes[0].children.len(), 1);
        assert_eq!(nodes.last().unwrap().path, "/World/Room/Emitter");
        assert!(matches!(
            nodes.last().unwrap().object,
            NodeObject::Mesh { .. }
        ));
    }
}
