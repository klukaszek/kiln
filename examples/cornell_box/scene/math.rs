pub fn perspective_rh_zo(fovy_rad: f32, aspect: f32, near: f32, far: f32) -> [f64; 16] {
    let f = 1.0 / (fovy_rad as f64 / 2.0).tan();
    let (near, far) = (near as f64, far as f64);
    let mut m = [0.0f64; 16];
    m[0] = f / aspect as f64;
    m[5] = f;
    m[10] = far / (near - far);
    m[11] = -1.0;
    m[14] = far * near / (near - far);
    m
}

pub fn face_normal(p: &[[f32; 3]]) -> [f32; 3] {
    if p.len() < 3 {
        return [0.0, 1.0, 0.0];
    }
    let a = [p[1][0] - p[0][0], p[1][1] - p[0][1], p[1][2] - p[0][2]];
    let b = [p[2][0] - p[0][0], p[2][1] - p[0][1], p[2][2] - p[0][2]];
    norm3([
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ])
}

pub fn norm3(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len > 1e-8 {
        [v[0] / len, v[1] / len, v[2] / len]
    } else {
        [0.0, 1.0, 0.0]
    }
}
