//! Pinhole camera model. Right-handed, y-up, camera looks down its local −z
//! (standard GL-style view space); wgpu NDC (z ∈ [0,1]) via glam's *_rh
//! projection helpers.

use glam::{Mat4, Quat, Vec2, Vec3};

#[derive(Debug, Clone, Copy)]
pub struct Camera {
    pub position: Vec3,
    pub rotation: Quat,
    /// Vertical field of view in radians.
    pub fov_y: f32,
    pub near: f32,
    pub far: f32,
}

impl Camera {
    pub fn view_matrix(&self) -> Mat4 {
        Mat4::from_rotation_translation(self.rotation, self.position).inverse()
    }

    /// Projection with wgpu depth range (z ∈ [0,1] — glam's "directx" convention).
    pub fn proj_matrix(&self, aspect: f32) -> Mat4 {
        glam::camera::rh::proj::directx::perspective(self.fov_y, aspect, self.near, self.far)
    }

    /// Pixel focal lengths (fx, fy) for a viewport, as used by EWA projection.
    pub fn focal_px(&self, viewport: Vec2) -> Vec2 {
        let fy = 0.5 * viewport.y / (0.5 * self.fov_y).tan();
        // Square pixels: fx = fy. Horizontal FOV follows from the aspect ratio.
        Vec2::new(fy, fy)
    }

    /// Unit forward direction (−z of the camera frame).
    pub fn forward(&self) -> Vec3 {
        self.rotation * Vec3::NEG_Z
    }

    /// Camera at `eye` looking toward `target` (y-up world).
    pub fn look_at(eye: Vec3, target: Vec3, up: Vec3) -> Self {
        let fwd = (target - eye).normalize();
        let right = fwd.cross(up).normalize();
        let up2 = right.cross(fwd);
        Self {
            position: eye,
            rotation: glam::Quat::from_mat3(&glam::Mat3::from_cols(right, up2, -fwd)),
            ..Default::default()
        }
    }
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            fov_y: 60f32.to_radians(),
            near: 0.05,
            far: 1000.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn view_matrix_moves_world_opposite() {
        let cam = Camera {
            position: Vec3::new(0.0, 0.0, 5.0),
            ..Default::default()
        };
        // A point at the origin should be 5 units in front of the camera (−z).
        let p = cam.view_matrix().transform_point3(Vec3::ZERO);
        assert!((p - Vec3::new(0.0, 0.0, -5.0)).length() < 1e-6);
    }

    #[test]
    fn forward_is_neg_z() {
        let cam = Camera::default();
        assert!((cam.forward() - Vec3::NEG_Z).length() < 1e-6);
    }

    #[test]
    fn look_at_faces_target() {
        let cam = Camera::look_at(Vec3::new(5.0, 3.0, 5.0), Vec3::ZERO, Vec3::Y);
        let expect = (Vec3::ZERO - cam.position).normalize();
        assert!((cam.forward() - expect).length() < 1e-5);
    }

    #[test]
    fn focal_matches_fov() {
        let cam = Camera {
            fov_y: 90f32.to_radians(),
            ..Default::default()
        };
        let f = cam.focal_px(Vec2::new(1600.0, 900.0));
        // tan(45°) = 1 → fy = h/2.
        assert!((f.y - 450.0).abs() < 1e-3);
    }
}
