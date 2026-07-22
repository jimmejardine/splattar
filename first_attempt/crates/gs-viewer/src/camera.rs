//! First-person fly camera: WASD + QE/Space-Ctrl vertical, mouse look,
//! scroll-to-adjust speed. Works in upright (y-up) world space; the scene
//! transform is applied when converting to a render camera.

use glam::{Quat, Vec3};
use gs_core::Camera;

use crate::input::InputState;

pub struct FlyCamera {
    pub position: Vec3,
    /// Radians; 0 looks down −z.
    pub yaw: f32,
    /// Radians, clamped to ±89°.
    pub pitch: f32,
    /// Auto base speed, set per frame by the caller from the distance to the
    /// scene content — far away ("zoomed out") moves much faster than up
    /// close. Units/second.
    pub speed: f32,
    /// User multiplier on top of the auto speed (mouse scroll).
    pub speed_mult: f32,
    pub fov_y: f32,
    pub mouse_sensitivity: f32,
}

impl Default for FlyCamera {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            yaw: 0.0,
            pitch: 0.0,
            speed: 2.0,
            speed_mult: 1.0,
            fov_y: 60f32.to_radians(),
            mouse_sensitivity: 0.0025,
        }
    }
}

impl FlyCamera {
    /// Place the camera to frame a bounding box (used at scene load).
    pub fn framing(bbox: (Vec3, Vec3)) -> Self {
        let center = (bbox.0 + bbox.1) * 0.5;
        let radius = 0.5 * (bbox.1 - bbox.0).length();
        Self {
            position: center + Vec3::new(0.0, 0.0, 2.2 * radius.max(0.1)),
            speed: radius.max(0.1),
            ..Default::default()
        }
    }

    pub fn rotation(&self) -> Quat {
        Quat::from_rotation_y(self.yaw) * Quat::from_rotation_x(self.pitch)
    }

    /// Snap to a recorded camera pose given in SCENE space (renderer
    /// convention: looks −z). The fly camera lives in upright space, so the
    /// pose is mapped through `scene_rot` first; roll is dropped (yaw/pitch
    /// camera), which is what you want when replaying handheld footage.
    pub fn snap_to(&mut self, position: Vec3, rotation: Quat, scene_rot: Quat) {
        self.position = scene_rot * position;
        let f = (scene_rot * rotation) * Vec3::NEG_Z;
        self.yaw = (-f.x).atan2(-f.z);
        self.pitch = f.y.clamp(-1.0, 1.0).asin();
    }

    pub fn update(&mut self, dt: f32, input: &InputState) {
        self.yaw -= input.mouse_dx * self.mouse_sensitivity;
        self.pitch = (self.pitch - input.mouse_dy * self.mouse_sensitivity)
            .clamp(-89f32.to_radians(), 89f32.to_radians());
        if input.scroll != 0.0 {
            self.speed_mult =
                (self.speed_mult * 1.15f32.powf(input.scroll)).clamp(0.05, 20.0);
        }

        let rot = self.rotation();
        let forward = rot * Vec3::NEG_Z;
        let right = rot * Vec3::X;
        let mut wish = Vec3::ZERO;
        if input.forward {
            wish += forward;
        }
        if input.back {
            wish -= forward;
        }
        if input.right {
            wish += right;
        }
        if input.left {
            wish -= right;
        }
        if input.up {
            wish += Vec3::Y;
        }
        if input.down {
            wish -= Vec3::Y;
        }
        if wish != Vec3::ZERO {
            let sprint = if input.sprint { 4.0 } else { 1.0 };
            self.position += wish.normalize() * self.speed * self.speed_mult * sprint * dt;
        }
    }

    /// Convert to a render camera, mapping from upright world space into scene
    /// space via `scene_rot` (e.g. the 180° flip for COLMAP-convention data).
    pub fn to_camera(&self, scene_rot: Quat) -> Camera {
        let inv = scene_rot.inverse();
        Camera {
            position: inv * self.position,
            rotation: inv * self.rotation(),
            fov_y: self.fov_y,
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn moves_forward_along_view() {
        let mut cam = FlyCamera::default();
        let input = InputState {
            forward: true,
            ..Default::default()
        };
        cam.update(1.0, &input);
        assert!(cam.position.z < -1.0, "moved down -z: {:?}", cam.position);
    }

    #[test]
    fn snap_to_reproduces_pose() {
        // A scene-space pose (renderer convention) must round-trip through
        // snap_to → to_camera up to the dropped roll.
        let mut cam = FlyCamera::default();
        let scene_rot = Quat::from_rotation_z(std::f32::consts::PI); // default flip
        let pos = Vec3::new(1.0, 2.0, -3.0);
        let rot = Quat::from_euler(glam::EulerRot::YXZ, 0.7, -0.3, 0.0);
        cam.snap_to(pos, rot, scene_rot);
        let out = cam.to_camera(scene_rot);
        assert!((out.position - pos).length() < 1e-5, "{:?}", out.position);
        let fwd_in = rot * Vec3::NEG_Z;
        let fwd_out = out.rotation * Vec3::NEG_Z;
        assert!(
            fwd_in.dot(fwd_out) > 0.9999,
            "forward mismatch: {fwd_in:?} vs {fwd_out:?}"
        );
    }

    #[test]
    fn pitch_clamps() {
        let mut cam = FlyCamera::default();
        let input = InputState {
            mouse_dy: -1e6,
            ..Default::default()
        };
        cam.update(0.016, &input);
        assert!(cam.pitch <= 89f32.to_radians() + 1e-6);
    }
}
