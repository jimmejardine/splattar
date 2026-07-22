//! Does photometric tracking actually recover a camera pose?
//!
//! This is the load-bearing question for the whole architecture: everything
//! after it assumes a frame can be located against the map by descending the
//! photometric residual. Ground truth rather than a reimplementation — a known
//! room and a known camera make "did it recover the right answer" a fact
//! (CLAUDE.md).

use glam::{Quat, Vec3};
use gs_kernels::RasterCamera;
use gs_map::{GpuMap, synthetic};
use gs_track::{TrackConfig, Tracker};

const W: u32 = 320;
const H: u32 = 320;

struct Rig {
    ctx: gs_wgpu::GpuContext,
    map: GpuMap,
    tracker: Tracker,
    focal: f32,
}

fn rig() -> Option<Rig> {
    let ctx = match gs_map::gpu() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("SKIPPING GPU tracking test: {e:#}");
            return None;
        }
    };
    let surfels = synthetic::room(24, 3.0);
    let map = GpuMap::new(&ctx, &surfels, surfels.len() as u32, W, H);
    let tracker = Tracker::new(&ctx, &map.raster, W, H);
    let focal = (H as f32 * 0.5) / (30.0_f32.to_radians()).tan();
    Some(Rig {
        ctx,
        map,
        tracker,
        focal,
    })
}

impl Rig {
    fn camera(&self, center: Vec3, yaw: f32) -> RasterCamera {
        RasterCamera {
            center,
            quat: Quat::from_rotation_y(yaw),
            focal: self.focal,
            sh_degree: 0,
        }
    }

    /// Render `truth`, then try to recover it starting from `start`.
    fn recover(
        &self,
        truth: &RasterCamera,
        start: &RasterCamera,
        cfg: &TrackConfig,
    ) -> (RasterCamera, gs_track::TrackReport) {
        let frame = self.map.render_f32(&self.ctx, truth);
        self.tracker.set_target(&self.ctx, &frame);
        let mut cam = start.clone();
        // Room half-extent 3.0, camera near the middle: ~3 units of scene depth.
        let report = self.tracker.track(
            &self.ctx,
            &self.map.raster,
            self.map.live(),
            &mut cam,
            3.0,
            cfg,
        );
        (cam, report)
    }
}

fn errors(a: &RasterCamera, b: &RasterCamera) -> (f32, f32) {
    let pos = (a.center - b.center).length();
    let rot = a.quat.angle_between(b.quat).to_degrees();
    (pos, rot)
}

/// The headline: a displaced camera must converge back onto the truth.
#[test]
fn tracking_recovers_a_displaced_camera() {
    let Some(rig) = rig() else { return };
    let truth = rig.camera(Vec3::new(0.4, 0.1, -0.3), 0.6);
    let start = RasterCamera {
        center: truth.center + Vec3::new(0.08, -0.05, 0.06),
        quat: truth.quat * Quat::from_rotation_x(1.5_f32.to_radians()),
        ..truth.clone()
    };

    let (before_pos, before_rot) = errors(&start, &truth);
    let cfg = TrackConfig::default();
    let (got, report) = rig.recover(&truth, &start, &cfg);
    let (after_pos, after_rot) = errors(&got, &truth);

    eprintln!(
        "pose error {before_pos:.4} m / {before_rot:.2}deg -> {after_pos:.4} m / {after_rot:.2}deg \
         in {} iters; residual {:.5} -> {:.5}, coverage {:.2}",
        report.iterations,
        report.residuals.first().copied().unwrap_or(f32::NAN),
        report.final_residual(),
        report.coverage,
    );

    assert!(
        report.improved(),
        "residual never fell: {:?}",
        report.residuals
    );
    assert!(
        after_pos < before_pos * 0.5,
        "position barely improved: {before_pos:.4} -> {after_pos:.4}"
    );
    assert!(
        after_rot < before_rot * 0.5,
        "rotation barely improved: {before_rot:.2} -> {after_rot:.2}"
    );
}

/// Starting AT the answer must not wander away from it. A tracker that drifts
/// off a correct pose would corrupt the map on every already-good frame.
#[test]
fn tracking_holds_a_correct_pose() {
    let Some(rig) = rig() else { return };
    let truth = rig.camera(Vec3::new(-0.2, 0.0, 0.5), -0.3);
    let (got, report) = rig.recover(&truth, &truth, &TrackConfig::default());
    let (pos, rot) = errors(&got, &truth);
    eprintln!(
        "held pose: drifted {pos:.5} m / {rot:.3}deg over {} iters",
        report.iterations
    );
    assert!(pos < 0.01, "drifted {pos:.5} m from a correct pose");
    assert!(rot < 0.2, "drifted {rot:.3} deg from a correct pose");
}

/// The residual must be a usable objective: lower for a better pose. If it did
/// not order poses correctly there would be nothing to descend, and the
/// recovery test above could pass by luck.
#[test]
fn the_residual_orders_poses_by_correctness() {
    let Some(rig) = rig() else { return };
    let truth = rig.camera(Vec3::new(0.1, 0.0, 0.2), 0.2);
    let frame = rig.map.render_f32(&rig.ctx, &truth);
    rig.tracker.set_target(&rig.ctx, &frame);

    // One iteration only: measure the residual where it starts, do not descend.
    let probe = |offset: f32| -> f32 {
        let mut cam = RasterCamera {
            center: truth.center + Vec3::new(offset, 0.0, 0.0),
            ..truth.clone()
        };
        rig.tracker
            .track(
                &rig.ctx,
                &rig.map.raster,
                rig.map.live(),
                &mut cam,
                3.0,
                &TrackConfig {
                    iterations: 1,
                    ..Default::default()
                },
            )
            .final_residual()
    };
    let (exact, near, far) = (probe(0.0), probe(0.05), probe(0.25));
    eprintln!("residual by offset: 0.0 -> {exact:.5}, 0.05 -> {near:.5}, 0.25 -> {far:.5}");
    assert!(exact < near, "exact pose not the minimum: {exact} vs {near}");
    assert!(near < far, "residual not monotone: {near} vs {far}");
}
