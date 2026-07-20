//! Visual-odometry front-end and submap registration. Geometry runs anchor-out
//! (never bootstrap at a segment boundary); registration is deferred and
//! continuous (islands merge wherever overlap appears). nalgebra is quarantined
//! here when it arrives in M6.
