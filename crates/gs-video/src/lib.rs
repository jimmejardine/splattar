//! Video ingest: demux, decode, color conversion, sharpness scoring, keyframe
//! promotion. Video is the only product input; per-frame PTS is carried
//! everywhere (iPhone footage is VFR — never frame_index / fps).
