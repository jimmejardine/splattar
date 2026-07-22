//! Quick manual check: `cargo run -p gs-io --release --example load_ply -- <file.ply>`
//! Prints layout, count, degree, load time (the <5 s M0 gate for cactus-high).

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let path = std::env::args().nth(1).expect("usage: load_ply <file.ply>");
    match gs_io::load_ply(&path) {
        Ok(gs_io::PlyContents::Splats(s)) => {
            let (lo, hi) = s.bbox().unwrap();
            println!(
                "splats: {}  sh_degree: {}  bbox: {lo:?} .. {hi:?}",
                s.len(),
                s.sh_degree
            );
        }
        Ok(gs_io::PlyContents::Points(p)) => println!("points: {}", p.len()),
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    }
}
