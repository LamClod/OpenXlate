use std::env;
use std::path::PathBuf;

fn main() {
    let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let out_path: PathBuf =
        PathBuf::from(&crate_dir).join("generated").join("kernel.h");

    if let Ok(bindings) = cbindgen::generate(&crate_dir) {
        let _ = std::fs::create_dir_all(out_path.parent().unwrap());
        bindings.write_to_file(&out_path);
    }

    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=cbindgen.toml");
}
