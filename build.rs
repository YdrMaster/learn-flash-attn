use build_script_cfg::Cfg;
use find_cuda_helper::find_cuda_root;

fn main() {
    let cuda = Cfg::new("cuda");
    if find_cuda_root().is_some() {
        cuda.define()
    }
}
