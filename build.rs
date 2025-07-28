use build_script_cfg::Cfg;
use find_cuda_helper::find_cuda_root;
use search_maca_tools::find_maca_root;

fn main() {
    let cuda = Cfg::new("cuda");
    if find_maca_root().is_some() || find_cuda_root().is_some() {
        cuda.define()
    }
}
