use super::{Attention, FlashAttnCfg};

impl FlashAttnCfg {
    pub fn compute_cuda(&self, _reqs: &mut [Attention]) {
        todo!()
    }
}
