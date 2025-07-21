pub mod cpu;

#[cfg(cuda)]
pub mod cuda;

type Tdata = f64;

#[derive(Clone, Debug)]
pub struct FlashAttnCfg {
    pub h: usize,
    pub kvh: usize,
    pub d: usize,
    pub tile_seq: usize,
    pub tile_ctx: usize,
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct KernelCfg {
    pub g: usize,
    pub d: usize,
    pub bs: usize,
    pub scale: f32,
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct KVPage {
    pub k: *mut Tdata,
    pub v: *mut Tdata,
}

unsafe impl Sync for KVPage {}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct Strides2D {
    pub head: isize,
    pub seq: isize,
}

impl Strides2D {
    pub const fn offset(&self, head: usize, seq: usize) -> isize {
        head as isize * self.head + seq as isize * self.seq
    }
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct KernelReq {
    pub q: *const Tdata,
    pub q_strides: Strides2D,
    pub k: *const Tdata,
    pub k_strides: Strides2D,
    pub v: *const Tdata,
    pub v_strides: Strides2D,
    pub pages_start: usize,
    pub kv_strides: Strides2D,
    pub o: *mut Tdata,
    pub o_strides: Strides2D,
    pub mask: *const bool,
    pub l: *mut Tdata,
    pub m: *mut Tdata,
    pub n: usize,
    pub s: usize,
}

unsafe impl Sync for KernelReq {}

impl FlashAttnCfg {
    pub fn to_kernel_cfg(&self) -> KernelCfg {
        let &Self {
            h,
            kvh,
            d,
            tile_ctx,
            ..
        } = self;
        assert_eq!(h % kvh, 0);
        KernelCfg {
            g: h / kvh,
            d,
            bs: tile_ctx,
            scale: (d as f32).sqrt().recip(),
        }
    }

    pub fn shared_elements(&self) -> usize {
        let &Self {
            d,
            tile_seq,
            tile_ctx,
            ..
        } = self;
        tile_seq * d + tile_ctx * d + tile_ctx * d + tile_seq * tile_ctx
    }
}

#[cfg(test)]
mod test;
