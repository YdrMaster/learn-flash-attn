use crate::FlashAttnCfg;

type Tdata = f64;

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub(super) struct KernelCfg {
    pub g: usize,
    pub d: usize,
    pub bs: usize,
    pub scale: Tdata,
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub(super) struct KVPage {
    pub k: *const Tdata,
    pub v: *const Tdata,
}

unsafe impl Send for KVPage {}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub(super) struct Strides2D {
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
pub(super) struct KernelReq {
    pub q: *const Tdata,
    pub q_strides: Strides2D,
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

impl FlashAttnCfg {
    pub(super) fn to_kernel_cfg(&self) -> KernelCfg {
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
            scale: (d as f64).sqrt().recip(),
        }
    }

    pub(super) fn shared_elements(&self) -> usize {
        let &Self {
            d,
            tile_seq,
            tile_ctx,
            ..
        } = self;
        tile_seq * d + tile_ctx * d + tile_ctx * d + tile_seq * tile_ctx
    }
}
