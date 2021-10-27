use std::mem;

pub fn inv_sqrtx64(x: f64) -> f64 {
    // Magic number based on Chris Lomont work:
    // const MAGIC_U32: u32 = 0x5f375a86;
    // The Original Magic Number:
    // const MAGIC_32: u32 = 0x5f3759df;
    let xhalf = 0.5f64 * x;
    let mut i: i64 = unsafe { mem::transmute(x) };
    i = 0x5f3759df - (i >> 1);
    let mut res: f64 = unsafe { mem::transmute(i) };
    res = res * (1.5f64 - xhalf * res * res);
    res = res * (1.5f64 - xhalf * res * res);
    res
}

pub fn inv_sqrtx86(x: f32) -> f32 {
    let xhalf = 0.5f32 * x;
    let mut i: i32 = unsafe { mem::transmute(x) };
    i = 0x5f3759df - (i >> 1);
    let mut res: f32 = unsafe { mem::transmute(i) };
    res = res * (1.5f32 - xhalf * res * res);
    res = res * (1.5f32 - xhalf * res * res);
    res
}
