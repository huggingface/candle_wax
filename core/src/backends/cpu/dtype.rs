pub trait CpuDtype: Clone {}

impl CpuDtype for f32 {}
impl CpuDtype for f64 {}
impl CpuDtype for i32 {}
impl CpuDtype for i64 {}