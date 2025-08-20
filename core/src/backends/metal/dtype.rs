pub trait MetalDtype: Clone {} // Mine 

impl MetalDtype for f32 {} // Local Foreign - Ok
impl MetalDtype for f64 {}
impl MetalDtype for i32 {}
impl MetalDtype for i64 {}
