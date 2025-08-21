pub trait Zero {
    fn zero() -> Self;
}

impl Zero for f32 {
    fn zero() -> Self {
        0.0
    }
}

impl Zero for f64 {
    fn zero() -> Self {
        0.0
    }
}

impl Zero for i32 {
    fn zero() -> Self {
        0
    }
}

impl Zero for i64 {
    fn zero() -> Self {
        0
    }
}

pub trait Two {
    fn two() -> Self;
}

impl Two for f32 {
    fn two() -> Self {
        2.0
    }
}

impl Two for f64 {
    fn two() -> Self {
        2.0
    }
}

impl Two for i32 {
    fn two() -> Self {
        2
    }
}

impl Two for i64 {
    fn two() -> Self {
        2
    }
}
