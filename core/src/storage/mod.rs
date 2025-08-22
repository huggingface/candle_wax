pub mod cpu;
pub(crate) mod metal;

pub trait Storage {
    type Inner;
}
