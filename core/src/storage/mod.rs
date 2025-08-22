pub mod cpu;
pub mod metal;

pub trait Storage {
    type Inner;
}