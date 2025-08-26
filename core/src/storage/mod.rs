pub mod cpu;

pub trait Storage: Clone {
    type Inner;
}
