use core::{
    Layout,
    backends::{
        Backend, LazyBackend,
        broadcast::{Broadcast, BroadcastFunc},
        map::{Map, MapFunc},
        reduce::{Reduce, ReduceFunc},
    },
    storage::Storage,
    tensor::{LazyTensor, Tensor},
};
use egg::{
    CostFunction, EGraph, Extractor, Id, Rewrite, Runner, Subst, Var, define_language, rewrite,
};
use macros::BackendOps;
use regex::Regex;
use std::{collections::HashMap, sync::Arc};