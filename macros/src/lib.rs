use proc_macro::TokenStream;
use quote::quote;
use syn::{DeriveInput, parse_macro_input};

mod backend;
use backend::{generate_op_impl, parse_ops_attribute};

#[proc_macro_derive(BackendOps, attributes(backend_ops))]
pub fn derive_backend_ops(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    let backend_name = &input.ident; // e.g., CpuBackend
    let ops = parse_ops_attribute(&input.attrs);

    let implementations: Vec<_> = ops
        .iter()
        .map(|op_type| generate_op_impl(backend_name, op_type))
        .collect();

    let expanded = quote! {
        #(#implementations)*
    };

    TokenStream::from(expanded)
}
