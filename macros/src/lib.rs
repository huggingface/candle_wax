use proc_macro::TokenStream;
use quote::quote;
use syn::{DeriveInput, parse_macro_input};

mod storage;
use storage::{extract_inner_type, generate_op_impl, parse_ops_attribute as storage_parse_ops_attribute};

mod tensor;
use tensor::{generate_tensor_op_impl, parse_ops_attribute as tensor_parse_ops_attribute};

#[proc_macro_derive(StorageOps, attributes(storage_ops))]
pub fn derive_storage_ops(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    let storage_name = &input.ident; // e.g., CpuStorage
    let inner_type = extract_inner_type(&input); // e.g., CpuDtype
    let ops = storage_parse_ops_attribute(&input.attrs);

    let implementations: Vec<_> = ops
        .iter()
        .map(|op_type| generate_op_impl(storage_name, &inner_type, op_type))
        .collect();

    let expanded = quote! {
        #(#implementations)*
    };

    TokenStream::from(expanded)
}


#[proc_macro_derive(TensorOps, attributes(tensor_ops))]
pub fn derive_tensor_ops(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    let ops = tensor_parse_ops_attribute(&input.attrs);

    let implementations: Vec<_> = ops
        .iter()
        .map(|op_type| generate_tensor_op_impl(op_type))
        .collect();

    let expanded = quote! {
        #(#implementations)*
    };

    TokenStream::from(expanded)
}