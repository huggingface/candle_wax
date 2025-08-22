use quote::{format_ident, quote};
use syn::{Attribute, Expr, ExprArray, ExprLit, Lit, parse::Parse};

pub fn parse_ops_attribute(attrs: &[Attribute]) -> Vec<String> {
    for attr in attrs {
        if attr.path().is_ident("backend_ops") {
            let mut ops_result = None;
            let _ = attr.parse_nested_meta(|meta| {
                if meta.path.is_ident("ops")
                    && let Ok(expr) = meta.value().and_then(|v| syn::Expr::parse(v))
                    && let syn::Expr::Array(array) = expr
                {
                    ops_result = Some(parse_string_array(&array));
                }
                Ok(())
            });
            if let Some(ops) = ops_result {
                return ops;
            }
        }
    }

    // Default: all known operations
    vec!["Map".to_string(), "Reduce".to_string()]
}

fn parse_string_array(array: &ExprArray) -> Vec<String> {
    array
        .elems
        .iter()
        .filter_map(|elem| {
            if let Expr::Lit(ExprLit {
                lit: Lit::Str(lit_str),
                ..
            }) = elem
            {
                Some(lit_str.value())
            } else {
                None
            }
        })
        .collect()
}

pub fn generate_op_impl(backend_name: &syn::Ident, op_type: &str) -> proc_macro2::TokenStream {
    let op_ident = format_ident!("{}", op_type);
    let op_func_ident = format_ident!("{}Func", op_type);

    let (method_signature, method_call) = match op_type {
        "Map" => (
            quote! {
                fn map(tensor: &Tensor<S, B>, f: F) -> Tensor<T, B>
            },
            quote! {
                let layout = tensor.layout.clone();
                let storage = f.call(&tensor.layout, &tensor.storage);
                Tensor::new(layout, storage)
            },
        ),
        "Reduce" => (
            quote! {
                fn reduce(tensor: &Tensor<S, B>, dim: i32, f: F) -> Tensor<T, B>
            },
            quote! {
                let layout = tensor.layout.clone();
                let storage = f.call(&tensor.layout, &tensor.storage, dim);
                Tensor::new(layout, storage)
            },
        ),
        _ => panic!("Unknown operation type: {}", op_type),
    };

    quote! {
        impl<B, S, T, U, V, F> #op_ident<B, S, T, U, V, F> for #backend_name
        where
            B: Backend,
            S: Storage<Inner = U>,
            T: Storage<Inner = V>,
            F: #op_func_ident<S, T, U, V>,
        {
            #method_signature {
                #method_call
            }
        }
    }
}
