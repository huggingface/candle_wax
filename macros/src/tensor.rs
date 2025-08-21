use quote::{format_ident, quote};
use syn::{Attribute, Expr, ExprArray, ExprLit, Lit, parse::Parse};

pub fn parse_ops_attribute(attrs: &[Attribute]) -> Vec<String> {
    for attr in attrs {
        if attr.path().is_ident("tensor_ops") {
            let mut ops_result = None;
            let _ = attr.parse_nested_meta(|meta| {
                if meta.path.is_ident("ops") {
                    if let Ok(expr) = meta.value().and_then(|v| syn::Expr::parse(v)) {
                        if let syn::Expr::Array(array) = expr {
                            ops_result = Some(parse_string_array(&array));
                        }
                    }
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

pub fn generate_tensor_op_impl(op_type: &str) -> proc_macro2::TokenStream {
    let op_ident = format_ident!("{}", op_type);
    let op_func_ident = format_ident!("{}Func", op_type);

    let (method_signature, tensor_constructor) = match op_type {
        "Map" => (
            quote! { fn map(&self, f: F) -> Self::OutStorage },
            quote! {
                Tensor {
                    layout: self.layout.clone(),
                    storage: f.call(&self.layout, &self.storage),
                }
            },
        ),
        "Reduce" => (
            quote! { fn reduce(&self, dim: i32, f: F) -> Self::OutStorage },
            quote! {
                Tensor {
                    layout: self.layout.clone(),
                    storage: f.call(&self.layout, dim, &self.storage) ,
                }
            },
        ),
        _ => panic!("Unknown operation type: {}", op_type),
    };

    quote! {
        impl<S, T, U, V, F> #op_ident<U, V, F> for Tensor<S>
        where
            S: Storage<Inner = U>,
            T: Storage<Inner = V>,
            F: #op_func_ident<U, V, InputStorage<U> = S, OutputStorage<V> = T>,
        {
            type OutStorage = Tensor<T>;

            #method_signature {
                #tensor_constructor
            }
        }
    }
}
