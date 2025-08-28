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
    vec![
        "Map".to_string(),
        "Reduce".to_string(),
        "Broadcast".to_string(),
    ]
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

    match op_type {
        "Map" => quote! {
            impl<B, S, T, U, V, F> #op_ident<B, S, T, U, V, F> for #backend_name
            where
                B: Backend,
                S: Storage<Inner = U>,
                T: Storage<Inner = V>,
                F: #op_func_ident<S, T, U, V>,
            {
                fn map(layout: &Layout, storage: &S, f: F) -> T {
                    f.call(&layout, &storage)
                }
            }
        },
        "Reduce" => quote! {
            impl<B, S, T, U, V, F> #op_ident<B, S, T, U, V, F> for #backend_name
            where
                B: Backend,
                S: Storage<Inner = U>,
                T: Storage<Inner = V>,
                F: #op_func_ident<S, T, U, V>,
            {
                fn reduce(layout: &Layout, storage: &S, dim: i32, f: F) -> T {
                    f.call(&layout, &storage, dim)
                }
            }
        },
        "Broadcast" => quote! {
            impl<B, R, S, T, U, V, W, F> #op_ident<B, R, S, T, U, V, W, F> for #backend_name
            where
                B: Backend,
                R: Storage<Inner = U>,
                S: Storage<Inner = V>,
                T: Storage<Inner = W>,
                F: #op_func_ident<R, S, T, U, V, W>,
            {
                fn broadcast(lhs_layout: &Layout, lhs_storage: &R, rhs_layout: &Layout, rhs_storage: &S, corresponding_dims: &[(i32, i32)], f: F) -> T{
                    f.call(lhs_layout, lhs_storage, rhs_layout, rhs_storage, corresponding_dims)
                }
            }
        },
        _ => panic!("Unknown operation type: {}", op_type),
    }
}
