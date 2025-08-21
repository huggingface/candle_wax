use quote::{format_ident, quote};
use syn::{Attribute, DeriveInput, Expr, ExprArray, ExprLit, Lit, parse::Parse};

pub fn extract_inner_type(input: &DeriveInput) -> syn::Ident {
    // Extract the trait bound from T: SomeType
    if let Some(param) = input.generics.params.first() {
        if let syn::GenericParam::Type(type_param) = param {
            if let Some(bound) = type_param.bounds.first() {
                if let syn::TypeParamBound::Trait(trait_bound) = bound {
                    if let Some(segment) = trait_bound.path.segments.last() {
                        return segment.ident.clone();
                    }
                }
            }
        }
    }

    panic!("Could not extract inner type from struct definition");
}

pub fn parse_ops_attribute(attrs: &[Attribute]) -> Vec<String> {
    for attr in attrs {
        if attr.path().is_ident("storage_ops") {
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

pub fn generate_op_impl(
    storage_name: &syn::Ident,
    inner_type: &syn::Ident,
    op_type: &str,
) -> proc_macro2::TokenStream {
    let op_ident = format_ident!("{}", op_type);
    let op_func_ident = format_ident!("{}Func", op_type);

    let (_method_name, method_signature, method_call) = match op_type {
        "Map" => (
            format_ident!("map"),
            quote! { fn map(&self, layout: &Layout, f: F) -> Self::OutputStorage },
            quote! { f.call(layout, self) },
        ),
        "Reduce" => (
            format_ident!("reduce"),
            quote! { fn reduce(&self, layout: &Layout, dim: i32, f: F) -> Self::OutputStorage },
            quote! { f.call(layout, dim, self) },
        ),
        _ => panic!("Unknown operation type: {}", op_type),
    };

    quote! {
        impl<U, V, F> #op_ident<U, V, F> for #storage_name<U>
        where
            U: #inner_type,
            V: #inner_type,
            F: #op_func_ident<U, V, InputStorage<U> = #storage_name<U>, OutputStorage<V> = #storage_name<V>>,
        {
            type OutputStorage = #storage_name<V>;

            #method_signature {
                #method_call
            }
        }
    }
}
