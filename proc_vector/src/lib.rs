extern crate proc_macro;
use proc_macro::TokenStream;

#[proc_macro]
pub fn fn_simple_as(ty: TokenStream) -> TokenStream {
    let typename = ty.to_string();
    format!("pub fn as_{type}s(&self) -> Vector<{type}> {{
			Vector {{ 
				x: self.x as {type},
				y: self.y as {type},
                z: self.z as {type},
			}}
		}}",
		type = typename)
    .parse()
    .unwrap()
}

#[proc_macro]
pub fn fn_lower_bounded_as(toks: TokenStream) -> TokenStream {
    const USAGE_MSG: &'static str =
        "Invalid usage, expected: fn_lower_bounded_as!(source_type, destination_type, lower_bound)";

    let mut iter = toks.into_iter();

    let src_ty = iter.next().expect(USAGE_MSG);
    iter.next().expect(USAGE_MSG); // Comma

    let dst_ty = iter.next().expect(USAGE_MSG);
    iter.next().expect(USAGE_MSG); // Comma

    let lower_bound = iter.next().expect(USAGE_MSG);
    if let Some(_) = iter.next() {
        std::panic::panic_any(USAGE_MSG);
    }

    format!(
        "pub fn as_{dst_type}s(&self) -> Vector<{dst_type}> {{
			Vector {{
				x: {src_type}::max({lower_bound}, self.x) as {dst_type},
				y: {src_type}::max({lower_bound}, self.y) as {dst_type},
                z: {src_type}::max({lower_bound}, self.z) as {dst_type},
			}}
		}}",
        src_type = src_ty.to_string(),
        dst_type = dst_ty.to_string(),
        lower_bound = lower_bound.to_string()
    )
    .parse()
    .unwrap()
}
