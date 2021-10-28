# Vector, 3-dimentionnal vector structure for Rust

1. [vector](#vector)
2. [Using math_vector](#using-math_vector)

## vector

A simple and convenient 3D vector type without excessive use of external dependencies. If other vector crates are swiss-army knives, math_vector is a spoon; safe, intuitive, and convenient. As an added bonus, you won't run into any excursions with the law using this library thanks to the awfully permissive Unlicense.

## Using math_vector

You probably don't need any documentation to get by with the `Vector` type; functions like `dot`, `length`, and `angle` are hopefully all named intuitively enough for you feel them out. If you do find yourself wondering about certain bits of functionality, then be sure to take a look at the [in-code documentation](src/lib.rs), where you can find examples and explanations of everything on offer.

To add math_vector as a dependency in any of your rust project, just add the following in your cargo.toml dependencies' section :

```toml
math_vector = { git = "https://github.com/ThomasByr/rust-math_vector" }
```
