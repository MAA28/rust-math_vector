//! # vector
//! A simple and convenient 3D vector library without excessive use of external
//! dependencies. If other vector crates are swiss-army knives, math_vector is a
//! spoon; safe, intuitive, and convenient. As an added bonus, you won't run
//! into any excursions with the law using this library thanks to the awfully
//! permissive Unlicense.
//!
//! The only type in this crate is [`Vector`], which is highly generic;
//! shifting functionality depending upon the traits implemented by its internal
//! components' types.
//!
//! [`Vector`]: struct.Vector.html
//!
//! # Example
//! ```
//! use math_vector::Vector;
//!
//! fn main() {
//!     // Vectors have fields X and Y, these can be of any type
//!     let v1: Vector<i32> = Vector { x: 10, y: 5 , z: 3 };
//!
//!     // Alternatively you can use new(..) to condense instantiation
//!     let v2: Vector<f64> = Vector::new(13.0, 11.5, 9.0);
//!
//!     // There are two ways to cast between Vectors, depending on the source
//!     // and target types.
//!     //
//!     // If the target type has a implementation of From<SourceType>, then you
//!     // can either use source.into_vec() or Vector::from_vec(source).
//!     assert_eq!(Vector::new(10.0, 5.0, 3.0), v1.into_vec());
//!     assert_eq!(Vector::new(10.0, 5.0, 3.0), Vector::from_vec(v1));
//!
//!     // If there is no From or Into implementation, then you're out of luck
//!     // unless you are using specific primitives, such as i32 and f64. In
//!     // this case you can use specialised functions, as shown below:
//!     assert_eq!(Vector::new(13, 11, 9), v2.as_i32s());
//!
//!     // The full list of interoperable primitives is as follows:
//!     //   - i32, i64, isize
//!     //   - u32, u64, usize
//!     //   - f32, f64
//!
//!     // As primitives generally implement From/Into for lossless casts,
//!     // an as_Ts() function is not available for those types, and
//!     // from(..)/into() should be favoured.
//!     //
//!     // Casts between signed and unsigned primitives will perform bounds
//!     // checking, so casting the vector (-10.0, 2.0, 11) to a Vector<u32> will
//!     // result in the vector (0, 2, 11).
//!
//!     // For types with an Add and Mul implementation, the functions dot() and
//!     // length_squared() are available. For access to length(), normalise(),
//!     // or angle() however, you must be using either Vector<f32> or
//!     // Vector<f64>.
//!     let _v1_len_sq = v1.length_squared();
//!     let v2_len = v2.length();
//!     let v2_dir = v2.normalise();
//!     println!("{} {} {}", v2_dir.x, v2_dir.y, v2_dir.z);
//!
//!     // Assuming the operator traits are implemented for the types involved,
//!     // you can add and subtract Vectors from one-another, as well as
//!     // multiply and divide them with scalar values.
//!     assert_eq!(v2.is_close(v2_dir * v2_len), true);
//!     assert_eq!(Vector::new(23.0, 16.5, 12.0),  v2 + v1.into_vec()) ;
//!
//!     // If you feel the need to multiply or divide individual components of
//!     // vectors with the same type, you can use mul_components(...) or
//!     // div_components(...) provided that their types can be multiplied or
//!     // divided.
//!
//!     // For any Vector<T>, there is an implementation of
//!     // From<(T, T)> and From<[T; 2]>
//!     let v4: Vector<f64> = Vector::new(1.5, 2.3, 3.1);
//!     assert_eq!(v4, (1.5, 2.3, 3.1).into());
//!     assert_eq!(v4, [1.5, 2.3, 3.1].into());
//!
//!     // Additionally, there is an Into<(T, T)> implementation for any types
//!     // that the vector components have their own Into implementations for
//!     assert_eq!((1.5, 2.3, 3.1), v4.into());
//!
//!     // If you want the normal of a vector you can just call normal()
//!     let v5 = Vector::new(-10.0, -2.3, 0.0);
//!     assert_eq!(Vector::new(2.3, -10.0, 0.0), v5.normal());
//!
//!     // You can get a vector consisting of only the x, y or z axis
//!     // component of a vector by calling abscissa() or ordinate() or applicate()
//!     // respectively
//!     let v6 = Vector::new(12.3, 83.2, -42.6);
//!     assert_eq!(Vector::new(12.3, 0.0, 0.0), v6.abscissa());
//!     assert_eq!(Vector::new(0.0, 83.2, 0.0), v6.ordinate());
//!     assert_eq!(Vector::new(0.0, 0.0, -42.6), v6.applicate());
//! }
//! ```

mod protocol;
#[cfg(test)]
mod test;

use proc_vector::{fn_lower_bounded_as, fn_simple_as};
use protocol::{InvSqrt32, InvSqrt64, EPSILON};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// A 2D vector, containing an `x` and a `y` component. While many types can be
/// used for a `Vector`'s components, the traits they implement determine
/// what functions are available.
///
/// Provided that the components implement the necessary traits, `Vector`s
/// can be added to or subtracted from one-another, and they can be mulitplied
/// and divided by scalar values.
///
/// There are generally two options for converting between `Vector` types. If
/// the internal components' type has an implementation of `Into` that targets
/// the desired type, then [`into_vec()`] can be called from the source object,
/// or [`from_vec(..)`] can be called and the source object can be provided.
///
/// If no `Into` implementation exists, then the only option is to use one of the
/// flavours of casting with `as`. These are in the form `as_types()`, and are only
/// implemented for specific types of components. An example usage would look like
/// this:
/// ```
/// use math_vector::Vector;
/// let f64_vector: Vector<f64> = Vector::new(10.3, 11.1, 0.0);
/// let i32_vector: Vector<i32> = f64_vector.as_i32s();
/// assert_eq!(Vector::new(10, 11, 0), i32_vector);
/// ```
///
/// Implementations of `as_types()` are only available when an implementation of
/// [`into_vec()`] is unavailable. This is to seperate between the lossless casting
/// of primitives with `into()` and `from(..)`, and the lossy casting between
/// primitives of varying detail.
///
/// Casts from signed types to unsigned types have a small additional check that
/// ensures a lower bound of 0 on the signed value, to reduce the chances of
/// experiencing undefined behaviour. This means that a `Vector<f64>` with a
/// value of `(-10.3, 11.1, 0.0)` would become `(0, 11, 0)` when cast to a `Vector<u32>`
/// with [`as_u32s()`].
///
/// The current list of interoperable types that can be cast with the `as` family of
/// functions is as follows:
///   - `i32`
///   - `i64`,
///   - `isize`
///   - `u32`
///   - `u64`
///   - `usize`
///   - `f32`
///   - `f64`
///
/// [`into_vec()`]: struct.Vector.html#method.into_vec
/// [`from_vec(..)`]: struct.Vector.html#method.from_vec
/// [`as_u32s()`]: struct.Vector.html#method.as_u32s-1
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Vector<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}

impl<T: Copy + Clone> Vector<T> {
    /// Create a new `Vector` with the provided components.
    pub fn new(x: T, y: T, z: T) -> Self {
        Self { x, y, z }
    }

    /// Construct a `Vector` with all components set to the provided value.
    pub fn all(value: T) -> Self {
        Self {
            x: value,
            y: value,
            z: value,
        }
    }

    /// Convert a `Vector2` of type `U` to one of type `T`. Available only when
    /// type T has implemented `From<U>`.
    ///
    /// # Example
    /// ```
    /// use math_vector::Vector;
    /// let i32_vector: Vector<i32> = Vector::new(25, 8, 0);
    /// let f64_vector: Vector<f64> = Vector::from_vec(i32_vector);
    /// assert_eq!(Vector::new(25.0, 8.0, 0.0), f64_vector);
    /// ```
    pub fn from_vec<U: Into<T> + Copy + Clone>(src: Vector<U>) -> Vector<T> {
        Vector {
            x: src.x.into(),
            y: src.y.into(),
            z: src.z.into(),
        }
    }

    /// Convert a `Vector2` of type `T` to one of type `U`. Available only when
    /// type T has implemented `Into<U>`.
    ///
    /// # Example
    /// ```
    /// use math_vector::Vector;
    /// let i32_vector: Vector<i32> = Vector::new(25, 8, 0);
    /// let i32_vector: Vector<i32> = Vector::new(25, 8, 0);
    /// let f64_vector: Vector<f64> = i32_vector.into_vec();
    /// assert_eq!(Vector::new(25.0, 8.0, 0.0), f64_vector);
    /// ```
    pub fn into_vec<U: From<T>>(self) -> Vector<U> {
        Vector {
            x: self.x.into(),
            y: self.y.into(),
            z: self.z.into(),
        }
    }
}

impl<T: Default> Vector<T> {
    /// Default construct a `Vector` with all components set to 0.
    pub fn default() -> Self {
        Self {
            x: T::default(),
            y: T::default(),
            z: T::default(),
        }
    }

    /// Returns a vector with only the horizontal component of the current one
    ///
    /// # Example
    /// ```
    /// use math_vector::Vector;
    /// let v = Vector::new(10, 20, 30);
    /// assert_eq!(Vector::new(10, 0, 0), v.abscissa());
    /// ```
    pub fn abscissa(self) -> Self {
        Self {
            x: self.x,
            y: Default::default(),
            z: Default::default(),
        }
    }

    /// Returns a vector with only the vertical component of the current one
    ///
    /// # Example
    /// ```
    /// use math_vector::Vector;
    /// let v = Vector::new(10, 20, 30);
    /// assert_eq!(Vector::new(0, 20, 0), v.ordinate());
    pub fn ordinate(self) -> Self {
        Self {
            x: Default::default(),
            y: self.y,
            z: Default::default(),
        }
    }

    /// Returns a vector with only the depth component of the current one
    ///
    /// # Example
    /// ```
    /// use math_vector::Vector;
    /// let v = Vector::new(10, 20, 30);
    /// assert_eq!(Vector::new(0, 0, 30), v.applicate());
    pub fn applicate(self) -> Self {
        Self {
            x: Default::default(),
            y: Default::default(),
            z: self.z,
        }
    }
}

impl<T> Vector<T>
where
    T: Mul<T, Output = T> + Copy + Clone,
{
    /// Returns a new vector with components equal to each of the current vector's
    /// components multiplied by the corresponding component of the provided vector
    ///
    /// # Example
    /// ```
    /// use math_vector::Vector;
    /// let v1 = Vector::new(11.0, -2.5, 3.0);
    /// let v2 = Vector::new(0.5, -2.0, 1.0);
    /// assert_eq!(Vector::new(5.5, 5.0, 3.0), v1.mul_components(v2));
    /// ```
    pub fn mul_components(self, other: Self) -> Self {
        Self {
            x: self.x * other.x,
            y: self.y * other.y,
            z: self.z * other.z,
        }
    }
}

impl<T> Vector<T>
where
    T: Div<T, Output = T> + Copy + Clone,
{
    /// Returns a new vector with components equal to each of the current vector's
    /// components divided by the corresponding component of the provided vector
    ///
    /// # Example
    /// ```
    /// use math_vector::Vector;
    /// let v1 = Vector::new(11.0, -2.5, 3.0);
    /// let v2 = Vector::new(0.5, -2.0, 1.0);
    /// assert_eq!(Vector::new(22.0, 1.25, 3.0), v1.div_components(v2));
    /// ```
    pub fn div_components(self, other: Self) -> Self {
        Self {
            x: self.x / other.x,
            y: self.y / other.y,
            z: self.z / other.z,
        }
    }
}

impl<T, U> Neg for Vector<T>
where
    T: Neg<Output = U> + Copy + Clone,
{
    type Output = Vector<U>;
    fn neg(self) -> Self::Output {
        Self::Output {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

impl<T> Vector<T>
where
    T: Neg<Output = T> + Copy + Clone,
{
    /// Returns a vector perpendicular to the current one in the 2d plane.
    ///
    /// # Example
    /// ```
    /// use math_vector::Vector;
    /// let v = Vector::new(21.3, -98.1, 0.0);
    /// assert_eq!(Vector::new(98.1, 21.3, 0.0), v.normal());
    /// ```
    pub fn normal(self) -> Self {
        Self {
            x: -self.y,
            y: self.x,
            z: self.z,
        }
    }
}

impl<T, U, V> Vector<T>
where
    T: Mul<T, Output = U> + Copy + Clone,
    U: Add<U, Output = V> + Copy + Clone,
    V: Add<U, Output = V> + Copy + Clone,
{
    /// Get the scalar/dot product of the two `Vector`.
    pub fn dot(v1: Self, v2: Self) -> V {
        v1.x * v2.x + v1.y * v2.y + v1.z * v2.z
    }

    /// Get the squared length of a `Vector`. This is more performant than using
    /// `length()` -- which is only available for `Vector<f32>` and `Vector<f64>`
    /// -- as it does not perform any square root operation.
    pub fn length_squared(self) -> V {
        self.x * self.x + self.y * self.y + self.z * self.z
    }
}

impl<T, U> Vector<T>
where
    T: Mul<T, Output = U> + Copy + Clone,
    U: Sub<U, Output = T> + Copy + Clone,
{
    /// Get the cross product of the two `Vector`.
    pub fn cross(v1: Self, v2: Self) -> Self {
        Self {
            x: v1.y * v2.z - v1.z * v2.y,
            y: v1.z * v2.x - v1.x * v2.z,
            z: v1.x * v2.y - v1.y * v2.x,
        }
    }
}

impl<T> Vector<T>
where
    T: Sub<T, Output = T> + Mul<T, Output = T> + Add<T, Output = T> + Copy + Clone,
{
    /// Linearly interpolates between two vectors
    pub fn lerp(start: Self, end: Self, progress: T) -> Self {
        start + ((end - start) * progress)
    }
}

// From/Into Implementations

impl<T, U> Into<(U, U, U)> for Vector<T>
where
    T: Into<U> + Copy + Clone,
{
    fn into(self) -> (U, U, U) {
        (self.x.into(), self.y.into(), self.z.into())
    }
}

impl<T, U> From<(U, U, U)> for Vector<T>
where
    T: From<U>,
    U: Copy + Clone,
{
    fn from(src: (U, U, U)) -> Vector<T> {
        Vector {
            x: src.0.into(),
            y: src.1.into(),
            z: src.2.into(),
        }
    }
}

impl<T, U> From<[U; 3]> for Vector<T>
where
    T: From<U>,
    U: Copy + Clone,
{
    fn from(src: [U; 3]) -> Vector<T> {
        Vector {
            x: src[0].into(),
            y: src[1].into(),
            z: src[2].into(),
        }
    }
}

// Specific Primitive Implementations

impl Vector<f32> {
    /// If two vectors are almost equal, return true.
    ///
    /// # Example
    /// ```
    /// use math_vector::Vector;
    /// let v = Vector::<f32>::new(-1.0, 0.0, 1.0);
    /// let u = Vector::<f32>::new(-1.000001, 0.000001, 1.000001);
    /// assert_eq!(v.is_close(u), true);
    pub fn is_close(self, other: Self) -> bool {
        let x_diff = self.x - other.x;
        let y_diff = self.y - other.y;
        let z_diff = self.z - other.z;

        (x_diff * x_diff + y_diff * y_diff + z_diff * z_diff) <= EPSILON as f32
    }

    /// Get the length of the vector. If possible, favour `length_squared()` over
    /// this function, as it is more performant.
    pub fn length(self) -> f32 {
        f32::sqrt(self.length_squared())
    }

    /// Get a new vector with the same direction as this vector, but with a length
    /// of 1.0. If the the length of the vector is 0, then the original vector is
    /// returned.
    pub fn normalise(self) -> Self {
        let l = self.length_squared();
        if l == 0.0 {
            return self;
        } else {
            return self * InvSqrt32::inv_sqrt32(l);
        }
    }

    /// Get the distance between two vectors.
    pub fn distance(self, other: Vector<f32>) -> f32 {
        (self - other).length()
    }

    /// Get the vector's direction in radians in the 2d plane.
    pub fn angle(self) -> f32 {
        self.y.atan2(self.x)
    }

    /// Get the angle between two vectors in radians.
    pub fn angle_between(self, other: Vector<f32>) -> f32 {
        let dot = Vector::<f32>::dot(self, other);
        let det = Vector::<f32>::cross(self, other).length();
        f32::atan2(det, dot)
    }

    /// Performs rotation of the vector by the given angle in radians
    ///
    /// # Parameters
    /// - `angle` - The angle to rotate by in radians
    /// - `axis` - The axis to rotate around (note it should be a unit vector)
    pub fn rotate(self, angle: f32, axis: Vector<f32>) -> Self {
        let (sin, cos) = angle.sin_cos();
        let x = (cos + axis.x * axis.x * (1.0 - cos))
            + (axis.x * axis.y * (1.0 - cos) - axis.y * sin)
            + (axis.x * axis.z * (1.0 - cos) - axis.y * sin);
        let y = (axis.x * axis.y * (1.0 - cos) + axis.z * sin)
            + (cos + axis.y * axis.y * (1.0 - cos))
            + (axis.y * axis.z * (1.0 - cos) - axis.x * sin);
        let z = (axis.x * axis.z * (1.0 - cos) + axis.y * sin)
            + (axis.y * axis.z * (1.0 - cos) + axis.x * sin)
            + (cos + axis.z * axis.z * (1.0 - cos));
        Self { x, y, z }
    }

    /// Performs rotation of the vector by the given angle in radians around the z axis
    pub fn rotate_z(self, angle: f32) -> Self {
        let (sin, cos) = angle.sin_cos();
        let x = self.x * cos - self.y * sin;
        let y = self.x * sin + self.y * cos;
        Self { x, y, z: self.z }
    }

    fn_simple_as!(i32);
    fn_simple_as!(i64);
    fn_simple_as!(isize);
    fn_lower_bounded_as!(f32, u32, 0.0);
    fn_lower_bounded_as!(f32, u64, 0.0);
    fn_lower_bounded_as!(f32, usize, 0.0);
}

impl Vector<f64> {
    /// If two vectors are almost equal, return true.
    ///
    /// # Example
    /// ```
    /// use math_vector::Vector;
    /// let v = Vector::<f64>::new(-1.0, 0.0, 1.0);
    /// let u = Vector::<f64>::new(-1.000001, 0.000001, 1.000001);
    /// assert_eq!(v.is_close(u), true);
    pub fn is_close(self, other: Self) -> bool {
        let x_diff = self.x - other.x;
        let y_diff = self.y - other.y;
        let z_diff = self.z - other.z;

        (x_diff * x_diff + y_diff * y_diff + z_diff * z_diff) <= EPSILON
    }

    /// Get the length of the vector. If possible, favour `length_squared()` over
    /// this function, as it is more performant.
    pub fn length(self) -> f64 {
        f64::sqrt(self.length_squared())
    }

    /// Get a new vector with the same direction as this vector, but with a length
    /// of 1.0. If the the length of the vector is 0, then the original vector is
    /// returned.
    pub fn normalise(self) -> Self {
        let l = self.length_squared();
        if l == 0.0 {
            return self;
        } else {
            return self * InvSqrt64::inv_sqrt64(l);
        }
    }

    /// Get the distance between two vectors.
    pub fn distance(self, other: Vector<f64>) -> f64 {
        (self - other).length()
    }

    /// Get the vector's direction in radians in the 2d plane.
    pub fn angle(self) -> f64 {
        self.y.atan2(self.x)
    }

    /// Get the angle between two vectors in radians.
    pub fn angle_between(self, other: Vector<f64>) -> f64 {
        let dot = Vector::<f64>::dot(self, other);
        let det = Vector::<f64>::cross(self, other).length();
        f64::atan2(det, dot)
    }

    /// Performs rotation of the vector by the given angle in radians
    ///
    /// # Parameters
    /// - `angle` - The angle to rotate by in radians
    /// - `axis` - The axis to rotate around (note it should be a unit vector)
    pub fn rotate(self, angle: f64, axis: Vector<f64>) -> Self {
        let (sin, cos) = angle.sin_cos();
        let x = (cos + axis.x * axis.x * (1.0 - cos))
            + (axis.x * axis.y * (1.0 - cos) - axis.y * sin)
            + (axis.x * axis.z * (1.0 - cos) - axis.y * sin);
        let y = (axis.x * axis.y * (1.0 - cos) + axis.z * sin)
            + (cos + axis.y * axis.y * (1.0 - cos))
            + (axis.y * axis.z * (1.0 - cos) - axis.x * sin);
        let z = (axis.x * axis.z * (1.0 - cos) + axis.y * sin)
            + (axis.y * axis.z * (1.0 - cos) + axis.x * sin)
            + (cos + axis.z * axis.z * (1.0 - cos));
        Self { x, y, z }
    }

    /// Performs rotation of the vector by the given angle in radians around the z axis
    pub fn rotate_z(self, angle: f64) -> Self {
        let (sin, cos) = angle.sin_cos();
        let x = self.x * cos - self.y * sin;
        let y = self.x * sin + self.y * cos;
        Self { x, y, z: self.z }
    }

    fn_simple_as!(i32);
    fn_simple_as!(i64);
    fn_simple_as!(isize);
    fn_simple_as!(f32);
    fn_lower_bounded_as!(f64, u32, 0.0);
    fn_lower_bounded_as!(f64, u64, 0.0);
    fn_lower_bounded_as!(f64, usize, 0.0);
}

impl Vector<i32> {
    fn_simple_as!(isize);
    fn_simple_as!(f32);
    fn_simple_as!(f64);
    fn_lower_bounded_as!(i32, u32, 0);
    fn_lower_bounded_as!(i32, u64, 0);
    fn_lower_bounded_as!(i32, usize, 0);
}

impl Vector<i64> {
    fn_simple_as!(i32);
    fn_simple_as!(isize);
    fn_simple_as!(f32);
    fn_simple_as!(f64);
    fn_lower_bounded_as!(i64, u32, 0);
    fn_lower_bounded_as!(i64, u64, 0);
    fn_lower_bounded_as!(i64, usize, 0);
}

impl Vector<isize> {
    fn_simple_as!(i32);
    fn_simple_as!(i64);
    fn_simple_as!(f32);
    fn_simple_as!(f64);
    fn_lower_bounded_as!(isize, u32, 0);
    fn_lower_bounded_as!(isize, u64, 0);
    fn_lower_bounded_as!(isize, usize, 0);
}

impl Vector<u32> {
    fn_simple_as!(i32);
    fn_simple_as!(i64);
    fn_simple_as!(isize);
    fn_simple_as!(f32);
    fn_simple_as!(f64);
    fn_simple_as!(usize);
}

impl Vector<u64> {
    fn_simple_as!(i32);
    fn_simple_as!(i64);
    fn_simple_as!(isize);
    fn_simple_as!(f32);
    fn_simple_as!(f64);
    fn_simple_as!(u32);
    fn_simple_as!(usize);
}

impl Vector<usize> {
    fn_simple_as!(i32);
    fn_simple_as!(i64);
    fn_simple_as!(isize);
    fn_simple_as!(f32);
    fn_simple_as!(f64);
    fn_simple_as!(u32);
    fn_simple_as!(u64);
}

// Ops Implementations

impl<T, O> Add<Vector<T>> for Vector<T>
where
    T: Add<T, Output = O> + Copy + Clone,
{
    type Output = Vector<O>;
    fn add(self, rhs: Vector<T>) -> Self::Output {
        Vector {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl<T, O> Add<&Vector<T>> for &Vector<T>
where
    T: Add<T, Output = O> + Copy + Clone,
{
    type Output = Vector<O>;
    fn add(self, rhs: &Vector<T>) -> Self::Output {
        Vector {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl<T> AddAssign<Vector<T>> for Vector<T>
where
    T: Add<T, Output = T> + Copy + Clone,
{
    fn add_assign(&mut self, rhs: Vector<T>) {
        self.x = self.x + rhs.x;
        self.y = self.y + rhs.y;
        self.z = self.z + rhs.z;
    }
}

impl<T, O> Sub<Vector<T>> for Vector<T>
where
    T: Sub<T, Output = O> + Copy + Clone,
{
    type Output = Vector<O>;
    fn sub(self, rhs: Vector<T>) -> Self::Output {
        Vector {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl<T, O> Sub<&Vector<T>> for &Vector<T>
where
    T: Sub<T, Output = O> + Copy + Clone,
{
    type Output = Vector<O>;
    fn sub(self, rhs: &Vector<T>) -> Self::Output {
        Vector {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl<T> SubAssign<Vector<T>> for Vector<T>
where
    T: Sub<T, Output = T> + Copy + Clone,
{
    fn sub_assign(&mut self, rhs: Vector<T>) {
        self.x = self.x - rhs.x;
        self.y = self.y - rhs.y;
        self.z = self.z - rhs.z;
    }
}

impl<T, O> Mul<T> for Vector<T>
where
    T: Mul<T, Output = O> + Copy + Clone,
{
    type Output = Vector<O>;
    fn mul(self, rhs: T) -> Self::Output {
        Vector {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
        }
    }
}

impl<T, O> Mul<T> for &Vector<T>
where
    T: Mul<T, Output = O> + Copy + Clone,
{
    type Output = Vector<O>;
    fn mul(self, rhs: T) -> Self::Output {
        Self::Output {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
        }
    }
}

impl<T> MulAssign<T> for Vector<T>
where
    T: Mul<T, Output = T> + Copy + Clone,
{
    fn mul_assign(&mut self, rhs: T) {
        self.x = self.x * rhs;
        self.y = self.y * rhs;
        self.z = self.z * rhs;
    }
}

impl<T, O> Div<T> for Vector<T>
where
    T: Div<T, Output = O> + Copy + Clone,
{
    type Output = Vector<O>;
    fn div(self, rhs: T) -> Self::Output {
        Self::Output {
            x: self.x / rhs,
            y: self.y / rhs,
            z: self.z / rhs,
        }
    }
}

impl<T, O> Div<T> for &Vector<T>
where
    T: Div<T, Output = O> + Copy + Clone,
{
    type Output = Vector<O>;
    fn div(self, rhs: T) -> Self::Output {
        Self::Output {
            x: self.x / rhs,
            y: self.y / rhs,
            z: self.z / rhs,
        }
    }
}

impl<T> DivAssign<T> for Vector<T>
where
    T: Div<T, Output = T> + Copy + Clone,
{
    fn div_assign(&mut self, rhs: T) {
        self.x = self.x / rhs;
        self.y = self.y / rhs;
        self.z = self.z / rhs;
    }
}
