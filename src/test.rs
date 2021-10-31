use crate::Vector;

#[test]
fn dot() {
    let v1 = Vector::new(10.0, 5.0, 0.0);
    let v2 = Vector::new(1.5, 2.0, 0.0);

    let result = Vector::dot(v1, v2);

    assert_eq!(25.0, result);
}

#[test]
fn from_vec() {
    let iv = Vector::new(10, 5, -1);
    let fv = Vector::from_vec(iv);
    assert_eq!(Vector::new(10.0, 5.0, -1.0), fv);
}

#[test]
fn into_vec() {
    let iv = Vector::new(10, 5, -1);
    let fv = iv.into_vec();
    assert_eq!(Vector::new(10.0, 5.0, -1.0), fv);
}

#[test]
fn from_tuple() {
    let ituple = (10, 5, -1);
    let fv = ituple.into();
    assert_eq!(Vector::new(10.0, 5.0, -1.0), fv);
}

#[test]
fn from_array() {
    let arr = [10, 5, -1];
    let fv = arr.into();
    assert_eq!(Vector::new(10.0, 5.0, -1.0), fv);
}

#[test]
fn length_squared() {
    let v = Vector::new(10, 5, 0);
    let r = v.length_squared();
    assert_eq!(125, r);
}

#[test]
fn length_f32() {
    let v: Vector<f32> = Vector::new(3.0, 4.0, 0.0);
    let r: f32 = v.length();
    assert_eq!(5.0, r);
}

#[test]
fn length_f64() {
    let v: Vector<f64> = Vector::new(3.0, 4.0, 0.0);
    let r: f64 = v.length();
    assert_eq!(5.0, r);
}

#[test]
fn angle_f32() {
    let v: Vector<f32> = Vector::new(2.0, 2.0, 0.0);
    let r: f32 = v.angle();
    assert_eq!(std::f32::consts::PI / 4.0, r);
}

#[test]
fn angle_f64() {
    let v: Vector<f64> = Vector::new(2.0, 2.0, 0.0);
    let r: f64 = v.angle();
    assert_eq!(std::f64::consts::PI / 4.0, r);
}

#[test]
fn add() {
    let v1 = Vector::new(10.0, 5.0, 3.0);
    let v2 = Vector::new(1.5, 2.0, 1.0);

    let result = v1 + v2;

    assert_eq!(Vector::new(11.5, 7.0, 4.0), result);
}

#[test]
fn add_assign() {
    let mut v1 = Vector::new(10.0, 5.0, 3.0);
    let v2 = Vector::new(1.5, 2.0, 1.0);

    v1 += v2;

    assert_eq!(Vector::new(11.5, 7.0, 4.0), v1);
}

#[test]
fn sub() {
    let v1 = Vector::new(10.0, 5.0, 3.0);
    let v2 = Vector::new(1.5, 2.0, 1.0);

    let result = v1 - v2;

    assert_eq!(Vector::new(8.5, 3.0, 2.0), result);
}

#[test]
fn sub_assign() {
    let mut v1 = Vector::new(10.0, 5.0, 3.0);
    let v2 = Vector::new(1.5, 2.0, 1.0);

    v1 -= v2;

    assert_eq!(Vector::new(8.5, 3.0, 2.0), v1);
}

#[test]
fn mul() {
    let v = Vector::new(10.0, 5.0, 3.0);
    let f = 2.0;

    let result = v * f;

    assert_eq!(Vector::new(20.0, 10.0, 6.0), result);
}

#[test]
fn mul_assign() {
    let mut v = Vector::new(10.0, 5.0, 3.0);
    let f = 2.0;

    v *= f;

    assert_eq!(Vector::new(20.0, 10.0, 6.0), v);
}

#[test]
fn div() {
    let v = Vector::new(10.0, 5.0, 3.0);
    let f = 2.0;

    let result = v / f;

    assert_eq!(Vector::new(5.0, 2.5, 1.5), result);
}

#[test]
fn div_assign() {
    let mut v = Vector::new(10.0, 5.0, 3.0);
    let f = 2.0;

    v /= f;

    assert_eq!(Vector::new(5.0, 2.5, 1.5), v);
}

#[test]
fn f64_as_i32() {
    let fv: Vector<f64> = Vector::new(10.5, 11.2, 11.9);
    let iv = fv.as_i32s();
    assert_eq!(Vector::new(10, 11, 11), iv);
}

#[test]
fn f32_as_u32() {
    let fv: Vector<f32> = Vector::new(10.5, 11.2, 11.9);
    let uv = fv.as_u32s();
    assert_eq!(Vector::new(10, 11, 11), uv);
}

#[test]
fn f32_as_u32_bounded() {
    let fv: Vector<f32> = Vector::new(-10.5, -11.2, 11.9);
    let uv = fv.as_u32s();
    assert_eq!(Vector::new(0, 0, 11), uv);
}

#[test]
fn lerp() {
    let start = Vector::new(5.0, 10.0, 15.0);
    let end = Vector::new(10.0, 11.5, 12.0);

    let result = Vector::lerp(start, end, 0.5);

    assert_eq!(Vector::new(7.5, 10.75, 13.5), result);
}

#[test]
fn neg() {
    let v = Vector::new(10.3, -5.4, -3.2);
    assert_eq!(Vector::new(-10.3, 5.4, 3.2), -v);
}

#[test]
fn normalise_f32() {
    let v = Vector::<f32>::new(1.0, 1.0, 1.0).normalise();
    let sqrt3 = f32::sqrt(3.0);
    let u = Vector::<f32>::new(1.0 / sqrt3, 1.0 / sqrt3, 1.0 / sqrt3);
    assert_eq!(v.is_close(u), true);
}

#[test]
fn normalise_f64() {
    let v = Vector::<f64>::new(1.0, 1.0, 1.0).normalise();
    let sqrt3 = f64::sqrt(3.0);
    let u = Vector::<f64>::new(1.0 / sqrt3, 1.0 / sqrt3, 1.0 / sqrt3);
    assert_eq!(v.is_close(u), true);
}

#[test]
fn rotate_z_f32() {
    let v = Vector::<f32>::new(1.0, 0.0, 0.0);
    let r = v.rotate_z(std::f32::consts::PI / 2.0);
    assert_eq!(Vector::<f32>::new(0.0, 1.0, 0.0).is_close(r), true);
}

#[test]
fn rotate_z_f64() {
    let v = Vector::<f64>::new(1.0, 0.0, 0.0);
    let r = v.rotate_z(std::f64::consts::PI / 2.0);
    assert_eq!(Vector::<f64>::new(0.0, 1.0, 0.0).is_close(r), true);
}

#[test]
fn rotate_y_f32() {
    let v = Vector::<f32>::new(1.0, 0.0, 0.0);
    let r = v.rotate_y(std::f32::consts::PI / 2.0);
    assert_eq!(Vector::<f32>::new(0.0, 0.0, -1.0).is_close(r), true);
}

#[test]
fn rotate_y_f64() {
    let v = Vector::<f64>::new(1.0, 0.0, 0.0);
    let r = v.rotate_y(std::f64::consts::PI / 2.0);
    assert_eq!(Vector::<f64>::new(0.0, 0.0, -1.0).is_close(r), true);
}

#[test]
fn rotate_x_f32() {
    let v = Vector::<f32>::new(0.0, 1.0, 0.0);
    let r = v.rotate_x(std::f32::consts::PI / 2.0);
    assert_eq!(Vector::<f32>::new(0.0, 0.0, 1.0).is_close(r), true);
}

#[test]
fn rotate_x_f64() {
    let v = Vector::<f64>::new(0.0, 1.0, 0.0);
    let r = v.rotate_x(std::f64::consts::PI / 2.0);
    assert_eq!(Vector::<f64>::new(0.0, 0.0, 1.0).is_close(r), true);
}

#[test]
fn rotate_f32() {
    let v = Vector::<f32>::new(1.0, 0.0, 0.0);
    let axis = Vector::<f32>::new(1.0, 1.0, 0.0).normalise();
    let r = v.rotate(std::f32::consts::PI / 2.0, axis);
    assert_eq!(
        Vector::<f32>::new(0.5, 0.5, -2f32.sqrt().recip()).is_close(r),
        true
    );
}

#[test]
fn rotate_f64() {
    let v = Vector::<f64>::new(1.0, 0.0, 0.0);
    let axis = Vector::<f64>::new(1.0, 1.0, 0.0).normalise();
    let r = v.rotate(std::f64::consts::PI / 2.0, axis);
    assert_eq!(
        Vector::<f64>::new(0.5, 0.5, -2f64.sqrt().recip()).is_close(r),
        true
    );
}
