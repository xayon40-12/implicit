use crate::vector_space::VectorSpace;

pub fn fixedpoint<V: VectorSpace + Copy + std::fmt::Debug>(
    mut x0: V,
    accuracy: f32,
    f: impl Fn(V) -> V,
) -> (V, usize) {
    let mut fx = [f(x0); 2];
    let mut gx = [fx[0].sub(x0); 2];
    let mut x1 = fx[0];
    let mut count = 1;
    let err = |x0: V, x1: V| (x0.sub(x1).norm2()) / (x1.norm2().max(x0.norm2() + 1e-20));
    while err(x0, x1) > accuracy && count < 100 {
        x0 = x1;
        fx[1] = f(x1);
        gx[1] = fx[1].sub(x1);
        count += 1;
        let gx01 = gx[0].sub(gx[1]);
        let gx0101 = gx01.dot(gx01);
        let a = -gx[1].dot(gx01) / gx0101;
        x1 = fx[0].scal_mul(a).add(fx[1].scal_mul(1.0 - a));
        fx[0] = fx[1];
        gx[0] = gx[1];
    }
    (x1, count)
}
