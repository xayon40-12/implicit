use crate::vector_space::VectorSpace;

pub fn newton<V: VectorSpace + Copy + std::fmt::Debug>(
    mut x0: V,
    max_iter: usize,
    f: &dyn Fn(V) -> V,
) -> (V, usize) {
    let e = 0.5f32;

    // return fixedpoint(x0, e * e, max_iter+1, f);

    // NOTE: instead of directly using a fixed-point
    // - we construct the newton method `g'(x_n)*z_n = -g(x_n)` with `z_n = x_{n+1} - x_n`
    // - we use the approx `g'(u)v ~= (g(u+e*v)-g(u))/e`
    // - finaly we solve with a fixed point `f(z_n) = (g(x_n+e*z_n)-g(x_n))/e + g(x_n) + z_n`
    // - the results are actually better by multiplying the newton part by sqrt(e) `f(z_n) = (g(x_n+e*z_n)-g(x_n))/sqrt(e) + sqrt(e)*g(x_n) + z_n`
    let g = |x| f(x).sub(x);
    let esqrtrecip = e.sqrt().recip();
    let mut count_tot = 0;
    for i in 0..max_iter {
        // do only one Newton iteration
        let gx0 = g(x0);
        let gx0emgx0 = gx0.scal_mul(e - 1.0); // g(x0)*e - g(x0)
        let fnewt = |x: V| {
            g(x0.add(x.scal_mul(e)))
                .add(gx0emgx0)
                .scal_mul(esqrtrecip)
                .add(x)
        };
        let start = gx0.scal_mul(e.sqrt());
        let dx = fnewt(start);
        count_tot += 2;
        x0 = x0.add(dx);
    }
    (x0, count_tot)
}

pub fn fixedpoint_newton<V: VectorSpace + Copy + std::fmt::Debug>(
    mut x0: V,
    accuracy: f32,
    max_iter: usize,
    f: impl Fn(V) -> V,
) -> (V, usize) {
    let e = 1e-1f32; //(accuracy*10.0).min(0.5).max(1e-2);

    // return fixedpoint(x0, e * e, max_iter+1, f);

    // NOTE: instead of directly using a fixed-point
    // - we construct the newton method `g'(x_n)*z_n = -g(x_n)` with `z_n = x_{n+1} - x_n`
    // - we use the approx `g'(u)v ~= (g(u+e*v)-g(u))/e`
    // - finaly we solve with a fixed point `f(z_n) = (g(x_n+e*z_n)-g(x_n))/e + g(x_n) + z_n`
    // - the results are actually better by multiplying the newton part by sqrt(e) `f(z_n) = (g(x_n+e*z_n)-g(x_n))/sqrt(e) + sqrt(e)*g(x_n) + z_n`
    let g = |x| f(x).sub(x);
    let mut count_tot = 0;
    for i in 0..2 {
        // do only one Newton iteration
        let gx0 = g(x0);
        let esqrtrecip = e.sqrt().recip();
        let gx0emgx0 = gx0.scal_mul(e - 1.0); // g(x0)*e - g(x0)
        let start = gx0.scal_mul(e.sqrt());
        let fnewt = |x: V| {
            g(x0.add(x.scal_mul(e)))
                .add(gx0emgx0)
                .scal_mul(esqrtrecip)
                .add(x)
        };
        let (dx, count) = fixedpoint(start, e * e, max_iter, &fnewt);
        count_tot += count + 1; // +1 for fx0
        x0 = x0.add(dx);
    }
    (x0, count_tot)
}

pub fn fixedpoint<V: VectorSpace + Copy + std::fmt::Debug>(
    mut x0: V,
    accuracy: f32,
    max_iter: usize,
    f: impl Fn(V) -> V,
) -> (V, usize) {
    let mut fx = [f(x0); 2];
    let mut gx = [fx[0].sub(x0); 2];
    let mut x1 = fx[0];
    let mut count = 1;
    let err = |x0: V, x1: V| (x0.sub(x1).norm2()) / (x1.norm2().max(x0.norm2() + 1e-20));
    while err(x0, x1) > accuracy && count < max_iter {
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
