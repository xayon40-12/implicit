use crate::fixed_point::{self, fixedpoint, fixedpoint_newton, newton};
pub mod schemes;
use crate::vector_space::VectorSpace;
use schemes::RK;

pub fn explicit<const N: usize, const M: usize>(
    x: [f32; M],
    aij: [[f32; N]; N],
    f: &impl Fn(f32, [f32; M]) -> [f32; M],
    t: f32,
    dt: f32,
) -> ([[f32; M]; N], usize) {
    let mut k = [[0.0; M]; N];
    for i in 0..N {
        k[i] = f(
            t + dt * aij[i].iter().sum::<f32>(),
            x.add(
                aij[i]
                    .into_iter()
                    .zip(k.into_iter())
                    .map(|(a, k)| k.scal_mul(a))
                    .fold([0.0; M], |a, k| a.add(k))
                    .scal_mul(dt),
            ),
        );
    }
    (k, 1)
}

pub fn implicit<const N: usize, const M: usize>(
    x: [f32; M],
    aij: [[f32; N]; N],
    f: &impl Fn(f32, [f32; M]) -> [f32; M],
    t: f32,
    dt: f32,
    max_iter: usize,
    _k0: [[f32; M]; N],
) -> ([[f32; M]; N], usize) {
    let fk = |k: [[f32; M]; N]| {
        let mut res = [[0.0; M]; N];
        for i in 0..N {
            let aijk = aij[i]
                .into_iter()
                .zip(k)
                .map(|(a, k)| k.scal_mul(a))
                .fold([0.0; M], |a, k| a.add(k));
            res[i] = f(
                t + dt * aij[i].iter().sum::<f32>(),
                x.add(aijk.scal_mul(dt)),
            );
        }
        res
    };
    let zeros = [[0.0; M]; N];
    // newton(zeros, max_iter, &fk)
    // fixedpoint_newton(zeros, dt, max_iter, fk)
    fixedpoint(zeros, 1e-6, max_iter, fk)
}

pub fn rk_step<const N: usize, const M: usize>(
    x: [f32; M],
    (aij, bj, is_explicit): RK<N>,
    f: &impl Fn(f32, [f32; M]) -> [f32; M],
    t: f32,
    dt: f32,
    max_iter: usize,
    k0: [[f32; M]; N],
) -> ([f32; M], [[f32; M]; N], usize) {
    let (k, count) = if is_explicit {
        explicit(x, aij, &f, t, dt)
    } else {
        implicit(x, aij, &f, t, dt, max_iter, k0)
    };
    let x = x.add(
        bj.into_iter()
            .zip(k)
            .map(|(b, k)| k.scal_mul(b))
            .fold([0.0; M], |a, k| a.add(k))
            .scal_mul(dt),
    );
    (x, k, count)
}

pub fn integrate<const N: usize, const M: usize, T>(
    mut x: [f32; M],
    t0: f32,
    dt: f32,
    t_max: f32,
    rk: RK<N>,
    f: impl Fn(f32, [f32; M]) -> [f32; M],
    out: impl Fn([f32; M]) -> T,
    max_iter: usize,
) -> (Vec<f32>, Vec<T>, usize, usize, usize, usize, f32) {
    let n = (t_max / dt) as usize;
    let mut history = Vec::with_capacity(n);
    let mut ts = Vec::with_capacity(n);
    let mut count_tot = 0;
    let mut max_count = 0;
    let mut t = t0;
    history.push(out(x));
    ts.push(t0);
    let mut k = [[0.0; M]; N];
    for _ in 0..n {
        let (x1, k1, count) = rk_step(x, rk, &f, t, dt, max_iter, k);
        k = k1;
        x = x1;
        history.push(out(x));
        max_count = max_count.max(count);
        count_tot += count;
        t += dt;
        ts.push(t);
    }
    (ts, history, count_tot * N, count_tot, max_count, n, t)
}
