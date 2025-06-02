use crate::fixed_point::fixedpoint;
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
) -> ([[f32; M]; N], usize) {
    let zeros = [[0.0; M]; N];
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
    fixedpoint(zeros, 1e-7, fk) //TODO find a better accuracy threshold, or maybe 0.0 but by including a few history of the iteration of x to compare to older values to detect loops
}

pub fn integrate<const N: usize, const M: usize, T>(
    mut x: [f32; M],
    t0: f32,
    dt: f32,
    t_max: f32,
    (aij, bj, is_explicit): RK<N>,
    f: impl Fn(f32, [f32; M]) -> [f32; M],
    out: impl Fn([f32; M]) -> T,
) -> (Vec<f32>, Vec<T>, usize, usize, usize, usize, f32) {
    let n = (t_max / dt) as usize;
    let mut history = Vec::with_capacity(n);
    let mut ts = Vec::with_capacity(n);
    let mut count_tot = 0;
    let mut max_count = 0;
    let mut t = t0;
    history.push(out(x));
    ts.push(t0);
    for _ in 0..n {
        let (k, count) = if is_explicit {
            explicit(x, aij, &f, t, dt)
        } else {
            implicit(x, aij, &f, t, dt)
        };
        let x1 = x.add(
            bj.into_iter()
                .zip(k)
                .map(|(b, k)| k.scal_mul(b))
                .fold([0.0; M], |a, k| a.add(k))
                .scal_mul(dt),
        );
        x = x1;
        history.push(out(x));
        max_count = max_count.max(count);
        count_tot += count;
        t += dt;
        ts.push(t);
    }
    (ts, history, count_tot * N, count_tot, max_count, n, t)
}
