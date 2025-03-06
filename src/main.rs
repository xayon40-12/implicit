pub trait VectorSpace<T = f32> {
    fn dot(self, rhs: Self) -> T;
    fn norm2(self) -> T
    where
        Self: Sized + Copy,
    {
        self.dot(self)
    }
    fn add(self, rhs: Self) -> Self;
    fn sub(self, rhs: Self) -> Self;
    fn mul(self, rhs: Self) -> Self;
    fn div(self, rhs: Self) -> Self;
    fn times(self, rhs: T) -> Self;
}

impl VectorSpace<f32> for f32 {
    fn dot(self, rhs: Self) -> f32 {
        self * rhs
    }
    fn add(self, rhs: Self) -> Self {
        self + rhs
    }
    fn sub(self, rhs: Self) -> Self {
        self - rhs
    }
    fn mul(self, rhs: Self) -> Self {
        self * rhs
    }
    fn div(self, rhs: Self) -> Self {
        self / rhs
    }
    fn times(self, rhs: f32) -> Self {
        self * rhs
    }
}
impl<const N: usize, V: VectorSpace<f32> + Copy> VectorSpace<f32> for [V; N] {
    fn dot(self, rhs: Self) -> f32 {
        self.into_iter()
            .zip(rhs.into_iter())
            .map(|(v, w)| v.dot(w))
            .fold(0.0, f32::add)
    }
    fn add(mut self, rhs: Self) -> Self {
        self.iter_mut()
            .zip(rhs.into_iter())
            .for_each(|(v, w)| *v = v.add(w));
        self
    }
    fn sub(mut self, rhs: Self) -> Self {
        self.iter_mut()
            .zip(rhs.into_iter())
            .for_each(|(v, w)| *v = v.sub(w));
        self
    }
    fn mul(mut self, rhs: Self) -> Self {
        self.iter_mut()
            .zip(rhs.into_iter())
            .for_each(|(v, w)| *v = v.mul(w));
        self
    }
    fn div(mut self, rhs: Self) -> Self {
        self.iter_mut()
            .zip(rhs.into_iter())
            .for_each(|(v, w)| *v = v.div(w));
        self
    }
    fn times(mut self, rhs: f32) -> Self {
        self.iter_mut().for_each(|v| *v = v.times(rhs));
        self
    }
}

pub fn fixedpoint<V: VectorSpace + Copy>(
    mut x0: V,
    accuracy: f32,
    f: impl Fn(V) -> V,
) -> (V, usize) {
    let mut fx0 = f(x0);
    let mut gx0 = fx0.sub(x0);
    let mut x1 = fx0;
    let mut count = 1;
    while (x0.sub(x1).norm2()) / x1.norm2().max(x0.norm2()) > accuracy && count < 100 {
        count += 1;
        x0 = x1;
        let fx1 = f(x1);
        let gx1 = fx1.sub(x1);
        let gx10 = gx1.sub(gx0);
        if gx10.norm2() == 0.0 {
            break;
        }
        let a = gx1.dot(gx10) / gx10.dot(gx10);
        x1 = fx0.times(a).add(fx1.times(1.0 - a));
        fx0 = fx1;
        gx0 = gx1;
    }
    (x1, count)
}

pub fn integrate<const N: usize, const M: usize>(
    mut x: [f32; M],
    dt: f32,
    t_max: f32,
    aij: [[f32; N]; N],
    bj: [f32; N],
    f: impl Fn([f32; M]) -> [f32; M],
) -> (usize, usize, [f32; M], f32) {
    let mut k = [[0.0; M]; N];
    let mut count;
    let mut count_tot = 0;
    let mut max_count = 0;
    let mut t = 0.0;
    let n = (t_max / dt) as usize;
    for _ in 0..n {
        let fk = |k: [[f32; M]; N]| {
            let mut res = [[0.0; M]; N];
            for i in 0..N {
                let aijk = aij[i]
                    .into_iter()
                    .zip(k)
                    .map(|(a, k)| k.times(a))
                    .fold([0.0; M], |a, k| a.add(k));
                res[i] = f(x.add(aijk.times(dt)));
            }
            res
        }; // TODO: use RadauIIA for higher accuracy and stability
        (k, count) = fixedpoint(k, dt * dt, fk);
        x = x.add(
            bj.into_iter()
                .zip(k)
                .map(|(b, k)| k.times(b))
                .fold([0.0; M], |a, k| a.add(k))
                .times(dt),
        );
        max_count = max_count.max(count);
        count_tot += count;
        t += dt;
        // let sol = f([t; M])[0].exp();
        // println!(
        //     "t: {t:.3}   count: {count}, val: {:.10e}, sol: {sol:.10e}, err: {:.10e} ",
        //     x[0],
        //     (sol - x[0]) / sol
        // );
    }
    let sol = f([t; M])[0].exp();
    println!(
        "count tot: {count_tot}, max: {max_count}, x: {:.10e}, err: {:.10e}",
        x[0],
        (sol - x[0]) / sol
    );
    (count_tot, max_count, x, t)
}

fn main() {
    let sgn = |x: f32| -x;
    let dt = 7.5e-4;
    let x = [1.0];
    for i in 0..8 {
        let m = 10.0f32.powf(i as f32 * 0.5);
        println!("m: {m:e}\n");
        let f = |x: [f32; 1]| [sgn(m * x[0])];
        let t_max = sgn(10.0f32.powf(sgn(10.0)).ln()) / m;

        println!("Euler:");
        let bj = [1.0];
        let aij = [[1.0]];
        integrate(x, dt, t_max, aij, bj, f);
        println!("LabattoIIIC2:");
        let bj = [0.5, 0.5];
        let aij = [[0.5, -0.5], bj];
        integrate(x, dt, t_max, aij, bj, f);
        println!("RadauIIA2:");
        let bj = [0.75, 0.25];
        let aij = [[5.0 / 12.0, -1.0 / 12.0], bj];
        integrate(x, dt, t_max, aij, bj, f);
        println!("LabattoIIIC3:");
        let bj = [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0];
        let aij = [
            [1.0 / 6.0, -1.0 / 3.0, 1.0 / 6.0],
            [1.0 / 6.0, 5.0 / 12.0, -1.0 / 12.0],
            bj,
        ];
        integrate(x, dt, t_max, aij, bj, f);
        println!("RadauIIA3:");
        let s6 = 6.0f32.sqrt();
        let bj = [4.0 / 9.0 - s6 / 36.0, 4.0 / 9.0 + s6 / 36.0, 1.0 / 9.0];
        let aij = [
            [
                11.0 / 45.0 - 7.0 * s6 / 360.0,
                37.0 / 225.0 - 169.0 * s6 / 1800.0,
                -2.0 / 225.0 + s6 / 75.0,
            ],
            [
                37.0 / 225.0 + 169.0 * s6 / 1800.0,
                11.0 / 45.0 + 7.0 * s6 / 360.0,
                -2.0 / 225.0 - s6 / 75.0,
            ],
            bj,
        ];
        integrate(x, dt, t_max, aij, bj, f);
        println!("\n");
    }
}
