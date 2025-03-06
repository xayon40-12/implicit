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
    while (x0.sub(x1).norm2()) / x1.norm2() > accuracy && count < 10 {
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

pub fn integrate(mut x: f32, dt: f32, t_max: f32, f: impl Fn(f32) -> f32) -> (usize, usize, f32) {
    let mut k = 0.0;
    let mut count;
    let mut count_tot = 0;
    let mut max_count = 0;
    let mut t = 0.0;
    let n = (t_max / dt) as usize;
    for _ in 0..n {
        let fk = |k: f32| f(x.add(k.times(dt))); // TODO: use RadauIIA for higher accuracy and stability
        (k, count) = fixedpoint(k, dt * dt, fk);
        x = x.add(k.times(dt));
        max_count = max_count.max(count);
        count_tot += count;
        t += dt;
        let sol = f(t).exp();
        println!(
            "t: {t:.3}   count: {count}, val: {x:.10e}, sol: {sol:.10e}, err: {:.10e} ",
            (sol - x) / sol
        );
    }
    (count_tot, max_count, x)
}

fn main() {
    let sgn = |x: f32| -x;
    let m = 1e4f32;
    let f = |x: f32| sgn(m * x);
    let dt = 1e-3;
    let t_max = sgn(10.0f32.powf(sgn(15.0)).ln()) / m;
    let x = 1.0;
    let (tot, max, x) = integrate(x, dt, t_max, f);
    println!("count tot: {tot}, max: {max}, x: {x:.10e}");
}
