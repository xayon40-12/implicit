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
    fn scal_mul(self, rhs: T) -> Self;
    fn normalized(self) -> Self;
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
    fn scal_mul(self, rhs: f32) -> Self {
        self * rhs
    }
    fn normalized(self) -> Self {
        1.0
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
    fn scal_mul(mut self, rhs: f32) -> Self {
        self.iter_mut().for_each(|v| *v = v.scal_mul(rhs));
        self
    }
    fn normalized(self) -> Self {
        let n = self.norm2().sqrt();
        self.scal_mul(n.recip())
    }
}
