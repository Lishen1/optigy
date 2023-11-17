use nalgebra::{DMatrixViewMut, DVectorView, DVectorViewMut, RealField, Vector2};
use optigy::prelude::Variable;

#[derive(Debug, Clone)]
pub struct E2<R = f64>
where
    R: RealField,
{
    pub val: Vector2<R>,
}

impl<R> Variable<R> for E2<R>
where
    R: RealField,
{
    fn local(&self, linearization_point: &Self, mut tangent: DVectorViewMut<R>)
    where
        R: RealField,
    {
        let d = self.val.clone() - linearization_point.val.clone();
        tangent.copy_from(&d);
    }

    fn retract(&mut self, delta: DVectorView<R>)
    where
        R: RealField,
    {
        self.val = self.val.clone() + delta
    }

    fn dim(&self) -> usize {
        2
    }

    fn retract_local_jacobian(&self, _linearization_point: &Self, mut jacobian: DMatrixViewMut<R>) {
        jacobian.fill_with_identity();
    }
}
impl<R> E2<R>
where
    R: RealField,
{
    pub fn new(x: f64, y: f64) -> Self {
        E2 {
            val: Vector2::new(R::from_f64(x).unwrap(), R::from_f64(y).unwrap()),
        }
    }
}
