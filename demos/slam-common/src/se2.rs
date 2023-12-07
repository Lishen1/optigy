use manif::{se2::SE2Tangent, LieGroupBase, TangentBase};
use nalgebra::{convert, DVectorView, DVectorViewMut, Isometry2, Vector2, Vector3, VectorView3};

use optigy::{core::variable::Variable, prelude::Real};

#[derive(Debug, Clone)]
pub struct SE2<R = f64>
where
    R: Real,
{
    pub origin: Isometry2<R>,
}

impl<R> Variable<R> for SE2<R>
where
    R: Real,
{
    // value is linearization point
    fn local(&self, linearization_point: &Self, mut tangent: DVectorViewMut<R>)
    where
        R: Real,
    {
        let d = (linearization_point.origin.inverse() * &self.origin)
            .log_map()
            .0;
        tangent.copy_from(&d);
    }

    //self is linearization point
    fn retract(&mut self, delta: DVectorView<R>)
    where
        R: Real,
    {
        let t: Vector3<R> = VectorView3::<R>::from(&delta).clone_owned();
        self.origin = self.origin * SE2Tangent(t).exp_map();
    }

    fn dim(&self) -> usize {
        3
    }

    // fn retract_local_jacobian(&self, _linearization_point: &Self, mut jacobian: DMatrixViewMut<R>) {
    //     jacobian.fill_with_identity();
    // }
}
impl<R> SE2<R>
where
    R: Real,
{
    pub fn new(x: f64, y: f64, theta: f64) -> Self {
        SE2 {
            origin: Isometry2::new(Vector2::new(convert(x), convert(y)), convert(theta)),
        }
    }
}
