use std::marker::PhantomData;

use nalgebra::{
    DMatrixViewMut, DVector, DVectorView, DVectorViewMut, RealField, SMatrix, Vector2, Vector3,
};
use num::Float;
use sophus_rs::lie::rotation2::{Isometry2, Rotation2};

use optigy::core::variable::Variable;

#[derive(Debug, Clone)]
pub struct SE2<R = f64>
where
    R: RealField + Float,
{
    pub origin: Isometry2,
    __marker: PhantomData<R>,
}

impl<R> Variable<R> for SE2<R>
where
    R: RealField + Float,
{
    // value is linearization point
    fn local(&self, linearization_point: &Self, mut tangent: DVectorViewMut<R>)
    where
        R: RealField,
    {
        // let d = (self.origin.inverse().multiply(&value.origin)).log();
        let d = (linearization_point.origin.inverse().multiply(&self.origin)).log();
        // let translation = d.translation;
        // let subgroup_params = d.rotation;
        // let subgroup_tangent = subgroup_params.log();
        // let d = self.pose.clone() - value.pose.clone();
        let l = DVector::<R>::from_column_slice(d.cast::<R>().as_slice());
        tangent.copy_from(&l);
    }

    //self is linearization point
    fn retract(&mut self, delta: DVectorView<R>)
    where
        R: RealField + Float,
    {
        self.origin = self.origin.clone().multiply(&Isometry2::exp(&Vector3::new(
            delta[0].to_f64().unwrap(),
            delta[1].to_f64().unwrap(),
            delta[2].to_f64().unwrap(),
        )));
    }

    fn dim(&self) -> usize {
        3
    }

    fn retract_local_jacobian(&self, _linearization_point: &Self, mut jacobian: DMatrixViewMut<R>) {
        jacobian.fill_with_identity();
    }
}
impl<R> SE2<R>
where
    R: RealField + Float,
{
    pub fn new(x: f64, y: f64, theta: f64) -> Self {
        SE2 {
            origin: Isometry2::from_t_and_subgroup(
                &Vector2::new(x, y),
                &Rotation2::exp(&SMatrix::<f64, 1, 1>::from_column_slice(
                    vec![theta].as_slice(),
                )),
            ),
            __marker: PhantomData,
        }
    }
}
