use std::cell::RefCell;

use nalgebra::{DMatrix, DVector, DVectorView, RealField, SMatrix, Vector2, Vector3};
use num::Float;
use sophus_rs::lie::rotation2::{Isometry2, Rotation2};

use optigy::{
    core::variable::{TangentReturn, Variable},
    prelude::JacobianReturn,
};

#[derive(Debug, Clone)]
pub struct SE2<R = f64>
where
    R: RealField + Float,
{
    pub origin: Isometry2,
    local: RefCell<DVector<R>>,
    jac: RefCell<DMatrix<R>>,
}

impl<R> Variable<R> for SE2<R>
where
    R: RealField + Float,
{
    // value is linearization point
    fn local(&self, linearization_point: &Self) -> TangentReturn<R>
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
        *self.local.borrow_mut() = l;
        self.local.borrow()
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

    fn retract_local_jacobian(&self, _linearization_pointt: &Self) -> JacobianReturn<R> {
        let i = DMatrix::<R>::identity(3, 3);
        *self.jac.borrow_mut() = i;
        self.jac.borrow()
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
            local: RefCell::new(DVector::<R>::zeros(3)),
            jac: RefCell::new(DMatrix::<R>::identity(3, 3)),
        }
    }
}
