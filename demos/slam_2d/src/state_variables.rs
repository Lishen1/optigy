use std::cell::RefCell;

use nalgebra::{DMatrix, DVector, DVectorView, RealField, Vector2};
use optigy::{
    core::variable::TangentReturn,
    prelude::{JacobianReturn, Variable},
};

#[derive(Debug, Clone)]
pub struct E2<R = f64>
where
    R: RealField,
{
    pub val: Vector2<R>,
    local: RefCell<DVector<R>>,
    jac: RefCell<DMatrix<R>>,
}

impl<R> Variable<R> for E2<R>
where
    R: RealField,
{
    fn local(&self, linearization_point: &Self) -> TangentReturn<R>
    where
        R: RealField,
    {
        let d = self.val.clone() - linearization_point.val.clone();
        let l = DVector::<R>::from_column_slice(d.as_slice());
        *self.local.borrow_mut() = l;
        self.local.borrow()
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

    fn retract_local_jacobian(&self, _linearization_point: &Self) -> JacobianReturn<R> {
        let i = DMatrix::<R>::identity(2, 2);
        *self.jac.borrow_mut() = i;
        self.jac.borrow()
    }
}
impl<R> E2<R>
where
    R: RealField,
{
    pub fn new(x: f64, y: f64) -> Self {
        E2 {
            val: Vector2::new(R::from_f64(x).unwrap(), R::from_f64(y).unwrap()),
            local: RefCell::new(DVector::<R>::zeros(3)),
            jac: RefCell::new(DMatrix::<R>::identity(2, 2)),
        }
    }
}
