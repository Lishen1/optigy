use std::cell::RefCell;

use nalgebra::{DMatrix, DVector, SMatrix, Vector2};
use sophus_rs::lie::rotation2::{Isometry2, Rotation2};

use crate::core::{
    factor::{ErrorReturn, Factor, Jacobians, JacobiansReturn},
    key::Vkey,
    loss_function::{GaussianLoss, LossFunction},
    variables::Variables,
    variables_container::VariablesContainer,
    Real,
};

use super::se3::SE2;

#[derive(Clone)]
pub struct PriorFactor<LF = GaussianLoss, R = f64>
where
    R: Real,
    LF: LossFunction<R>,
{
    pub error: RefCell<DVector<R>>,
    pub jacobians: RefCell<Jacobians<R>>,
    pub keys: Vec<Vkey>,
    pub origin: Isometry2,
    pub loss: Option<LF>,
}
impl<LF, R> PriorFactor<LF, R>
where
    R: Real,
    LF: LossFunction<R>,
{
    pub fn new(key: Vkey, x: f64, y: f64, theta: f64, loss: Option<LF>) -> Self {
        let keys = vec![key];
        let jacobians = DMatrix::<R>::identity(3, 3 * keys.len());
        PriorFactor {
            error: RefCell::new(DVector::zeros(3)),
            jacobians: RefCell::new(jacobians),
            keys,
            origin: Isometry2::from_t_and_subgroup(
                &Vector2::new(x, y),
                &Rotation2::exp(&SMatrix::<f64, 1, 1>::from_column_slice(
                    vec![theta].as_slice(),
                )),
            ),
            loss,
        }
    }
    pub fn from_se2(key: Vkey, origin: Isometry2, loss: Option<LF>) -> Self {
        let keys = vec![key];
        let jacobians = DMatrix::<R>::identity(3, 3 * keys.len());
        PriorFactor {
            error: RefCell::new(DVector::zeros(3)),
            jacobians: RefCell::new(jacobians),
            keys,
            origin,
            loss,
        }
    }
}
impl<LF, R> Factor<R> for PriorFactor<LF, R>
where
    R: Real,
    LF: LossFunction<R>,
{
    type L = LF;
    fn error<C>(&self, variables: &Variables<C, R>) -> ErrorReturn<R>
    where
        C: VariablesContainer<R>,
    {
        let v0: &SE2<R> = variables.get(self.keys()[0]).unwrap();
        let diff = (self.origin.inverse().multiply(&v0.origin)).log();
        {
            self.error.borrow_mut().copy_from(&diff.cast::<R>());
        }
        self.error.borrow()
    }

    fn jacobians<C>(&self, _variables: &Variables<C, R>) -> JacobiansReturn<R>
    where
        C: VariablesContainer<R>,
    {
        //identity
        // {
        //     compute_numerical_jacobians(variables, self, &mut self.jacobians.borrow_mut());
        // }
        self.jacobians.borrow()
    }

    fn dim(&self) -> usize {
        3
    }

    fn keys(&self) -> &[Vkey] {
        &self.keys
    }

    fn loss_function(&self) -> Option<&Self::L> {
        self.loss.as_ref()
    }
}
