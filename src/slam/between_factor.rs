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
pub struct BetweenFactor<LF = GaussianLoss, R = f64>
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
impl<LF, R> BetweenFactor<LF, R>
where
    R: Real,
    LF: LossFunction<R>,
{
    pub fn new(key0: Vkey, key1: Vkey, x: f64, y: f64, theta: f64, loss: Option<LF>) -> Self {
        let keys = vec![key0, key1];
        let mut jacobians = DMatrix::<R>::zeros(3, 3 * keys.len());
        jacobians.columns_mut(3, 3).fill_diagonal(R::one());
        BetweenFactor {
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
}
impl<LF, R> Factor<R> for BetweenFactor<LF, R>
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
        let v1: &SE2<R> = variables.get(self.keys()[1]).unwrap();

        let diff = v0.origin.inverse().multiply(&v1.origin);
        let diff = (self.origin.inverse().multiply(&diff)).log();
        {
            self.error.borrow_mut().copy_from(&diff.cast::<R>());
        }
        self.error.borrow()
    }

    fn jacobians<C>(&self, variables: &Variables<C, R>) -> JacobiansReturn<R>
    where
        C: VariablesContainer<R>,
    {
        {
            let v0: &SE2<R> = variables.get(self.keys()[0]).unwrap();
            let v1: &SE2<R> = variables.get(self.keys()[1]).unwrap();
            let hinv = -v0.origin.adj();
            let hcmp1 = v1.origin.inverse().adj();
            let j = (hcmp1 * hinv).cast::<R>();
            self.jacobians.borrow_mut().columns_mut(0, 3).copy_from(&j);
            {
                let v0: &SE2<R> = variables.get(self.keys()[0]).unwrap();
                let v1: &SE2<R> = variables.get(self.keys()[1]).unwrap();
                let hinv = -v0.origin.adj();
                let hcmp1 = v1.origin.inverse().adj();
                let j = (hcmp1 * hinv).cast::<R>();
                self.jacobians.borrow_mut().columns_mut(0, 3).copy_from(&j);
            }
            // {
            //     compute_numerical_jacobians(variables, self, &mut self.jacobians.borrow_mut());
            // }
        }
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
