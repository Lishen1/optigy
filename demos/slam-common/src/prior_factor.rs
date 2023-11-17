use std::marker::PhantomData;

use nalgebra::{DMatrixViewMut, DVectorViewMut, SMatrix, Vector2};
use sophus_rs::lie::rotation2::{Isometry2, Rotation2};

use optigy::core::{
    factor::Factor,
    key::Vkey,
    loss_function::{GaussianLoss, LossFunction},
    variables::Variables,
    variables_container::VariablesContainer,
    Real,
};

use super::se2::SE2;

#[derive(Clone)]
pub struct PriorFactor<LF = GaussianLoss, R = f64>
where
    R: Real,
    LF: LossFunction<R>,
{
    pub keys: Vec<Vkey>,
    pub origin: Isometry2,
    pub loss: Option<LF>,
    __marker: PhantomData<R>,
}
impl<LF, R> PriorFactor<LF, R>
where
    R: Real,
    LF: LossFunction<R>,
{
    pub fn new(key: Vkey, x: f64, y: f64, theta: f64, loss: Option<LF>) -> Self {
        let keys = vec![key];
        PriorFactor {
            keys,
            origin: Isometry2::from_t_and_subgroup(
                &Vector2::new(x, y),
                &Rotation2::exp(&SMatrix::<f64, 1, 1>::from_column_slice(
                    vec![theta].as_slice(),
                )),
            ),
            loss,
            __marker: PhantomData,
        }
    }
    pub fn from_se2(key: Vkey, origin: Isometry2, loss: Option<LF>) -> Self {
        let keys = vec![key];
        PriorFactor {
            keys,
            origin,
            loss,
            __marker: PhantomData,
        }
    }
}
impl<LF, R> Factor<R> for PriorFactor<LF, R>
where
    R: Real,
    LF: LossFunction<R>,
{
    type L = LF;
    fn error<C>(&self, variables: &Variables<C, R>, mut error: DVectorViewMut<R>)
    where
        C: VariablesContainer<R>,
    {
        let v0: &SE2<R> = variables.get(self.keys()[0]).unwrap();
        let diff = (self.origin.inverse().multiply(&v0.origin)).log();
        error.copy_from(&diff.cast::<R>());
    }

    fn jacobian<C>(&self, _variables: &Variables<C, R>, mut jacobian: DMatrixViewMut<R>)
    where
        C: VariablesContainer<R>,
    {
        {
            // compute_numerical_jacobians(variables, self, &mut jacobians);
        }
        jacobian.fill_with_identity();
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
