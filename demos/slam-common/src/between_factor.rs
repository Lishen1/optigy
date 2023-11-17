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
pub struct BetweenFactor<LF = GaussianLoss, R = f64>
where
    R: Real,
    LF: LossFunction<R>,
{
    pub keys: Vec<Vkey>,
    pub origin: Isometry2,
    pub loss: Option<LF>,
    __marker: PhantomData<R>,
}
impl<LF, R> BetweenFactor<LF, R>
where
    R: Real,
    LF: LossFunction<R>,
{
    pub fn new(key0: Vkey, key1: Vkey, x: f64, y: f64, theta: f64, loss: Option<LF>) -> Self {
        let keys = vec![key0, key1];
        BetweenFactor {
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
}
impl<LF, R> Factor<R> for BetweenFactor<LF, R>
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
        let v1: &SE2<R> = variables.get(self.keys()[1]).unwrap();

        let diff = v0.origin.inverse().multiply(&v1.origin);
        let diff = (self.origin.inverse().multiply(&diff)).log();
        error.copy_from(&diff.cast::<R>());
    }

    fn jacobian<C>(&self, variables: &Variables<C, R>, mut jacobian: DMatrixViewMut<R>)
    where
        C: VariablesContainer<R>,
    {
        jacobian.columns_mut(3, 3).fill_with_identity();
        let v0: &SE2<R> = variables.get(self.keys()[0]).unwrap();
        let v1: &SE2<R> = variables.get(self.keys()[1]).unwrap();
        let hinv = -v0.origin.adj();
        let hcmp1 = v1.origin.inverse().adj();
        let j = (hcmp1 * hinv).cast::<R>();
        jacobian.columns_mut(0, 3).copy_from(&j);
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
