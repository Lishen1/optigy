use manif::LieGroupBase;
use nalgebra::{convert, DMatrixViewMut, DVectorViewMut, Dyn, Isometry2, Matrix3x6, Vector2, U3};

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
    pub keys: [Vkey; 2],
    pub origin: Isometry2<R>,
    pub loss: Option<LF>,
}
impl<LF, R> BetweenFactor<LF, R>
where
    R: Real,
    LF: LossFunction<R>,
{
    pub fn new(key0: Vkey, key1: Vkey, x: f64, y: f64, theta: f64, loss: Option<LF>) -> Self {
        let keys = [key0, key1];
        BetweenFactor {
            keys,
            origin: Isometry2::new(Vector2::new(convert(x), convert(y)), convert(theta)).inverse(),
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
    type JCols = Dyn;
    type JRows = Dyn;
    fn error<C>(&self, variables: &Variables<C, R>, mut error: DVectorViewMut<R>)
    where
        C: VariablesContainer<R>,
    {
        let v0: &SE2<R> = variables.get(self.keys[0]).unwrap();
        let v1: &SE2<R> = variables.get(self.keys[1]).unwrap();

        let diff = v0.origin.inverse() * &v1.origin;
        // let diff = (self.origin.inverse() * &diff).log_map().0;
        let diff = (self.origin * &diff).log_map().0;
        error.copy_from(&diff);
    }

    // fn jacobian<C>(&self, variables: &Variables<C, R>, mut jacobian: DMatrixViewMut<R>)
    // where
    //     C: VariablesContainer<R>,
    // {
    //     let v0: &SE2<R> = variables.get(self.keys[0]).unwrap();
    //     let v1: &SE2<R> = variables.get(self.keys[1]).unwrap();
    //     // let hinv = -v0.origin.adj();
    //     // let hcmp1 = v1.origin.inverse().adj();
    //     // let j = hcmp1 * hinv;
    //     let hinv = v0.origin;
    //     let hcmp1 = v1.origin.inverse();
    //     let mut j = Matrix3x6::<R>::zeros();
    //     j.columns_generic_mut(0, U3)
    //         .copy_from(&-(hcmp1 * hinv).adj());
    //     j.columns_generic_mut(3, U3).fill_with_identity();
    //     jacobian.copy_from(&j);
    // }

    fn jacobian_error<C>(
        &self,
        variables: &Variables<C, R>,
        mut jacobian: DMatrixViewMut<R>,
        mut error: DVectorViewMut<R>,
    ) where
        C: VariablesContainer<R>,
    {
        let v0: &SE2<R> = variables.get(self.keys[0]).unwrap();
        let v1: &SE2<R> = variables.get(self.keys[1]).unwrap();

        let diff = v0.origin.inverse() * &v1.origin;

        let mut j = Matrix3x6::<R>::identity();
        j.columns_generic_mut(0, U3)
            .copy_from(&-diff.inverse().adj());
        j.columns_generic_mut(3, U3).fill_with_identity();
        jacobian.copy_from(&j);
        error.copy_from(&(self.origin * &diff).log_map().0);
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
