use manif::LieGroupBase;
use nalgebra::{convert, DMatrixViewMut, DVectorViewMut, Dyn, Isometry2, Vector2};

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
    pub origin: Isometry2<R>,
    pub loss: Option<LF>,
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
            origin: Isometry2::new(Vector2::new(convert(x), convert(y)), convert(theta)),
            loss,
        }
    }
    pub fn from_se2(key: Vkey, origin: Isometry2<R>, loss: Option<LF>) -> Self {
        let keys = vec![key];
        PriorFactor { keys, origin, loss }
    }
}
impl<LF, R> Factor<R> for PriorFactor<LF, R>
where
    R: Real,
    LF: LossFunction<R>,
{
    type L = LF;
    type JCols = Dyn;
    type JRows = Dyn;
    fn jacobian_shape(&self) -> (Self::JRows, Self::JCols) {
        (Dyn(3), Dyn(3))
    }
    fn error<C>(&self, variables: &Variables<C, R>, mut error: DVectorViewMut<R>)
    where
        C: VariablesContainer<R>,
    {
        let v0: &SE2<R> = variables.get(self.keys()[0]).unwrap();
        let diff = (self.origin.inverse() * &v0.origin).log_map().0;
        error.copy_from(&diff);
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
