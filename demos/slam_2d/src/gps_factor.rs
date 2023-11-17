use nalgebra::{matrix, vector, DMatrixViewMut, DVectorViewMut, Vector2};
use num::Float;
use optigy::prelude::{DiagonalLoss, Factor, Real, Variables, VariablesContainer, Vkey};
use slam_common::se2::SE2;
#[derive(Clone)]
pub struct GPSPositionFactor<R = f64>
where
    R: Real,
{
    pub keys: Vec<Vkey>,
    pub pose: Vector2<R>,
    pub loss: DiagonalLoss<R>,
}
impl<R> GPSPositionFactor<R>
where
    R: Real,
{
    pub fn new(key: Vkey, pose: Vector2<R>, sigmas: Vector2<R>) -> Self {
        let keys = vec![key];
        GPSPositionFactor {
            keys,
            pose,
            loss: DiagonalLoss::sigmas(&sigmas.as_view()),
        }
    }
}
#[allow(non_snake_case)]
impl<R> Factor<R> for GPSPositionFactor<R>
where
    R: Real,
{
    type L = DiagonalLoss<R>;
    fn error<C>(&self, variables: &Variables<C, R>, mut error: DVectorViewMut<R>)
    where
        C: VariablesContainer<R>,
    {
        let v0: &SE2<R> = variables.get(self.keys()[0]).unwrap();
        let pose = v0.origin.params();
        let pose = vector![pose[0], pose[1]];
        let d = self.pose - pose.cast::<R>();
        error.copy_from(&d);
    }

    fn jacobian<C>(&self, variables: &Variables<C, R>, mut jacobian: DMatrixViewMut<R>)
    where
        C: VariablesContainer<R>,
    {
        // compute_numerical_jacobians(variables, self, &mut self.jacobians.borrow_mut());
        // println!("J {}", self.jacobians.borrow());
        jacobian.fill_with_identity();

        let v0: &SE2<R> = variables.get(self.keys()[0]).unwrap();
        let th = -R::from_f64(v0.origin.log()[2]).unwrap();
        let R_inv =
            -matrix![Float::cos(th), -Float::sin(th); Float::sin(th), Float::cos(th) ].transpose();
        jacobian.view_mut((0, 0), (2, 2)).copy_from(&R_inv);
        // println!("R {}", R_inv);
    }

    fn dim(&self) -> usize {
        2
    }

    fn keys(&self) -> &[Vkey] {
        &self.keys
    }

    fn loss_function(&self) -> Option<&Self::L> {
        Some(&self.loss)
    }
}
