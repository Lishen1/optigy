use nalgebra::{
    matrix, vector, ComplexField, DMatrixView, DMatrixViewMut, DVectorViewMut, Matrix2, Vector2,
};
use optigy::prelude::{Factor, GaussianLoss, Real, Variables, VariablesContainer, Vkey};

use crate::E2;
use slam_common::se2::SE2;

#[derive(Clone)]
pub struct VisionFactor<R = f64>
where
    R: Real,
{
    keys: [Vkey; 2],
    ray: Vector2<R>,
    loss: GaussianLoss<R>,
}
impl<R> VisionFactor<R>
where
    R: Real,
{
    pub const LANDMARK_KEY: usize = 0;
    pub const POSE_KEY: usize = 1;
    pub fn new(landmark_id: Vkey, pose_id: Vkey, ray: Vector2<R>, cov: DMatrixView<R>) -> Self {
        VisionFactor {
            keys: [landmark_id, pose_id],
            ray,
            loss: GaussianLoss::<R>::covariance(cov.as_view()),
        }
    }
    pub fn ray(&self) -> &Vector2<R> {
        &self.ray
    }
}
#[allow(non_snake_case)]
impl<R> Factor<R> for VisionFactor<R>
where
    R: Real,
{
    type L = GaussianLoss<R>;

    fn error<C>(&self, variables: &Variables<C, R>, mut error: DVectorViewMut<R>)
    where
        C: VariablesContainer<R>,
    {
        let landmark_v: &E2<R> = variables
            .get(self.keys[VisionFactor::<R>::LANDMARK_KEY])
            .unwrap();
        let pose_v: &SE2<R> = variables
            .get(self.keys[VisionFactor::<R>::POSE_KEY])
            .unwrap();
        // let R_inv = pose_v.origin.inverse().matrix();
        // let R_inv = R_inv.fixed_view::<2, 2>(0, 0).to_owned();
        let th = pose_v.origin.rotation.angle();
        let R_inv =
            matrix![ComplexField::cos(th), -ComplexField::sin(th); ComplexField::sin(th), ComplexField::cos(th) ].transpose();
        let p = pose_v.origin.translation.vector;
        let l = landmark_v.val;
        // let l0 = pose_v.origin.inverse().transform(&landmark_v.val);
        let l0 = R_inv * (l - vector![p[0], p[1]]);

        let r = l0.normalize();

        error.copy_from(&(r - self.ray));

        // println!("err comp {}", self.error.borrow().norm());
    }

    fn jacobian<C>(&self, variables: &Variables<C, R>, mut jacobian: DMatrixViewMut<R>)
    where
        C: VariablesContainer<R>,
    {
        let landmark_v: &E2<R> = variables.get(self.keys[0]).unwrap();
        let pose_v: &SE2<R> = variables.get(self.keys[1]).unwrap();
        // let R_inv = pose_v.origin.inverse().matrix();
        // let R_inv = R_inv.fixed_view::<2, 2>(0, 0).to_owned();

        let th = pose_v.origin.rotation.angle();
        let R_inv =
            matrix![ComplexField::cos(th), -ComplexField::sin(th); ComplexField::sin(th), ComplexField::cos(th) ].transpose();

        let l = landmark_v.val;
        let p = pose_v.origin.translation.vector;
        let l0 = R_inv * (l - vector![p[0], p[1]]);
        // let l0 = R_inv * l - R_inv * p;

        let r = l0.normalize();
        let J_norm = (Matrix2::identity() - r * r.transpose()) / l0.norm();
        jacobian.columns_mut(0, 2).copy_from(&(J_norm * R_inv));
        jacobian.columns_mut(2, 2).copy_from(&(-J_norm));
        let th = pose_v.origin.rotation.angle();
        let x = landmark_v.val[0] - pose_v.origin.translation.x;
        let y = landmark_v.val[1] - pose_v.origin.translation.y;

        jacobian.columns_mut(4, 1).copy_from(
            &(J_norm
                * Vector2::new(
                    -x * ComplexField::sin(th) + y * ComplexField::cos(th),
                    -x * ComplexField::cos(th) - y * ComplexField::sin(th),
                )),
        );
        // println!("an J: {}", self.jacobians.borrow());
        // compute_numerical_jacobians(variables, self, &mut self.jacobians.borrow_mut());
        // println!("num J: {}", self.jacobians.borrow());
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
