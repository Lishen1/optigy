use nalgebra::{matrix, vector, DMatrixView, DMatrixViewMut, DVectorViewMut, Matrix2, Vector2};
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
        let th = R::from_f64(pose_v.origin.log()[2]).unwrap();
        let R_inv = matrix![th.cos(), -th.sin(); th.sin(), th.cos() ].transpose();
        let p = pose_v.origin.params().fixed_rows::<2>(0);
        let l = landmark_v.val;
        // let l0 = pose_v.origin.inverse().transform(&landmark_v.val);
        let l0 = R_inv * (l - vector![R::from_f64(p[0]).unwrap(), R::from_f64(p[1]).unwrap()]);

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

        let th = R::from_f64(pose_v.origin.log()[2]).unwrap();
        let R_inv = matrix![th.cos(), -th.sin(); th.sin(), th.cos() ].transpose();

        let l = landmark_v.val;
        let p = pose_v.origin.params().fixed_rows::<2>(0);
        let l0 = R_inv * (l - vector![R::from_f64(p[0]).unwrap(), R::from_f64(p[1]).unwrap()]);
        // let l0 = R_inv * l - R_inv * p;

        let r = l0.normalize();
        let J_norm = (Matrix2::identity() - r * r.transpose()) / l0.norm();
        jacobian.columns_mut(0, 2).copy_from(&(J_norm * R_inv));
        jacobian.columns_mut(2, 2).copy_from(&(-J_norm));
        let th = R::from_f64(pose_v.origin.log()[2]).unwrap();
        let x = landmark_v.val[0] - R::from_f64(pose_v.origin.params()[0]).unwrap();
        let y = landmark_v.val[1] - R::from_f64(pose_v.origin.params()[1]).unwrap();

        jacobian.columns_mut(4, 1).copy_from(
            &(J_norm * Vector2::new(-x * th.sin() + y * th.cos(), -x * th.cos() - y * th.sin())),
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
