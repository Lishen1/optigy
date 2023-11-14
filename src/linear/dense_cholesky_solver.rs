use std::marker::PhantomData;

use nalgebra::{DMatrix, DVector, RealField};
use num::Float;

use super::linear_solver::{DenseLinearSolver, LinearSolverStatus};

pub struct DenseCholeskySolver<R>
where
    R: RealField + Float,
{
    __marker: PhantomData<R>,
}
impl<R> DenseLinearSolver<R> for DenseCholeskySolver<R>
where
    R: RealField + Float,
{
    #[allow(non_snake_case)]
    fn solve(&self, A: &DMatrix<R>, b: &DVector<R>, x: &mut DVector<R>) -> LinearSolverStatus {
        match A.clone().cholesky() {
            Some(llt) => {
                x.copy_from(&llt.solve(b));
                LinearSolverStatus::Success
            }
            None => LinearSolverStatus::RankDeficiency,
        }
    }

    fn is_normal(&self) -> bool {
        true
    }

    fn is_normal_lower(&self) -> bool {
        true
    }
}
