use nalgebra::{DVector, RealField};

use crate::{
    core::{
        factors::Factors, factors_container::FactorsContainer, variable_ordering::VariableOrdering,
        variables::Variables, variables_container::VariablesContainer, Real,
    },
    linear::{
        linear_solver::{LinearSolverStatus, SparseLinearSolver},
        sparse_cholesky_solver::SparseCholeskySolver,
    },
};

use super::nonlinear_optimizer::{
    IterationData, LinSysWrapper, NonlinearOptimizationError, NonlinearOptimizerParams, OptIterate,
    OptimizerBaseParams,
};
#[derive(Debug, Default)]
pub struct GaussNewtonOptimizerParams {
    pub base: NonlinearOptimizerParams,
}
impl OptimizerBaseParams for GaussNewtonOptimizerParams {
    fn base(&self) -> &NonlinearOptimizerParams {
        &self.base
    }
}
#[derive(Default)]
pub struct GaussNewtonOptimizer<R = f64>
where
    R: Real + Default,
{
    /// linear solver
    pub linear_solver: SparseCholeskySolver<R>,
    pub params: GaussNewtonOptimizerParams,
}
impl<R> GaussNewtonOptimizer<R>
where
    R: Real + Default,
{
    pub fn with_params(params: GaussNewtonOptimizerParams) -> Self {
        GaussNewtonOptimizer {
            linear_solver: SparseCholeskySolver::<R>::default(),
            params,
        }
    }
}
impl<R> OptIterate<R> for GaussNewtonOptimizer<R>
where
    R: Real + Default,
{
    type S = SparseCholeskySolver<R>;
    #[allow(non_snake_case)]
    fn iterate<FC, VC>(
        &mut self,
        _factors: &Factors<FC, R>,
        variables: &mut Variables<VC, R>,
        variable_ordering: &VariableOrdering,
        lin_sys: LinSysWrapper<'_, R>,
        _variables_curr_err: f64,
    ) -> Result<IterationData, NonlinearOptimizationError>
    where
        R: RealField,
        VC: VariablesContainer<R>,
        FC: FactorsContainer<R>,
    {
        let mut dx: DVector<R> = DVector::zeros(variables.dim());
        //WARN sparse cholesky solver not working for lower triangular matrices
        let linear_solver_status = self.linear_solver.solve(lin_sys.A, lin_sys.b, &mut dx);
        dx.neg_mut(); // since Hx=-b

        // let mut A = DMatrix::<R>::from(lin_sys.A);
        // A.fill_upper_triangle_with_lower_triangle();
        // dx = A.clone().cholesky().unwrap().solve(lin_sys.b);
        if linear_solver_status == LinearSolverStatus::Success {
            variables.retract(dx.as_view(), variable_ordering);
            Ok(IterationData::default())
        } else if linear_solver_status == LinearSolverStatus::RankDeficiency {
            println!("Warning: linear system has rank deficiency");
            Err(NonlinearOptimizationError::RankDeficiency)
        } else {
            println!("Warning: linear solver returns invalid state");
            Err(NonlinearOptimizationError::Invalid)
        }
    }

    fn linear_solver(&mut self) -> &mut Self::S {
        &mut self.linear_solver
    }

    fn base_params(&self) -> &NonlinearOptimizerParams {
        &self.params.base
    }
}
#[cfg(test)]
mod tests {
    use nalgebra::DVector;
    use nalgebra_sparse::{pattern::SparsityPattern, CscMatrix};

    use crate::{
        core::{
            factor::tests::{FactorA, FactorB},
            factors::Factors,
            factors_container::FactorsContainer,
            key::Vkey,
            variable::tests::{VariableA, VariableB},
            variables::Variables,
            variables_container::VariablesContainer,
        },
        linear::linear_solver::SparseLinearSolver,
        nonlinear::{
            gauss_newton_optimizer::GaussNewtonOptimizer,
            linearization::linearization_hessian,
            nonlinear_optimizer::{LinSysWrapper, OptIterate},
            sparsity_pattern::{construct_hessian_sparsity, HessianTriangle},
        },
    };

    #[test]
    #[allow(non_snake_case)]
    fn iterate() {
        type Real = f64;
        let container = ().and_variable::<VariableA<Real>>().and_variable::<VariableB<Real>>();
        let mut variables = Variables::new(container);
        variables.add(Vkey(0), VariableA::<Real>::new(0.0));
        variables.add(Vkey(1), VariableB::<Real>::new(0.0));
        variables.add(Vkey(2), VariableB::<Real>::new(0.0));

        let container = ().and_factor::<FactorA<Real>>().and_factor::<FactorB<Real>>();
        let mut factors = Factors::new(container);
        factors.add(FactorA::new(1.0, None, Vkey(0), Vkey(1)));
        factors.add(FactorB::new(2.0, None, Vkey(1), Vkey(2)));
        let variable_ordering = variables.default_variable_ordering();
        let mut optimizer = GaussNewtonOptimizer::<Real>::default();
        let tri = HessianTriangle::Upper;
        let sparsity = construct_hessian_sparsity(&factors, &variables, &variable_ordering, tri);
        let A_rows: usize = sparsity.base.A_cols;
        let mut A_values = Vec::<Real>::new();
        A_values.resize(sparsity.total_nnz_AtA_cols, 0.0);
        let mut b: DVector<Real> = DVector::zeros(A_rows);
        linearization_hessian(&factors, &variables, &sparsity, &mut A_values, &mut b);
        let csc_pattern = SparsityPattern::try_from_offsets_and_indices(
            sparsity.base.A_cols,
            sparsity.base.A_cols,
            sparsity.outer_index.clone(),
            sparsity.inner_index.clone(),
        )
        .unwrap();
        let A = CscMatrix::try_from_pattern_and_values(csc_pattern, A_values.clone())
            .expect("CSC data must conform to format specifications");
        optimizer.linear_solver.initialize(&A);
        let opt_res = optimizer.iterate(
            &factors,
            &mut variables,
            &variable_ordering,
            LinSysWrapper::new(&A, &b),
            0.0,
        );
        assert!(opt_res.is_ok());
    }
}
