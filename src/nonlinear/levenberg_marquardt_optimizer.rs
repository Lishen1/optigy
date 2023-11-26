use nalgebra::DVector;
use nalgebra_sparse::CscMatrix;

use crate::{
    core::{
        factors::Factors, factors_container::FactorsContainer, variable_ordering::VariableOrdering,
        variables::Variables, variables_container::VariablesContainer, Real,
    },
    linear::{
        linear_solver::{LinearSolverStatus, SparseLinearSolver},
        sparse_cholesky_solver::SparseCholeskySolver,
    },
    prelude::NonlinearOptimizerVerbosityLevel,
};

use super::nonlinear_optimizer::{
    IterationData, LinSysWrapper, NonlinearOptimizationError, NonlinearOptimizerParams, OptIterate,
    OptimizerBaseParams,
};
#[derive(Debug, Clone)]
pub struct LevenbergMarquardtOptimizerParams {
    // initial lambda
    pub lambda_init: f64,
    // initial multiply factor to increase lambda
    pub lambda_increase_factor_init: f64,
    // multiply factor to increase lambda multiply factor
    pub lambda_increase_factor_update: f64,
    // minimal lambda decrease factor
    pub lambda_decrease_factor_min: f64,
    // minimal lambda
    pub lambda_min: f64,
    // max lambda
    pub lambda_max: f64,
    // minimal gain ratio (quality factor) to accept a step
    pub gain_ratio_thresh: f64,
    // if true use lambda * diag(A'A) for dumping,
    // if false use lambda * max(diag(A'A)) * I
    pub diagonal_damping: bool,
    //base params
    pub base: NonlinearOptimizerParams,
}
impl Default for LevenbergMarquardtOptimizerParams {
    fn default() -> Self {
        LevenbergMarquardtOptimizerParams {
            lambda_init: 1e-5,
            lambda_increase_factor_init: 2.0,
            lambda_increase_factor_update: 2.0,
            lambda_decrease_factor_min: 1.0 / 3.0,
            lambda_min: 1e-20,
            lambda_max: 1e10,
            gain_ratio_thresh: 1e-3,
            diagonal_damping: true,
            base: NonlinearOptimizerParams::default(),
        }
    }
}
impl OptimizerBaseParams for LevenbergMarquardtOptimizerParams {
    fn base(&self) -> &NonlinearOptimizerParams {
        &self.base
    }
}

#[derive(Default)]
pub struct LevenbergMarquardtOptimizer<R = f64>
where
    R: Real + Default,
{
    /// linear solver
    pub linear_solver: SparseCholeskySolver<R>,
    pub params: LevenbergMarquardtOptimizerParams,
    try_lambda_inited: bool,
    lambda: f64,
    lambda_increase_factor: f64,
    gain_ratio: f64,
    last_lambda: f64,
    last_lambda_sqrt: f64,
    linear_solver_inited: bool,
}
impl<R> LevenbergMarquardtOptimizer<R>
where
    R: Real + Default,
{
    pub fn with_params(params: LevenbergMarquardtOptimizerParams) -> Self {
        let mut opt = LevenbergMarquardtOptimizer {
            linear_solver: SparseCholeskySolver::default(),
            params: params.clone(),
            try_lambda_inited: false,
            lambda: params.lambda_init,
            lambda_increase_factor: params.lambda_increase_factor_init,
            gain_ratio: 0.0,
            last_lambda: 0.0,
            last_lambda_sqrt: 0.0,
            linear_solver_inited: false,
        };
        opt.reset();
        opt
    }
    fn increase_lambda(&mut self) {
        self.lambda *= self.lambda_increase_factor;
        self.lambda_increase_factor *= self.params.lambda_increase_factor_update;
    }
    fn decrease_lambda(&mut self) {
        self.lambda *= self
            .params
            .lambda_decrease_factor_min
            .max(1.0 - (2.0 * self.gain_ratio - 1.0).powi(3));
        self.lambda = self.params.lambda_min.max(self.lambda);
        self.lambda_increase_factor = self.params.lambda_increase_factor_init;
    }

    #[allow(non_snake_case)]
    fn dump_linear_system(
        &mut self,
        A: &mut CscMatrix<R>,
        _b: &mut DVector<R>,
        hessian_diag: &DVector<R>,
        hessian_diag_max: f64,
        _hessian_diag_sqrt: &DVector<R>,
        _hessian_diag_max_sqrt: f64,
    ) {
        if self.linear_solver.is_normal() {
            // hessian system
            if !self.try_lambda_inited {
                if self.params.diagonal_damping {
                    update_dumping_hessian_diag(A, hessian_diag, self.lambda, 0.0);
                } else {
                    update_dumping_hessian(A, self.lambda * hessian_diag_max, 0.0);
                }
            } else if self.params.diagonal_damping {
                update_dumping_hessian_diag(A, hessian_diag, self.lambda, self.last_lambda);
            } else {
                update_dumping_hessian(
                    A,
                    self.lambda * hessian_diag_max,
                    self.last_lambda * hessian_diag_max,
                );
            }
        } else {
            todo!()
            // // jacobian system
            // double lambda_sqrt = std::sqrt(lambda_);

            // if (!try_lambda_inited_) {
            //   // jacobian not resize yet
            //   if (params_.diagonal_damping) {
            //     internal::allocateDumpingJacobianDiag(A, b, j_sparsity_cache_,
            //                                           lambda_sqrt, hessian_diag_sqrt);
            //   } else {
            //     internal::allocateDumpingJacobian(A, b, j_sparsity_cache_,
            //                                       lambda_sqrt * hessian_diag_max_sqrt);
            //   }
            // } else {
            //   // jacobian already resize
            //   if (params_.diagonal_damping) {
            //     internal::updateDumpingJacobianDiag(A, hessian_diag_sqrt, lambda_sqrt,
            //                                         last_lambda_sqrt_);
            //   } else {
            //     internal::updateDumpingJacobian(
            //         A, lambda_sqrt * hessian_diag_max_sqrt,
            //         last_lambda_sqrt_ * hessian_diag_max_sqrt);
            //   }
            // }
            // last_lambda_sqrt_ = lambda_sqrt;
        }

        // update last lambda to current
        self.try_lambda_inited = true;
        self.last_lambda = self.lambda;
    }

    #[allow(non_snake_case)]
    #[allow(clippy::too_many_arguments)]
    fn try_lambda<FC, VC>(
        &mut self,
        lin_sys: LinSysWrapper<'_, R>,
        g: &DVector<R>,
        hessian_diag: &DVector<R>,
        hessian_diag_max: f64,
        hessian_diag_sqrt: &DVector<R>,
        hessian_diag_max_sqrt: f64,
        factors: &Factors<FC, R>,
        variables: &mut Variables<VC, R>,
        variable_ordering: &VariableOrdering,
        values_curr_err: f64,
    ) -> Result<IterationData, NonlinearOptimizationError>
    where
        FC: FactorsContainer<R>,
        VC: VariablesContainer<R>,
    {
        let mut A = lin_sys.A.clone();
        let mut b = lin_sys.b.clone();
        // dump linear system
        self.dump_linear_system(
            &mut A,
            &mut b,
            hessian_diag,
            hessian_diag_max,
            hessian_diag_sqrt,
            hessian_diag_max_sqrt,
        );

        // solve dumped linear system
        // init solver is not yet
        if !self.linear_solver_inited {
            self.linear_solver.initialize(&A);
            self.linear_solver_inited = true;
        }

        // solve
        let mut dx_lm: DVector<R> = DVector::zeros(variables.dim());
        let linear_solver_status = self.linear_solver.solve(&A, &b, &mut dx_lm);
        dx_lm.neg_mut(); // since Hx=-b

        match linear_solver_status {
            LinearSolverStatus::Success => {}
            LinearSolverStatus::RankDeficiency => {
                if self.params.base.verbosity_level
                    >= NonlinearOptimizerVerbosityLevel::Subiteration
                {
                    println!("dumped linear system has rank deficiency");
                }
                //TODO: impl error from
                return Err(NonlinearOptimizationError::RankDeficiency);
            }
            LinearSolverStatus::Invalid => {
                //TODO: impl error from
                return Err(NonlinearOptimizationError::Invalid);
            }
        }
        // calculate gain ratio

        // nonlinear error improvement
        let variables_to_update = variables.retracted(dx_lm.as_view(), variable_ordering);

        let values_update_err = 0.5 * factors.error_squared_norm(&variables_to_update);
        let nonlinear_err_update = values_curr_err - values_update_err;

        // linear error improvement
        // see imm3215 p.25, just notice here g = -g in the book(fixed)
        let linear_err_update = -if self.params.diagonal_damping {
            0.5 * dx_lm
                .dot(
                    &(hessian_diag
                        .component_mul(&dx_lm)
                        .scale(R::from_f64(self.lambda).unwrap())
                        + g),
                )
                .to_f64()
                .unwrap()
        } else {
            0.5 * dx_lm
                .dot(&(dx_lm.scale(R::from_f64(hessian_diag_max * self.lambda).unwrap()) + g))
                .to_f64()
                .unwrap()
        };

        self.gain_ratio = nonlinear_err_update / linear_err_update;

        if self.params.base.verbosity_level >= NonlinearOptimizerVerbosityLevel::Subiteration {
            println!("gain ratio: {}", self.gain_ratio);
        }

        if self.gain_ratio > self.params.gain_ratio_thresh {
            // try is success and update values
            *variables = variables_to_update;
            Ok(IterationData::new(true, values_update_err))
        } else {
            Err(NonlinearOptimizationError::ErrorIncrease)
        }
    }
}

#[allow(non_snake_case)]
fn update_dumping_hessian<R>(H: &mut CscMatrix<R>, diag: f64, diag_last: f64)
where
    R: Real + Default,
{
    for i in 0..H.ncols() {
        match H.get_entry_mut(i, i).unwrap() {
            nalgebra_sparse::SparseEntryMut::NonZero(v) => {
                *v += R::from_f64(diag - diag_last).unwrap()
            }
            nalgebra_sparse::SparseEntryMut::Zero => {}
        }
    }
}

#[allow(non_snake_case)]
fn update_dumping_hessian_diag<R>(
    H: &mut CscMatrix<R>,
    diags: &DVector<R>,
    lambda: f64,
    lambda_last: f64,
) where
    R: Real + Default,
{
    for i in 0..H.ncols() {
        match H.get_entry_mut(i, i).unwrap() {
            nalgebra_sparse::SparseEntryMut::NonZero(v) => {
                *v += R::from_f64(lambda - lambda_last).unwrap() * diags[i]
            }
            nalgebra_sparse::SparseEntryMut::Zero => {}
        }
    }
}
impl<R> OptIterate<R> for LevenbergMarquardtOptimizer<R>
where
    R: Real + Default,
{
    type S = SparseCholeskySolver<R>;
    #[allow(non_snake_case)]
    fn iterate<FC, VC>(
        &mut self,
        factors: &Factors<FC, R>,
        variables: &mut Variables<VC, R>,
        variable_ordering: &VariableOrdering,
        lin_sys: LinSysWrapper<'_, R>,
        variables_curr_err: f64,
    ) -> Result<IterationData, NonlinearOptimizationError>
    where
        FC: FactorsContainer<R>,
        VC: VariablesContainer<R>,
    {
        // get hessian diagonal once iter
        let hessian_diag = if self.linear_solver.is_normal() {
            DVector::<R>::from_row_slice(lin_sys.A.diagonal_as_csc().values())
        } else {
            todo!()
            // hessian_diag = internal::hessianDiagonal(A);
        };
        let mut hessian_diag_max = 0.0;
        let hessian_diag_max_sqrt = 0.0;

        let hessian_diag_sqrt = DVector::<R>::zeros(0);
        if !self.linear_solver.is_normal() {
            // hessian_diag_sqrt = hessian_diag.as_view().iter().map(|v: R| Float::sqrt(v));
            todo!();
            // if (!params_.diagonal_damping) {
            //     hessian_diag_max_sqrt = hessian_diag_sqrt.maxCoeff();
            // }
        } else if !self.params.diagonal_damping {
            hessian_diag_max = hessian_diag.max().to_f64().unwrap();
        }

        // calc Atb (defined by g here) if jacobian
        let g = if self.linear_solver.is_normal() {
            lin_sys.b.to_owned()
        } else {
            // TODO: improve memory efficiency
            lin_sys.A.transpose() * lin_sys.b
        };

        // current value error
        // let variables_curr_err = factors.error_squared_norm(variables);
        // try different lambda, until find a lambda to decrese error (return
        // SUCCESS),
        // or reach max lambda which still cannot decrese error (return
        // ERROR_INCREASE)
        self.try_lambda_inited = false;

        while self.lambda < self.params.lambda_max {
            if self.params.base.verbosity_level >= NonlinearOptimizerVerbosityLevel::Subiteration {
                println!("lambda: {}", self.lambda);
            }

            // try current lambda value
            let try_lambda_result = self.try_lambda(
                lin_sys.clone(),
                &g,
                &hessian_diag,
                hessian_diag_max,
                &hessian_diag_sqrt,
                hessian_diag_max_sqrt,
                factors,
                variables,
                variable_ordering,
                variables_curr_err,
            );

            match try_lambda_result {
                Ok(data) => {
                    // SUCCESS: decrease error, decrease lambda and return success
                    self.decrease_lambda();
                    return Ok(data);
                }

                Err(err) => {
                    if err == NonlinearOptimizationError::RankDeficiency
                        || err == NonlinearOptimizationError::ErrorIncrease
                    {
                        // RANK_DEFICIENCY and ERROR_INCREASE, incease lambda and try again
                        self.increase_lambda();
                    } else {
                        // INVALID: internal error
                        println!("Warning: linear solver returns invalid state");
                        return Err(NonlinearOptimizationError::Invalid);
                    }
                }
            }
        }

        // cannot decrease error with max lambda
        println!("Warning: LM cannot decrease error with max lambda");
        Err(NonlinearOptimizationError::ErrorIncrease)
    }

    fn linear_solver(&mut self) -> &mut Self::S {
        &mut self.linear_solver
    }
    fn base_params(&self) -> &NonlinearOptimizerParams {
        &self.params.base
    }

    fn reset(&mut self) {
        self.try_lambda_inited = false;
        self.lambda = self.params.lambda_init;
        self.lambda_increase_factor = self.params.lambda_increase_factor_init;
        self.gain_ratio = 0.0;
        self.last_lambda = 0.0;
        self.last_lambda_sqrt = 0.0;
        self.linear_solver_inited = false;
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
        let mut opt_iter = GaussNewtonOptimizer::default();
        let sparsity = construct_hessian_sparsity(
            &factors,
            &variables,
            &variable_ordering,
            HessianTriangle::Upper,
        );
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
        opt_iter.linear_solver.initialize(&A);
        let opt_res = opt_iter.iterate(
            &factors,
            &mut variables,
            &variable_ordering,
            LinSysWrapper::new(&A, &b),
            0.0,
        );
        assert!(opt_res.is_ok());
    }
}
