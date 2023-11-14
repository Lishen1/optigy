use std::{marker::PhantomData, time::Instant};

use nalgebra::{DVector, RealField};
use nalgebra_sparse::{pattern::SparsityPattern, CscMatrix};

use crate::{
    core::{
        factors::Factors, factors_container::FactorsContainer, variable_ordering::VariableOrdering,
        variables::Variables, variables_container::VariablesContainer, Real,
    },
    linear::linear_solver::SparseLinearSolver,
    nonlinear::sparsity_pattern::HessianTriangle,
    prelude::GaussNewtonOptimizer,
};

use super::{
    linearization::linearization_hessian,
    sparsity_pattern::{
        construct_hessian_sparsity, HessianSparsityPattern, JacobianSparsityPattern,
    },
};
/// return status of nonlinear optimization
#[derive(PartialEq, Eq, Debug)]
pub enum NonlinearOptimizationError {
    /// reach max iterations but not reach converge requirement
    MaxIteration,
    /// optimizer cannot decrease error and give up
    ErrorIncrease,
    /// linear system has rank deficiency
    RankDeficiency,
    /// something else is wrong with the optimization
    Invalid,
}
/// enum of linear solver types
#[derive(Debug, Clone)]
pub enum LinearSolverType {
    // Eigen Direct LDLt factorization
    Cholesky,
    // SuiteSparse CHOLMOD
    Cholmod,
    // SuiteSparse SPQR
    Qr,
    // Eigen Classical Conjugate Gradient Method
    Cg,
    // Eigen Conjugate Gradient Method without forming A'A
    Lscg,
    // cuSolverSP Cholesky factorization
    CudaCholesky,
    // Schur complement with reduced system solved by Eigen dense Cholesky
    SchurDenseCholesky,
}
// enum of nonlinear optimization verbosity level
#[derive(PartialOrd, Ord, PartialEq, Eq, Debug, Clone)]
pub enum NonlinearOptimizerVerbosityLevel {
    /// only print warning message to std::cerr when optimization does not success
    /// and terminated abnormally. Default verbosity level
    Warning,
    /// print per-iteration least square error sum to std::cout, also print
    /// profiling defails to std::cout after optimization is done, if miniSAM is
    /// compiled with internal profiling enabled
    Iteration,
    /// print more per-iteration detailed information to std::cout, e.g. trust
    /// regoin searching information
    Subiteration,
}
/// base class for nonlinear optimization settings
#[derive(Debug, Clone)]
pub struct NonlinearOptimizerParams {
    /// max number of iterations
    pub max_iterations: usize,
    /// relative error decrease threshold to stop
    pub min_rel_err_decrease: f64,
    /// absolute error decrease threshold to stop
    pub min_abs_err_decrease: f64,
    /// linear solver
    pub linear_solver_type: LinearSolverType,
    /// warning verbosity
    pub verbosity_level: NonlinearOptimizerVerbosityLevel,
}
impl Default for NonlinearOptimizerParams {
    fn default() -> Self {
        NonlinearOptimizerParams {
            max_iterations: 100,
            min_rel_err_decrease: 1e-5,
            min_abs_err_decrease: 1e-5,
            linear_solver_type: LinearSolverType::Cholesky,
            verbosity_level: NonlinearOptimizerVerbosityLevel::Warning,
        }
    }
}
pub trait OptimizerBaseParams {
    fn base(&self) -> &NonlinearOptimizerParams;
}
impl Default for NonlinearOptimizationError {
    fn default() -> Self {
        Self::Invalid
    }
}
#[allow(non_snake_case)]
#[derive(Clone)]
pub struct LinSysWrapper<'a, R>
where
    R: RealField,
{
    pub A: &'a CscMatrix<R>,
    pub b: &'a DVector<R>,
}
impl<'a, R> LinSysWrapper<'a, R>
where
    R: RealField,
{
    #[allow(non_snake_case)]
    pub fn new(A: &'a CscMatrix<R>, b: &'a DVector<R>) -> Self {
        LinSysWrapper { A, b }
    }
}
pub enum LinSpasityPattern {
    Jacobian(JacobianSparsityPattern),
    Hessian(HessianSparsityPattern),
}
impl Default for LinSpasityPattern {
    fn default() -> Self {
        LinSpasityPattern::Hessian(HessianSparsityPattern::default())
    }
}
pub struct IterationData {
    pub err_uptodate: bool,
    pub err_squared_norm: f64,
}
impl IterationData {
    pub fn new(err_uptodate: bool, err_squared_norm: f64) -> Self {
        IterationData {
            err_uptodate,
            err_squared_norm,
        }
    }
}
impl Default for IterationData {
    fn default() -> Self {
        IterationData {
            err_uptodate: false,
            err_squared_norm: 0.0,
        }
    }
}
pub trait OptIterate<R>
where
    R: Real,
{
    type S: SparseLinearSolver<R>;
    /// method to run a single iteration to update variables
    /// use to implement your own optimization iterate procedure
    /// need a implementation
    /// - if the iteration is successful return SUCCESS
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
        VC: VariablesContainer<R>;
    fn linear_solver(&self) -> &Self::S;
    fn base_params(&self) -> &NonlinearOptimizerParams;
    fn reset(&mut self) {}
}

// #[derive(Default)]
pub struct NonlinearOptimizer<O, R = f64>
where
    R: Real,
    O: OptIterate<R>,
{
    __marker: PhantomData<R>,
    /// linearization sparsity pattern
    pub sparsity: LinSpasityPattern,
    /// cached internal optimization status, used by iterate() method
    pub iterations: usize,
    /// error norm of values pass in iterate(), can be used by iterate
    /// (should be read-only)
    pub last_err_squared_norm: f64,
    /// error norm of values pass out iterate()
    /// writable by iterate(), if iterate update this value
    /// then set err_squared_norm_ to true
    pub err_squared_norm: f64,
    /// flag err_squared_norm_ is up to date by iterate()
    pub err_uptodate: bool,
    /// optimizer that implement iteration function
    pub opt: O,
}
impl<R> Default for NonlinearOptimizer<GaussNewtonOptimizer<R>, R>
where
    R: Real + Default,
{
    fn default() -> Self {
        NonlinearOptimizer {
            __marker: PhantomData,
            sparsity: LinSpasityPattern::default(),
            iterations: 0,
            last_err_squared_norm: 0.0,
            err_squared_norm: 0.0,
            err_uptodate: false,
            opt: GaussNewtonOptimizer::<R>::default(),
        }
    }
}

impl<O, R> NonlinearOptimizer<O, R>
where
    R: Real,
    O: OptIterate<R>,
{
    pub fn new(opt: O) -> Self {
        NonlinearOptimizer {
            __marker: PhantomData,
            sparsity: LinSpasityPattern::default(),
            iterations: 0,
            last_err_squared_norm: 0.0,
            err_squared_norm: 0.0,
            err_uptodate: false,
            opt,
        }
    }
    /// default optimization method with default error termination condition
    /// can be override in derived classes
    /// by default VariablesToEliminate is empty, do not eliminate any variable
    /// - if the optimization is successful return SUCCESS
    /// - if something else is returned, the value of opt_values may be undefined
    /// (depends on solver implementaion)
    #[allow(non_snake_case)]
    pub fn optimize_with_callback<VC, FC, BC>(
        &mut self,
        factors: &Factors<FC, R>,
        variables: &mut Variables<VC, R>,
        callback: Option<BC>,
    ) -> Result<(), NonlinearOptimizationError>
    where
        R: RealField,
        FC: FactorsContainer<R>,
        VC: VariablesContainer<R>,
        BC: Fn(usize, f64, &Factors<FC, R>, &Variables<VC, R>),
    {
        let tri = HessianTriangle::Upper;
        // linearization sparsity pattern
        let variable_ordering = variables.default_variable_ordering();
        let A_rows: usize;
        // let A_cols: usize;
        let mut A_values = Vec::<R>::new();
        let start = Instant::now();
        if self.opt.linear_solver().is_normal() {
            self.sparsity = LinSpasityPattern::Hessian(construct_hessian_sparsity(
                factors,
                variables,
                &variable_ordering,
                tri,
            ));
        } else {
            // self.j_sparsity = construct_jacobian_sparsity(factors, variables, &variable_ordering);
            // A_rows = self.j_sparsity.base.A_rows;
            // A_cols = self.j_sparsity.base.A_cols;
            todo!()
        }
        let csc_pattern = match &self.sparsity {
            LinSpasityPattern::Jacobian(_sparsity) => {
                todo!()
            }
            LinSpasityPattern::Hessian(sparsity) => SparsityPattern::try_from_offsets_and_indices(
                sparsity.base.A_cols,
                sparsity.base.A_cols,
                sparsity.outer_index.clone(),
                sparsity.inner_index.clone(),
            )
            .unwrap(),
        };
        match &self.sparsity {
            LinSpasityPattern::Jacobian(_) => todo!(),
            LinSpasityPattern::Hessian(sparsity) => {
                A_rows = sparsity.base.A_cols;
                // A_cols = sparsity.base.A_cols;
                A_values.resize(sparsity.total_nnz_AtA_cols, R::from_f64(0.0).unwrap());
            }
        }
        let _duration = start.elapsed();
        // println!("build starsity time: {:?}", duration);
        // init vars and errors
        self.iterations = 0;
        self.last_err_squared_norm = 0.5 * factors.error_squared_norm(variables);
        assert!(self.last_err_squared_norm.is_finite());

        let params = self.opt.base_params().clone();
        if params.verbosity_level >= NonlinearOptimizerVerbosityLevel::Iteration {
            println!("initial error: {}", self.last_err_squared_norm);
        }
        if callback.is_some() {
            callback.as_ref().unwrap()(
                self.iterations,
                self.last_err_squared_norm,
                factors,
                variables,
            );
        }
        let mut b: DVector<R> = DVector::zeros(A_rows);
        self.opt.reset();
        while self.iterations < params.max_iterations {
            b.fill(R::zero());
            A_values.fill(R::zero());
            let start = Instant::now();
            match &self.sparsity {
                LinSpasityPattern::Jacobian(_sparsity) => {
                    // jacobian linearization
                    // linearzation_jacobian(factors, variables, &self.j_sparsity, &mut A, &mut b);
                    todo!()
                }
                LinSpasityPattern::Hessian(sparsity) => {
                    if self.opt.linear_solver().is_normal_lower() {
                        // lower hessian linearization
                        linearization_hessian(factors, variables, sparsity, &mut A_values, &mut b);
                    } else {
                        // full hessian linearization
                        // linearzation_full_hessian(factors, variables, &self.h_sparsity, &mut A, &mut b);
                        todo!()
                    }
                }
            }
            let _duration = start.elapsed();
            // println!("linearize time: {:?}", duration);
            let A = CscMatrix::try_from_pattern_and_values(csc_pattern.clone(), A_values.clone())
                .expect("CSC data must conform to format specifications");
            // initiailize the linear solver if needed at first iteration
            if self.iterations == 0 {
                let start = Instant::now();
                self.opt.linear_solver().initialize(&A);
                let _duration = start.elapsed();
                // println!("linear_solver().initialize time: {:?}", duration);
            }
            let start = Instant::now();
            //iterate through
            let iterate_result = self.opt.iterate(
                factors,
                variables,
                &variable_ordering,
                LinSysWrapper::new(&A, &b),
                self.last_err_squared_norm,
            );
            let _duration = start.elapsed();
            // println!("opt iterate time: {:?}", duration);
            self.iterations += 1;

            //TODO: do better
            // check linear solver status and return if not success
            match iterate_result {
                Ok(data) => {
                    self.err_uptodate = data.err_uptodate;
                    self.err_squared_norm = data.err_squared_norm;
                }
                Err(err) => return Err(err),
            }

            // check error for stop condition
            let start = Instant::now();
            let curr_err: f64;
            if self.err_uptodate {
                // err has be updated by iterate()
                curr_err = self.err_squared_norm;
                self.err_uptodate = false;
            } else {
                curr_err = 0.5 * factors.error_squared_norm(variables);
            }
            assert!(curr_err.is_finite());

            if callback.is_some() {
                callback.as_ref().unwrap()(self.iterations, curr_err, factors, variables);
            }
            let _duration = start.elapsed();
            // println!("compute error time: {:?}", duration);

            if params.verbosity_level >= NonlinearOptimizerVerbosityLevel::Iteration {
                println!("iteration: {}, error: {}", self.iterations, curr_err);
            }

            if curr_err - self.last_err_squared_norm > 1e-20 {
                eprintln!("Warning: optimizer cannot decrease error");
                return Err(NonlinearOptimizationError::ErrorIncrease);
            }

            if self.error_stop_condition(self.last_err_squared_norm, curr_err) {
                if params.verbosity_level >= NonlinearOptimizerVerbosityLevel::Iteration {
                    println!("reach stop condition, optimization success");
                }
                return Ok(());
            }

            self.last_err_squared_norm = curr_err;
        }
        Err(NonlinearOptimizationError::MaxIteration)
    }
    pub fn optimize<VC, FC>(
        &mut self,
        factors: &Factors<FC, R>,
        variables: &mut Variables<VC, R>,
    ) -> Result<(), NonlinearOptimizationError>
    where
        R: RealField,
        FC: FactorsContainer<R>,
        VC: VariablesContainer<R>,
    {
        self.optimize_with_callback(
            factors,
            variables,
            None::<fn(usize, f64, &Factors<FC, R>, &Variables<VC, R>) -> ()>,
        )
    }
    /// default stop condition using error threshold
    /// return true if stop condition meets
    fn error_stop_condition(&self, last_err: f64, curr_err: f64) -> bool {
        let params = self.opt.base_params();
        ((last_err - curr_err) < params.min_abs_err_decrease)
            || ((last_err - curr_err) / last_err < params.min_rel_err_decrease)
    }
}
