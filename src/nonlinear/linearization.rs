use nalgebra::{DMatrix, DVector};
use std::{collections::HashSet, ops::AddAssign};

use crate::{
    core::{
        factor::Factor, factors::Factors, factors_container::FactorsContainer,
        loss_function::LossFunction, variables::Variables, variables_container::VariablesContainer,
        Real,
    },
    nonlinear::sparsity_pattern::HessianTriangle,
};

use super::sparsity_pattern::{HessianSparsityPattern, JacobianSparsityPattern};
use core::hash::Hash;
fn has_unique_elements<T>(iter: T) -> bool
where
    T: IntoIterator,
    T::Item: Eq + Hash,
{
    let mut uniq = HashSet::new();
    iter.into_iter().all(move |x| uniq.insert(x))
}
#[allow(non_snake_case)]
pub fn linearzation_jacobian<R, VC, FC>(
    factors: &Factors<FC, R>,
    variables: &Variables<VC, R>,
    sparsity: &JacobianSparsityPattern,
    A: &mut DMatrix<R>,
    b: &mut DVector<R>,
) where
    R: Real,
    VC: VariablesContainer<R>,
    FC: FactorsContainer<R>,
{
    assert_eq!(A.nrows(), sparsity.base.A_rows);
    assert_eq!(A.ncols(), sparsity.base.A_cols);
    assert_eq!(b.nrows(), sparsity.base.A_rows);
    let mut err_row_counter = 0;
    for f_index in 0..factors.len() {
        // factor dim
        let f_dim = factors.dim_at(f_index).unwrap();
        let keys = factors.keys_at(f_index).unwrap();
        let f_len = keys.len();

        assert!(has_unique_elements(keys));
        let mut jacobian_col = Vec::<usize>::with_capacity(f_len);
        let mut jacobian_col_local = Vec::<usize>::with_capacity(f_len);
        let mut jacobian_ncols = Vec::<usize>::with_capacity(f_len);
        let mut local_col: usize = 0;
        for vkey in keys {
            let key_idx = sparsity.base.var_ordering.search_key(*vkey).unwrap();
            jacobian_col.push(sparsity.base.var_col[key_idx]);
            jacobian_col_local.push(local_col);
            let var_dim = sparsity.base.var_dim[key_idx];
            jacobian_ncols.push(var_dim);
            local_col += var_dim;
        }
        let mut error = DVector::<R>::zeros(f_dim);
        let mut jacobian = DMatrix::<R>::zeros(f_dim, local_col);
        assert!(factors.jacobian_error_at(
            variables,
            jacobian.as_view_mut(),
            error.as_view_mut(),
            f_index
        ));

        //  whiten err and jacobians
        factors.weight_jacobians_error_in_place_at(
            variables,
            error.as_view_mut(),
            jacobian.as_view_mut(),
            f_index,
        );
        b.rows_mut(err_row_counter, f_dim).copy_from(&error);

        for j_idx in 0..jacobian_col.len() {
            let jacobian = &jacobian.columns(jacobian_col_local[j_idx], jacobian_ncols[j_idx]);
            A.view_mut(
                (err_row_counter, jacobian_col[j_idx]),
                (jacobian.nrows(), jacobian.ncols()),
            )
            .copy_from(jacobian);
        }

        err_row_counter += f_dim;
    }
}
fn linearize_hessian_single_factor<F, VC, T>(
    factor: &F,
    variables: &Variables<VC, T>,
    sparsity: &HessianSparsityPattern,
    hessian_values: &mut [T],
    gradient: &mut DVector<T>,
) where
    T: Real,
    F: Factor<T>,
    VC: VariablesContainer<T>,
{
    let tri = sparsity.tri;
    let f_keys = factor.keys();
    debug_assert!(has_unique_elements(f_keys));
    let f_len = factor.len();
    let f_dim: usize = factor.dim();
    let mut var_idx = Vec::<usize>::with_capacity(f_len);
    let mut jacobian_col = Vec::<usize>::with_capacity(f_len);
    let mut jacobian_col_local = Vec::<usize>::with_capacity(f_len);
    let mut jacobian_ncols = Vec::<usize>::with_capacity(f_len);
    let mut local_col: usize = 0;
    for key in f_keys {
        // A col start index
        let key_idx = sparsity.base.var_ordering.search_key(*key).unwrap();
        var_idx.push(key_idx);
        jacobian_col.push(sparsity.base.var_col[key_idx]);
        jacobian_col_local.push(local_col);
        let var_dim = sparsity.base.var_dim[key_idx];
        jacobian_ncols.push(var_dim);
        local_col += var_dim;
    }

    let mut error = DVector::<T>::zeros(f_dim);
    let mut jacobian = DMatrix::<T>::zeros(f_dim, local_col);
    factor.jacobian_error(variables, jacobian.as_view_mut(), error.as_view_mut());
    //  whiten err and jacobians
    if let Some(loss) = factor.loss_function() {
        loss.weight_jacobians_error_in_place(error.as_view_mut(), jacobian.as_view_mut());
    }
    // let jacobians = jacobians;
    // let error = error;

    // let mut stackJtJ = DMatrix::<R>::zeros(stackJ.ncols(), stackJ.ncols());
    // adaptive multiply for better speed
    // if stackJ.ncols() > 12 {
    //     // memset(stackJtJ.data(), 0, stackJ.cols() * stackJ.cols() * sizeof(double));
    //     // stackJtJ.selfadjointView<Eigen::Lower>().rankUpdate(stackJ.transpose());
    //     let sTs = stackJ.transpose() * stackJ.clone();
    //     stackJtJ.copy_from(&sTs);
    // } else {
    //     let sTs = stackJ.transpose() * stackJ.clone();
    //     stackJtJ.copy_from(&sTs);
    // }
    // let sts = ;
    // stackJtJ.copy_from(&(stackJ.transpose() * stackJ.clone()));

    let stackJtb = jacobian.transpose() * error;
    let stackJtJ = jacobian.transpose() * jacobian;
    // #ifdef MINISAM_WITH_MULTI_THREADS
    //   mutex_b.lock();
    // #endif

    for j_idx in 0..f_len {
        gradient
            .rows_mut(jacobian_col[j_idx], jacobian_ncols[j_idx])
            .add_assign(&stackJtb.rows(jacobian_col_local[j_idx], jacobian_ncols[j_idx]));
    }

    // #ifdef MINISAM_WITH_MULTI_THREADS
    //   mutex_b.unlock();
    //   mutex_A.lock();
    // #endif

    for j_idx in 0..f_len {
        // scan by row
        let nnz_AtA_vars_accum_var = sparsity.nnz_AtA_vars_accum[var_idx[j_idx]];
        let mut value_idx: usize = nnz_AtA_vars_accum_var;
        let j_col_local = jacobian_col_local[j_idx];
        let j_ncols = jacobian_ncols[j_idx];
        let j_col = jacobian_col[j_idx];
        for j in 0..j_ncols {
            let i_range = match tri {
                HessianTriangle::Upper => 0..j + 1,
                HessianTriangle::Lower => j..j_ncols,
            };
            let fill = |values: &mut [T], idx: &mut usize| {
                for i in i_range.clone() {
                    values[*idx] += stackJtJ[(j_col_local + i, j_col_local + j)];
                    *idx += 1;
                }
            };
            match tri {
                HessianTriangle::Upper => {
                    value_idx += sparsity.nnz_AtA_cols[j_col + j] - j - 1;
                    fill(hessian_values, &mut value_idx);
                }
                HessianTriangle::Lower => {
                    fill(hessian_values, &mut value_idx);
                    value_idx += sparsity.nnz_AtA_cols[j_col + j] + j - j_ncols;
                }
            }
        }
    }

    // #ifdef MINISAM_WITH_MULTI_THREADS
    //   mutex_A.unlock();
    // #endif

    // update lower/upper non-diag hessian blocks
    for j1_idx in 0..f_len {
        let j1_var_idx = var_idx[j1_idx];
        for j2_idx in 0..f_len {
            let j2_var_idx = var_idx[j2_idx];
            let comp = match tri {
                HessianTriangle::Upper => j1_var_idx < j2_var_idx,
                HessianTriangle::Lower => j1_var_idx > j2_var_idx,
            };
            // we know var_idx[j1_idx] != var_idx[j2_idx]
            // assume var_idx[j1_idx] >(lower) <(upper) var_idx[j2_idx]
            // insert to block location (j1_idx, j2_idx)
            if comp {
                let nnz_AtA_vars_accum_var2 = sparsity.nnz_AtA_vars_accum[j2_var_idx];
                let var2_dim = sparsity.base.var_dim[j2_var_idx];

                let inner_insert_var2_var1 = sparsity.inner_insert_map[j2_var_idx]
                    .get(&j1_var_idx)
                    .unwrap();
                let mut value_idx: usize;
                let val_offset: usize;
                match tri {
                    HessianTriangle::Upper => {
                        value_idx = nnz_AtA_vars_accum_var2 + inner_insert_var2_var1;
                        val_offset = 0;
                    }
                    HessianTriangle::Lower => {
                        value_idx = nnz_AtA_vars_accum_var2 + var2_dim + inner_insert_var2_var1;
                        val_offset = 1;
                    }
                }
                // #ifdef MINISAM_WITH_MULTI_THREADS
                //         mutex_A.lock();
                // #endif

                for j in 0..jacobian_ncols[j2_idx] {
                    for i in 0..jacobian_ncols[j1_idx] {
                        let (mut r, mut c) = (
                            jacobian_col_local[j1_idx] + i,
                            jacobian_col_local[j2_idx] + j,
                        );
                        // to access only lower part (if only lower computed)
                        if j1_idx <= j2_idx {
                            (r, c) = (c, r);
                        }
                        hessian_values[value_idx] += stackJtJ[(r, c)];
                        value_idx += 1;
                    }
                    value_idx += sparsity.nnz_AtA_cols[jacobian_col[j2_idx] + j]
                        - val_offset
                        - jacobian_ncols[j1_idx];
                }

                // #ifdef MINISAM_WITH_MULTI_THREADS
                //         mutex_A.unlock();
                // #endif
            }
        }
    }
}

#[allow(non_snake_case)]
#[inline(always)]
fn linearzation_hessian_single_factor<R, VC, FC>(
    f_index: usize,
    factors: &Factors<FC, R>,
    variables: &Variables<VC, R>,
    sparsity: &HessianSparsityPattern,
    AtA_values: &mut [R],
    Atb: &mut DVector<R>,
) where
    R: Real,
    VC: VariablesContainer<R>,
    FC: FactorsContainer<R>,
{
    let tri = sparsity.tri;
    let f_keys = factors.keys_at(f_index).unwrap();
    debug_assert!(has_unique_elements(f_keys));
    let f_len = f_keys.len();
    let f_dim: usize = factors.dim_at(f_index).unwrap();
    let mut var_idx = Vec::<usize>::with_capacity(f_len);
    let mut jacobian_col = Vec::<usize>::with_capacity(f_len);
    let mut jacobian_col_local = Vec::<usize>::with_capacity(f_len);
    let mut jacobian_ncols = Vec::<usize>::with_capacity(f_len);
    let mut local_col: usize = 0;
    for key in f_keys {
        // A col start index
        let key_idx = sparsity.base.var_ordering.search_key(*key).unwrap();
        var_idx.push(key_idx);
        jacobian_col.push(sparsity.base.var_col[key_idx]);
        jacobian_col_local.push(local_col);
        let var_dim = sparsity.base.var_dim[key_idx];
        jacobian_ncols.push(var_dim);
        local_col += var_dim;
    }

    let mut error = DVector::<R>::zeros(f_dim);
    let mut jacobian = DMatrix::<R>::zeros(f_dim, local_col);
    assert!(factors.jacobian_error_at(
        variables,
        jacobian.as_view_mut(),
        error.as_view_mut(),
        f_index
    ));
    //  whiten err and jacobians
    factors.weight_jacobians_error_in_place_at(
        variables,
        error.as_view_mut(),
        jacobian.as_view_mut(),
        f_index,
    );
    // let jacobians = jacobians;
    // let error = error;

    // let mut stackJtJ = DMatrix::<R>::zeros(stackJ.ncols(), stackJ.ncols());
    // adaptive multiply for better speed
    // if stackJ.ncols() > 12 {
    //     // memset(stackJtJ.data(), 0, stackJ.cols() * stackJ.cols() * sizeof(double));
    //     // stackJtJ.selfadjointView<Eigen::Lower>().rankUpdate(stackJ.transpose());
    //     let sTs = stackJ.transpose() * stackJ.clone();
    //     stackJtJ.copy_from(&sTs);
    // } else {
    //     let sTs = stackJ.transpose() * stackJ.clone();
    //     stackJtJ.copy_from(&sTs);
    // }
    // let sts = ;
    // stackJtJ.copy_from(&(stackJ.transpose() * stackJ.clone()));

    let stackJtb = jacobian.transpose() * error;
    let stackJtJ = jacobian.transpose() * jacobian;
    // #ifdef MINISAM_WITH_MULTI_THREADS
    //   mutex_b.lock();
    // #endif

    for j_idx in 0..f_len {
        Atb.rows_mut(jacobian_col[j_idx], jacobian_ncols[j_idx])
            .add_assign(&stackJtb.rows(jacobian_col_local[j_idx], jacobian_ncols[j_idx]));
    }

    // #ifdef MINISAM_WITH_MULTI_THREADS
    //   mutex_b.unlock();
    //   mutex_A.lock();
    // #endif

    for j_idx in 0..f_len {
        // scan by row
        let nnz_AtA_vars_accum_var = sparsity.nnz_AtA_vars_accum[var_idx[j_idx]];
        let mut value_idx: usize = nnz_AtA_vars_accum_var;
        let j_col_local = jacobian_col_local[j_idx];
        let j_ncols = jacobian_ncols[j_idx];
        let j_col = jacobian_col[j_idx];
        for j in 0..j_ncols {
            let i_range = match tri {
                HessianTriangle::Upper => 0..j + 1,
                HessianTriangle::Lower => j..j_ncols,
            };
            let fill = |values: &mut [R], idx: &mut usize| {
                for i in i_range.clone() {
                    values[*idx] += stackJtJ[(j_col_local + i, j_col_local + j)];
                    *idx += 1;
                }
            };
            match tri {
                HessianTriangle::Upper => {
                    value_idx += sparsity.nnz_AtA_cols[j_col + j] - j - 1;
                    fill(AtA_values, &mut value_idx);
                }
                HessianTriangle::Lower => {
                    fill(AtA_values, &mut value_idx);
                    value_idx += sparsity.nnz_AtA_cols[j_col + j] + j - j_ncols;
                }
            }
        }
    }

    // #ifdef MINISAM_WITH_MULTI_THREADS
    //   mutex_A.unlock();
    // #endif

    // update lower/upper non-diag hessian blocks
    for j1_idx in 0..f_len {
        let j1_var_idx = var_idx[j1_idx];
        for j2_idx in 0..f_len {
            let j2_var_idx = var_idx[j2_idx];
            let comp = match tri {
                HessianTriangle::Upper => j1_var_idx < j2_var_idx,
                HessianTriangle::Lower => j1_var_idx > j2_var_idx,
            };
            // we know var_idx[j1_idx] != var_idx[j2_idx]
            // assume var_idx[j1_idx] >(lower) <(upper) var_idx[j2_idx]
            // insert to block location (j1_idx, j2_idx)
            if comp {
                let nnz_AtA_vars_accum_var2 = sparsity.nnz_AtA_vars_accum[j2_var_idx];
                let var2_dim = sparsity.base.var_dim[j2_var_idx];

                let inner_insert_var2_var1 = sparsity.inner_insert_map[j2_var_idx]
                    .get(&j1_var_idx)
                    .unwrap();
                let mut value_idx: usize;
                let val_offset: usize;
                match tri {
                    HessianTriangle::Upper => {
                        value_idx = nnz_AtA_vars_accum_var2 + inner_insert_var2_var1;
                        val_offset = 0;
                    }
                    HessianTriangle::Lower => {
                        value_idx = nnz_AtA_vars_accum_var2 + var2_dim + inner_insert_var2_var1;
                        val_offset = 1;
                    }
                }
                // #ifdef MINISAM_WITH_MULTI_THREADS
                //         mutex_A.lock();
                // #endif

                for j in 0..jacobian_ncols[j2_idx] {
                    for i in 0..jacobian_ncols[j1_idx] {
                        let (mut r, mut c) = (
                            jacobian_col_local[j1_idx] + i,
                            jacobian_col_local[j2_idx] + j,
                        );
                        // to access only lower part (if only lower computed)
                        if j1_idx <= j2_idx {
                            (r, c) = (c, r);
                        }
                        AtA_values[value_idx] += stackJtJ[(r, c)];
                        value_idx += 1;
                    }
                    value_idx += sparsity.nnz_AtA_cols[jacobian_col[j2_idx] + j]
                        - val_offset
                        - jacobian_ncols[j1_idx];
                }

                // #ifdef MINISAM_WITH_MULTI_THREADS
                //         mutex_A.unlock();
                // #endif
            }
        }
    }
}

/// Computes hessian of cost function such:
/// $$H \Delta {\textbf{x}}=-\textbf{g}$$
#[allow(non_snake_case)]
pub fn linearization_hessian<R, VC, FC>(
    factors: &Factors<FC, R>,
    variables: &Variables<VC, R>,
    sparsity: &HessianSparsityPattern,
    AtA_values: &mut [R],
    Atb: &mut DVector<R>,
) where
    R: Real,
    VC: VariablesContainer<R>,
    FC: FactorsContainer<R>,
{
    for f_index in 0..factors.len() {
        linearzation_hessian_single_factor(f_index, factors, variables, sparsity, AtA_values, Atb);
    }
}

#[allow(non_snake_case)]
pub fn linearzation_full_hessian<R, VC, FC>(
    _factors: &Factors<FC, R>,
    _variables: &Variables<VC, R>,
    _sparsity: &HessianSparsityPattern,
    _A: &mut DMatrix<R>,
    _b: &mut DVector<R>,
) where
    R: Real,
    VC: VariablesContainer<R>,
    FC: FactorsContainer<R>,
{
    todo!()
}
#[cfg(test)]
mod tests {

    use matrixcompare::assert_matrix_eq;
    use nalgebra::{DMatrix, DVector, Matrix3x4};
    use nalgebra_sparse::{pattern::SparsityPattern, CscMatrix};
    use rand::{distributions::Uniform, prelude::Distribution};

    use crate::{
        core::{
            factor::tests::{FactorA, FactorB, RandomBlockFactor},
            factors::Factors,
            factors_container::FactorsContainer,
            key::Vkey,
            variable::tests::{RandomVariable, VariableA, VariableB},
            variable_ordering::VariableOrdering,
            variables::Variables,
            variables_container::VariablesContainer,
        },
        nonlinear::{
            linearization::{linearization_hessian, linearzation_jacobian},
            sparsity_pattern::{
                construct_hessian_sparsity, construct_jacobian_sparsity, HessianTriangle,
            },
        },
    };

    #[test]
    #[allow(non_snake_case)]
    fn linearize_jacobian() {
        type Real = f64;
        let container = ().and_variable::<VariableA<Real>>().and_variable::<VariableB<Real>>();
        let mut variables = Variables::new(container);
        variables.add(Vkey(0), VariableA::<Real>::new(1.0));
        variables.add(Vkey(1), VariableB::<Real>::new(5.0));
        variables.add(Vkey(2), VariableB::<Real>::new(10.0));

        let container = ().and_factor::<FactorA<Real>>().and_factor::<FactorB<Real>>();
        let mut factors = Factors::new(container);
        factors.add(FactorA::new(1.0, None, Vkey(0), Vkey(1)));
        factors.add(FactorB::new(2.0, None, Vkey(1), Vkey(2)));
        let variable_ordering = variables.default_variable_ordering();
        let pattern = construct_jacobian_sparsity(&factors, &variables, &variable_ordering);
        let mut A = DMatrix::<Real>::zeros(pattern.base.A_rows, pattern.base.A_cols);
        let mut b = DVector::<Real>::zeros(pattern.base.A_rows);
        linearzation_jacobian(&factors, &variables, &pattern, &mut A, &mut b);
        println!("A {:?}", A);
        println!("b {:?}", b);
    }

    #[test]
    #[allow(non_snake_case)]
    fn linearize_hessian_0() {
        type Real = f64;
        let container = ().and_variable::<RandomVariable<Real>>();
        let mut variables = Variables::new(container);
        variables.add(Vkey(0), RandomVariable::<Real>::default());
        variables.add(Vkey(1), RandomVariable::<Real>::default());
        variables.add(Vkey(2), RandomVariable::<Real>::default());

        let container =
            ().and_factor::<FactorA<Real>>()
                .and_factor::<FactorB<Real>>()
                .and_factor::<RandomBlockFactor<Real>>();
        let mut factors = Factors::new(container);
        factors.add(RandomBlockFactor::new(Vkey(0), Vkey(1)));
        factors.add(RandomBlockFactor::new(Vkey(0), Vkey(2)));
        factors.add(RandomBlockFactor::new(Vkey(1), Vkey(2)));
        // let variable_ordering = variables.default_variable_ordering();
        let variable_ordering = VariableOrdering::new(vec![Vkey(0), Vkey(1), Vkey(2)].as_slice());
        let pattern = construct_jacobian_sparsity(&factors, &variables, &variable_ordering);
        let _A = DMatrix::<Real>::zeros(pattern.base.A_rows, pattern.base.A_cols);
        let _b = DVector::<Real>::zeros(pattern.base.A_rows);
        // linearzation_jacobian(&factors, &variables, &pattern, &mut A, &mut b);
        // println!("A {}", A);
        // println!("b {}", b);
        // println!("J AtA: {}", A.transpose() * A.clone());
        // println!("J Atb: {}", A.transpose() * b.clone());
        let minor_indices = vec![1, 2, 0, 2, 4, 2, 1, 4]; //inner_index
        let major_offsets = vec![0, 2, 4, 5, 6, 8]; //outer_index
        let values = vec![22.0, 7.0, 3.0, 5.0, 14.0, 1.0, 17.0, 8.0]; //values

        // The dense representation of the CSC data, for comparison
        let _dense = Matrix3x4::new(1.0, 2.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 4.0, 0.0);

        let patt =
            SparsityPattern::try_from_offsets_and_indices(5, 5, major_offsets, minor_indices);
        // println!("patt: {:?}", patt);
        // The constructor validates the raw CSC data and returns an error if it is invalid.
        let csc = CscMatrix::try_from_pattern_and_values(patt.unwrap(), values)
            .expect("CSC data must conform to format specifications");
        let _csc_d: DMatrix<f64> = DMatrix::<f64>::from(&csc);
        // assert_matrix_eq!(csc, dense);
        // println!("csc {}", csc_d);
        let tri = HessianTriangle::Lower;
        let sparsity = construct_hessian_sparsity(&factors, &variables, &variable_ordering, tri);
        println!("corl vars {:?}", sparsity.corl_vars);
        println!("inner map {:?}", sparsity.inner_insert_map);
        let mut AtA_values = Vec::<f64>::with_capacity(sparsity.total_nnz_AtA_cols);
        AtA_values.resize(sparsity.total_nnz_AtA_cols, 0.0);
        let mut Atb = DVector::<f64>::zeros(sparsity.base.A_cols);
        linearization_hessian(&factors, &variables, &sparsity, &mut AtA_values, &mut Atb);
        // AtA_values.fill(1.0);
        // for i in 0..AtA_values.len() {
        //     AtA_values[i] = (i + 1) as Real;
        // }
        // println!("Atb {}", Atb);
        // println!("AtA_values {:?}", AtA_values);

        let minor_indices = sparsity.inner_index;
        let major_offsets = sparsity.outer_index;
        println!("minor_indeces: {:?}", minor_indices);
        println!("major_ossets: {:?}", major_offsets);
        let patt = SparsityPattern::try_from_offsets_and_indices(
            sparsity.base.A_cols,
            sparsity.base.A_cols,
            major_offsets,
            minor_indices,
        );
        let AtA = CscMatrix::try_from_pattern_and_values(patt.unwrap(), AtA_values)
            .expect("CSC data must conform to format specifications");
        let AtA: DMatrix<f64> = DMatrix::<f64>::from(&AtA);
        // assert_matrix_eq!(csc, dense);
        println!("AtA {}", AtA);
        // println!("Atb {}", Atb);
    }
    #[test]
    #[allow(non_snake_case)]
    fn linearize_hessian_random() {
        type Real = f64;
        for _ in 0..100 {
            let container = ().and_variable::<RandomVariable<Real>>();
            let mut variables = Variables::new(container);
            let variables_cnt = 100;
            for k in 0..variables_cnt {
                variables.add(Vkey(k), RandomVariable::<Real>::default());
            }

            let container = ().and_factor::<RandomBlockFactor<Real>>();
            let mut factors = Factors::new(container);
            let factors_cnt = 1000;
            let mut rng = rand::thread_rng();
            let rnd_key = Uniform::from(0..variables_cnt);
            for _ in 0..factors_cnt {
                let k0 = rnd_key.sample(&mut rng);
                let k1 = rnd_key.sample(&mut rng);
                if k0 == k1 {
                    continue;
                }
                factors.add(RandomBlockFactor::new(Vkey(k0), Vkey(k1)));
            }
            let variable_ordering = variables.default_variable_ordering();
            let pattern = construct_jacobian_sparsity(&factors, &variables, &variable_ordering);
            let mut A = DMatrix::<Real>::zeros(pattern.base.A_rows, pattern.base.A_cols);
            let mut b = DVector::<Real>::zeros(pattern.base.A_rows);
            linearzation_jacobian(&factors, &variables, &pattern, &mut A, &mut b);
            let JAtA = A.transpose() * A.clone();
            let JAtb = A.transpose() * b.clone();

            let tri = HessianTriangle::Lower;
            let sparsity =
                construct_hessian_sparsity(&factors, &variables, &variable_ordering, tri);
            let mut AtA_values = Vec::<f64>::with_capacity(sparsity.total_nnz_AtA_cols);
            AtA_values.resize(sparsity.total_nnz_AtA_cols, 0.0);
            let mut Atb = DVector::<f64>::zeros(sparsity.base.A_cols);
            linearization_hessian(&factors, &variables, &sparsity, &mut AtA_values, &mut Atb);

            let minor_indices = sparsity.inner_index;
            let major_offsets = sparsity.outer_index;
            let patt = SparsityPattern::try_from_offsets_and_indices(
                sparsity.base.A_cols,
                sparsity.base.A_cols,
                major_offsets,
                minor_indices,
            );
            let AtA = CscMatrix::try_from_pattern_and_values(patt.unwrap(), AtA_values)
                .expect("CSC data must conform to format specifications");
            let AtA: DMatrix<f64> = DMatrix::<f64>::from(&AtA);

            assert_matrix_eq!(AtA, JAtA.lower_triangle(), comp = abs, tol = 1e-9);
            assert_matrix_eq!(Atb, JAtb, comp = abs, tol = 1e-9);
        }
    }
}
