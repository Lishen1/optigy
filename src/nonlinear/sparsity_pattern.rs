use std::{collections::BTreeSet, slice::IterMut};

use crate::core::{
    factors::Factors, factors_container::FactorsContainer, variable_ordering::VariableOrdering,
    variables::Variables, variables_container::VariablesContainer, Real,
};
use hashbrown::HashMap;

/// base class for A and A'A sparsity pattern, if variable ordering is fixed,
/// only need to be constructed once for different linearzation runs
#[cfg_attr(debug_assertions, derive(Debug))]
#[allow(non_snake_case)]
#[derive(Default)]
pub struct SparsityPatternBase {
    /// basic size information
    pub A_rows: usize,
    /// = b size
    pub A_cols: usize,
    /// = A'A size
    pub var_ordering: VariableOrdering,

    /// var_dim: dim of each vars (using ordering of var_ordering)
    /// var_col: start col of each vars (using ordering of var_ordering)
    pub var_dim: Vec<usize>,
    pub var_col: Vec<usize>,
}

// struct store  given variable ordering
#[cfg_attr(debug_assertions, derive(Debug))]
#[derive(Default)]
pub struct JacobianSparsityPattern {
    pub base: SparsityPatternBase,
    /// Eigen::Sparse memory allocation information
    /// number of non-zeros count for each col of A, use for Eigen sparse matrix A
    /// reservation
    pub nnz_cols: Vec<usize>,

    /// start row of each factor
    pub factor_err_row: Vec<usize>,
}

/// struct store A'A lower part sparsity pattern given variable ordering
/// note this does not apply to full A'A!
#[cfg_attr(debug_assertions, derive(Debug))]
#[allow(non_snake_case)]
#[derive(Default)]
pub struct HessianSparsityPattern {
    pub base: SparsityPatternBase,
    /// number of non-zeros count for each col of AtA (each row of A)
    /// use for Eigen sparse matrix AtA reserve
    pub nnz_AtA_cols: Vec<usize>,
    pub total_nnz_AtA_cols: usize,

    /// accumulated nnzs in AtA before each var
    /// index: var idx, value: total skip nnz
    pub nnz_AtA_vars_accum: Vec<usize>,

    /// corl_vars: variable ordering position of all correlated vars of each var
    /// (not include self), set must be ordered
    pub corl_vars: Vec<BTreeSet<usize>>,

    /// inner index of each coorelated vars, exculde lower triangular part
    /// index: corl var idx, value: inner index
    pub inner_insert_map: Vec<HashMap<usize, usize>>,

    /// sparse matrix inner/outer index
    pub inner_index: Vec<usize>,
    pub inner_nnz_index: Vec<usize>,
    pub outer_index: Vec<usize>,
    /// lower/upper hessian triangular part
    pub tri: HessianTriangle,
}
/// construct Ax = b sparsity pattern cache from a factor graph and a set of
/// variables
pub fn construct_jacobian_sparsity<R, VC, FC>(
    factors: &Factors<FC, R>,
    variables: &Variables<VC, R>,
    variable_ordering: &VariableOrdering,
) -> JacobianSparsityPattern
where
    R: Real,
    VC: VariablesContainer<R>,
    FC: FactorsContainer<R>,
{
    let mut sparsity = JacobianSparsityPattern::default();
    // A size
    sparsity.base.A_rows = factors.dim();
    sparsity.base.A_cols = variables.dim();

    // var_dim: dim of each vars (using ordering of var_ordering)
    // var_col: start col of each vars (using ordering of var_ordering)
    sparsity.base.var_dim.reserve(variable_ordering.len());
    sparsity.base.var_col.reserve(variable_ordering.len());
    let mut col_counter: usize = 0;

    for i in 0..variable_ordering.len() {
        sparsity.base.var_col.push(col_counter);
        let vdim = variables.dim_at(variable_ordering[i]).unwrap();
        sparsity.base.var_dim.push(vdim);
        col_counter += vdim;
    }
    // counter for row of error
    sparsity.nnz_cols.resize(sparsity.base.A_cols, 0);
    // counter row of factor
    sparsity.factor_err_row.reserve(factors.len());
    let mut err_row_counter: usize = 0;

    for f_index in 0..factors.len() {
        // factor dim
        let f_dim = factors.dim_at(f_index).unwrap();
        let keys = factors.keys_at(f_index).unwrap();
        for vkey in keys {
            // A col start index
            let key_order_idx = variable_ordering.search_key(*vkey).unwrap();
            // A col non-zeros
            let var_col = sparsity.base.var_col[key_order_idx];
            for nz_col in var_col..(var_col + sparsity.base.var_dim[key_order_idx]) {
                sparsity.nnz_cols[nz_col] += f_dim;
            }
        }
        sparsity.factor_err_row.push(err_row_counter);
        err_row_counter += f_dim;
    }

    // copy var ordering
    sparsity.base.var_ordering = variable_ordering.clone();
    sparsity
}

#[derive(Debug, Copy, Clone, Default)]
pub enum HessianTriangle {
    #[default]
    Upper,
    Lower,
}

/// construct A'Ax = A'b sparsity pattern cache from a factor graph and a set of
/// variables
#[allow(non_snake_case)]
pub fn construct_hessian_sparsity<R, VC, FC>(
    factors: &Factors<FC, R>,
    variables: &Variables<VC, R>,
    variable_ordering: &VariableOrdering,
    tri: HessianTriangle,
) -> HessianSparsityPattern
where
    R: Real,
    VC: VariablesContainer<R>,
    FC: FactorsContainer<R>,
{
    let mut sparsity = HessianSparsityPattern {
        tri,
        ..Default::default()
    };

    // A size
    sparsity.base.A_rows = factors.dim();
    sparsity.base.A_cols = variables.dim();

    // var_dim: dim of each vars (using ordering of var_ordering)
    // var_col: start col of each vars (using ordering of var_ordering)
    sparsity.base.var_dim.reserve(variable_ordering.len());
    sparsity.base.var_col.reserve(variable_ordering.len());
    let mut col_counter: usize = 0;

    for i in 0..variable_ordering.len() {
        sparsity.base.var_col.push(col_counter);
        let vdim = variables.dim_at(variable_ordering[i]).unwrap();
        sparsity.base.var_dim.push(vdim);
        col_counter += vdim;
    }

    // AtA col correlated vars of lower part
    // does not include itself
    sparsity
        .corl_vars
        .resize(variable_ordering.len(), BTreeSet::default());

    for f_index in 0..factors.len() {
        let mut factor_key_order_idx: Vec<usize> = Vec::new();
        let keys = factors.keys_at(f_index).unwrap();
        factor_key_order_idx.reserve(keys.len());
        for (i, key) in keys.iter().enumerate() {
            factor_key_order_idx.push(variable_ordering.search_key(*key).unwrap_or_else(|| {
                panic!(
                    "{}th key {:?} form factor {} with keys {:?} should have corresponded variable",
                    i,
                    key,
                    factors.type_name_at(f_index).unwrap(),
                    keys
                )
            }));
        }
        for i in &factor_key_order_idx {
            for j in &factor_key_order_idx {
                let comp = match tri {
                    HessianTriangle::Upper => i > j,
                    HessianTriangle::Lower => i < j,
                };
                if comp {
                    sparsity.corl_vars[*i].insert(*j);
                };
            }
        }
    }

    sparsity.nnz_AtA_cols.resize(sparsity.base.A_cols, 0);
    sparsity
        .nnz_AtA_vars_accum
        .reserve(variable_ordering.len() + 1);
    sparsity.nnz_AtA_vars_accum.push(0);
    let mut last_nnz_AtA_vars_accum: usize = 0;

    for var_idx in 0..variable_ordering.len() {
        let self_dim = sparsity.base.var_dim[var_idx];
        let self_col = sparsity.base.var_col[var_idx];
        // self: lower/upper triangular part
        last_nnz_AtA_vars_accum += ((1 + self_dim) * self_dim) / 2;
        for i in 0..self_dim {
            let col = self_col + i;
            let nnz = match tri {
                HessianTriangle::Upper => i + 1,
                HessianTriangle::Lower => self_dim - i,
            };
            sparsity.nnz_AtA_cols[col] += nnz;
        }
        // non-self
        for corl_var_idx in &sparsity.corl_vars[var_idx] {
            last_nnz_AtA_vars_accum += sparsity.base.var_dim[*corl_var_idx] * self_dim;
            for col in self_col..(self_col + self_dim) {
                sparsity.nnz_AtA_cols[col] += sparsity.base.var_dim[*corl_var_idx];
            }
        }
        sparsity.nnz_AtA_vars_accum.push(last_nnz_AtA_vars_accum);
    }
    sparsity.total_nnz_AtA_cols = sparsity.nnz_AtA_cols.iter().sum();

    // where to insert nnz element
    sparsity
        .inner_insert_map
        .resize(variable_ordering.len(), HashMap::new());

    for var1_idx in 0..variable_ordering.len() {
        let mut nnzdim_counter: usize = 0;
        // non-self
        for var2_idx in &sparsity.corl_vars[var1_idx] {
            sparsity.inner_insert_map[var1_idx].insert(*var2_idx, nnzdim_counter);
            nnzdim_counter += sparsity.base.var_dim[*var2_idx];
        }
    }

    // prepare sparse matrix inner/outer index
    sparsity.inner_index.resize(sparsity.total_nnz_AtA_cols, 0);
    sparsity.inner_nnz_index.resize(sparsity.base.A_cols, 0);
    sparsity.outer_index.resize(sparsity.base.A_cols + 1, 0);

    let mut inner_index_ptr = sparsity.inner_index.iter_mut();
    // let mut inner_nnz_ptr = sparsity.inner_nnz_index.iter_mut();
    let mut outer_index_ptr = sparsity.outer_index.iter_mut();
    let mut out_counter: usize = 0;
    *outer_index_ptr.next().unwrap() = 0;

    for var_idx in 0..sparsity.base.var_dim.len() {
        let self_dim = sparsity.base.var_dim[var_idx];
        let self_col = sparsity.base.var_col[var_idx];
        for i in 0..self_dim {
            //inner_nnz not used (just for original eigen impl)
            // *inner_nnz_ptr.next().unwrap() =
            //     sparsity.nnz_AtA_cols[i + sparsity.base.var_col[var_idx]];
            out_counter += sparsity.nnz_AtA_cols[i + self_col];
            *outer_index_ptr.next().unwrap() = out_counter;

            let fill_non_self = |ptr: &mut IterMut<usize>| {
                for corl_idx in &sparsity.corl_vars[var_idx] {
                    for j in 0..sparsity.base.var_dim[*corl_idx] {
                        *ptr.next().unwrap() = sparsity.base.var_col[*corl_idx] + j;
                    }
                }
            };
            let fill_self = |ptr: &mut IterMut<usize>| {
                let j_range = match tri {
                    HessianTriangle::Upper => 0..i + 1,
                    HessianTriangle::Lower => i..self_dim,
                };
                for j in j_range {
                    *ptr.next().unwrap() = self_col + j;
                }
            };

            match tri {
                HessianTriangle::Upper => {
                    fill_non_self(&mut inner_index_ptr);
                    fill_self(&mut inner_index_ptr)
                }
                HessianTriangle::Lower => {
                    fill_self(&mut inner_index_ptr);
                    fill_non_self(&mut inner_index_ptr);
                }
            }
        }
    }

    // copy
    sparsity.base.var_ordering = variable_ordering.clone();
    sparsity
}
#[cfg(test)]
mod tests {
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
        nonlinear::sparsity_pattern::{
            construct_hessian_sparsity, construct_jacobian_sparsity, HessianTriangle,
        },
    };

    #[test]
    fn jacobian_sparsity_0() {
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
        factors.add(FactorB::new(3.0, None, Vkey(0), Vkey(2)));
        let variable_ordering = variables.default_variable_ordering();
        let pattern = construct_jacobian_sparsity(&factors, &variables, &variable_ordering);
        assert_eq!(pattern.base.A_rows, 9);
        assert_eq!(pattern.base.A_cols, 9);
        assert_eq!(pattern.base.var_dim, vec![3, 3, 3]);
        assert_eq!(pattern.base.var_col, vec![0, 3, 6]);
        assert_eq!(pattern.factor_err_row[0], 0);
        assert_eq!(pattern.factor_err_row[1], 3);
        assert_eq!(pattern.factor_err_row[2], 6);
        assert_eq!(
            pattern.nnz_cols[variable_ordering.search_key(Vkey(0)).unwrap() * 3],
            6
        );
        assert_eq!(
            pattern.nnz_cols[1 + variable_ordering.search_key(Vkey(0)).unwrap() * 3],
            6
        );
        assert_eq!(
            pattern.nnz_cols[2 + variable_ordering.search_key(Vkey(0)).unwrap() * 3],
            6
        );
        assert_eq!(
            pattern.nnz_cols[variable_ordering.search_key(Vkey(1)).unwrap() * 3],
            6
        );
        assert_eq!(
            pattern.nnz_cols[1 + variable_ordering.search_key(Vkey(1)).unwrap() * 3],
            6
        );
        assert_eq!(
            pattern.nnz_cols[2 + variable_ordering.search_key(Vkey(1)).unwrap() * 3],
            6
        );
        assert_eq!(
            pattern.nnz_cols[variable_ordering.search_key(Vkey(2)).unwrap() * 3],
            6
        );
        assert_eq!(
            pattern.nnz_cols[1 + variable_ordering.search_key(Vkey(2)).unwrap() * 3],
            6
        );
        assert_eq!(
            pattern.nnz_cols[2 + variable_ordering.search_key(Vkey(2)).unwrap() * 3],
            6
        );
    }
    #[test]
    fn jacobian_sparsity_1() {
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
        factors.add(FactorB::new(3.0, None, Vkey(0), Vkey(1)));
        let variable_ordering = variables.default_variable_ordering();
        let pattern = construct_jacobian_sparsity(&factors, &variables, &variable_ordering);
        assert_eq!(pattern.base.A_rows, 9);
        assert_eq!(pattern.base.A_cols, 9);
        assert_eq!(pattern.base.var_dim.len(), 3);
        assert_eq!(pattern.base.var_col.len(), 3);
        assert_eq!(pattern.base.var_dim, vec![3, 3, 3]);
        assert_eq!(pattern.base.var_col, vec![0, 3, 6]);
        assert_eq!(pattern.factor_err_row.len(), 3);
        assert_eq!(pattern.factor_err_row[0], 0);
        assert_eq!(pattern.factor_err_row[1], 3);
        assert_eq!(pattern.factor_err_row[2], 6);
        assert_eq!(pattern.nnz_cols.len(), 9);
        assert_eq!(
            pattern.nnz_cols[variable_ordering.search_key(Vkey(0)).unwrap() * 3],
            6
        );
        assert_eq!(
            pattern.nnz_cols[1 + variable_ordering.search_key(Vkey(0)).unwrap() * 3],
            6
        );
        assert_eq!(
            pattern.nnz_cols[2 + variable_ordering.search_key(Vkey(0)).unwrap() * 3],
            6
        );
        assert_eq!(
            pattern.nnz_cols[variable_ordering.search_key(Vkey(1)).unwrap() * 3],
            9
        );
        assert_eq!(
            pattern.nnz_cols[1 + variable_ordering.search_key(Vkey(1)).unwrap() * 3],
            9
        );
        assert_eq!(
            pattern.nnz_cols[2 + variable_ordering.search_key(Vkey(1)).unwrap() * 3],
            9
        );
        assert_eq!(
            pattern.nnz_cols[variable_ordering.search_key(Vkey(2)).unwrap() * 3],
            3
        );
        assert_eq!(
            pattern.nnz_cols[1 + variable_ordering.search_key(Vkey(2)).unwrap() * 3],
            3
        );
        assert_eq!(
            pattern.nnz_cols[2 + variable_ordering.search_key(Vkey(2)).unwrap() * 3],
            3
        );
    }
    #[test]
    fn jacobian_sparsity_2() {
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
        let pattern = construct_jacobian_sparsity(&factors, &variables, &variable_ordering);
        assert_eq!(pattern.base.A_rows, 6);
        assert_eq!(pattern.base.A_cols, 9);
        assert_eq!(pattern.base.var_dim.len(), 3);
        assert_eq!(pattern.base.var_col.len(), 3);
        assert_eq!(pattern.base.var_dim, vec![3, 3, 3]);
        assert_eq!(pattern.base.var_col, vec![0, 3, 6]);
        assert_eq!(pattern.factor_err_row.len(), 2);
        assert_eq!(pattern.factor_err_row[0], 0);
        assert_eq!(pattern.factor_err_row[1], 3);
        assert_eq!(pattern.nnz_cols.len(), 9);
        assert_eq!(
            pattern.nnz_cols[variable_ordering.search_key(Vkey(0)).unwrap() * 3],
            3
        );
        assert_eq!(
            pattern.nnz_cols[1 + variable_ordering.search_key(Vkey(0)).unwrap() * 3],
            3
        );
        assert_eq!(
            pattern.nnz_cols[2 + variable_ordering.search_key(Vkey(0)).unwrap() * 3],
            3
        );
        assert_eq!(
            pattern.nnz_cols[variable_ordering.search_key(Vkey(1)).unwrap() * 3],
            6
        );
        assert_eq!(
            pattern.nnz_cols[1 + variable_ordering.search_key(Vkey(1)).unwrap() * 3],
            6
        );
        assert_eq!(
            pattern.nnz_cols[2 + variable_ordering.search_key(Vkey(1)).unwrap() * 3],
            6
        );
        assert_eq!(
            pattern.nnz_cols[variable_ordering.search_key(Vkey(2)).unwrap() * 3],
            3
        );
        assert_eq!(
            pattern.nnz_cols[1 + variable_ordering.search_key(Vkey(2)).unwrap() * 3],
            3
        );
        assert_eq!(
            pattern.nnz_cols[2 + variable_ordering.search_key(Vkey(2)).unwrap() * 3],
            3
        );
    }
    #[test]
    fn hessian_sparsity_0() {
        type Real = f64;
        let container = ().and_variable::<VariableA<Real>>().and_variable::<VariableB<Real>>();
        let mut variables = Variables::new(container);
        variables.add(Vkey(0), VariableA::<Real>::new(0.0));
        variables.add(Vkey(1), VariableB::<Real>::new(0.0));

        let container = ().and_factor::<FactorA<Real>>().and_factor::<FactorB<Real>>();
        let mut factors = Factors::new(container);
        factors.add(FactorA::new(1.0, None, Vkey(0), Vkey(1)));
        factors.add(FactorB::new(2.0, None, Vkey(0), Vkey(1)));
        factors.add(FactorB::new(3.0, None, Vkey(0), Vkey(1)));
        let variable_ordering = variables.default_variable_ordering();
        let pattern = construct_hessian_sparsity(
            &factors,
            &variables,
            &variable_ordering,
            HessianTriangle::Lower,
        );
        assert_eq!(pattern.base.A_rows, 9);
        assert_eq!(pattern.base.A_cols, 6);
        assert_eq!(pattern.base.var_dim.len(), 2);
        assert_eq!(pattern.base.var_col.len(), 2);
        assert_eq!(pattern.base.var_dim, vec![3, 3]);
        assert_eq!(pattern.base.var_col, vec![0, 3]);
        assert_eq!(pattern.nnz_AtA_cols.len(), 6);
        assert_eq!(pattern.nnz_AtA_cols[0], 6);
        assert_eq!(pattern.nnz_AtA_cols[1], 5);
        assert_eq!(pattern.nnz_AtA_cols[2], 4);
        assert_eq!(pattern.nnz_AtA_cols[3], 3);
        assert_eq!(pattern.nnz_AtA_cols[4], 2);
        assert_eq!(pattern.nnz_AtA_cols[5], 1);

        assert_eq!(pattern.total_nnz_AtA_cols, 21);
        assert_eq!(pattern.nnz_AtA_vars_accum[0], 0);
        assert_eq!(pattern.nnz_AtA_vars_accum[1], 15);
        assert_eq!(pattern.nnz_AtA_vars_accum[2], 21);

        assert_eq!(pattern.corl_vars.len(), 2);
        assert!(pattern.corl_vars[0].get(&1).is_some());
        assert!(pattern.corl_vars[0].get(&0).is_none());
        assert!(pattern.corl_vars[1].get(&0).is_none());
        assert!(pattern.corl_vars[1].get(&1).is_none());

        // assert_eq!(
        //     pattern.nnz_cols[0 + variable_ordering.search_key(Key(0)).unwrap() * 3],
        //     3
        // );
        // assert_eq!(
        //     pattern.nnz_cols[1 + variable_ordering.search_key(Key(0)).unwrap() * 3],
        //     3
        // );
        // assert_eq!(
        //     pattern.nnz_cols[2 + variable_ordering.search_key(Key(0)).unwrap() * 3],
        //     3
        // );
        // assert_eq!(
        //     pattern.nnz_cols[0 + variable_ordering.search_key(Key(1)).unwrap() * 3],
        //     6
        // );
        // assert_eq!(
        //     pattern.nnz_cols[1 + variable_ordering.search_key(Key(1)).unwrap() * 3],
        //     6
        // );
        // assert_eq!(
        //     pattern.nnz_cols[2 + variable_ordering.search_key(Key(1)).unwrap() * 3],
        //     6
        // );
        // assert_eq!(
        //     pattern.nnz_cols[0 + variable_ordering.search_key(Key(2)).unwrap() * 3],
        //     3
        // );
        // assert_eq!(
        //     pattern.nnz_cols[1 + variable_ordering.search_key(Key(2)).unwrap() * 3],
        //     3
        // );
        // assert_eq!(
        //     pattern.nnz_cols[2 + variable_ordering.search_key(Key(2)).unwrap() * 3],
        //     3
        // );
    }
}
