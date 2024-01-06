use faer_core::Matrix;
use nalgebra::{
    ComplexField, DMatrix, DMatrixView, DMatrixViewMut, DVector, DVectorViewMut, Dyn,
    PermutationSequence, RawStorage, RealField,
};
use nalgebra_sparse::{pattern::SparsityPattern, CscMatrix};

use crate::{
    core::{
        factor::Factor, factors::Factors, factors_container::FactorsContainer, key::Vkey,
        loss_function::GaussianLoss, variable_ordering::VariableOrdering, variables::Variables,
        variables_container::VariablesContainer, Real,
    },
    nonlinear::{
        linearization::{linearization_hessian, linearzation_jacobian},
        sparsity_pattern::{
            construct_hessian_sparsity, construct_jacobian_sparsity, HessianTriangle,
        },
    },
};
#[derive(Clone)]
#[allow(non_snake_case)]
pub struct DenseMarginalizationPriorFactor<VC, R = f64>
where
    VC: VariablesContainer<R>,
    R: Real,
{
    ordering: VariableOrdering,
    pub A_prior: DMatrix<R>,
    pub b_prior: DVector<R>,
    linearization_point: Variables<VC, R>,
}
impl<VC, R> DenseMarginalizationPriorFactor<VC, R>
where
    VC: VariablesContainer<R>,
    R: Real,
{
    #[allow(non_snake_case)]
    fn new(
        A_prior: DMatrix<R>,
        b_prior: DVector<R>,
        linearization_point: Variables<VC, R>,
        keys: &[Vkey],
    ) -> Self {
        assert_eq!(linearization_point.dim(), A_prior.ncols());
        assert_eq!(A_prior.nrows(), b_prior.len());
        DenseMarginalizationPriorFactor {
            ordering: VariableOrdering::new(keys),
            A_prior: A_prior.clone(),
            b_prior,
            linearization_point,
        }
    }
}
impl<VC, R> Factor<R> for DenseMarginalizationPriorFactor<VC, R>
where
    VC: VariablesContainer<R>,
    R: Real,
{
    type L = GaussianLoss<R>;
    type JCols = Dyn;
    type JRows = Dyn;
    #[allow(non_snake_case)]
    fn error<C>(&self, variables: &Variables<C, R>, mut error: DVectorViewMut<R>)
    where
        C: VariablesContainer<R>,
    {
        let mut dx = DVector::<R>::zeros(self.A_prior.ncols());
        variables.local(&self.linearization_point, dx.as_view_mut(), &self.ordering);
        let r = self.A_prior.clone() * dx.clone() + self.b_prior.clone();
        error.copy_from(&r);
    }
    #[allow(non_snake_case)]
    fn jacobian<C>(&self, variables: &Variables<C, R>, mut jacobian: DMatrixViewMut<R>)
    where
        C: VariablesContainer<R>,
    {
        // compute_numerical_jacobians(variables, self, &mut self.jacobians.borrow_mut());
        // println!("J0: {}", self.jacobians.borrow());
        let mut jacobians = Vec::<DMatrix<R>>::default();
        variables.retract_local_jacobian(&self.linearization_point, &mut jacobians, &self.ordering);
        let mut vars_jacobian = DMatrix::<R>::zeros(
            self.linearization_point.dim(),
            self.linearization_point.dim(),
        );
        let mut offset = 0usize;
        for i in 0..jacobians.len() {
            let key = self.ordering.key(i).unwrap();
            let dim = self.linearization_point.dim_at(key).unwrap();
            vars_jacobian
                .view_mut((offset, offset), (dim, dim))
                .copy_from(&jacobians[i]);
            offset += dim;
        }
        // println!("J {}", vars_jacobian);
        debug_assert_eq!(vars_jacobian.nrows(), self.linearization_point.dim());
        let j = self.A_prior.clone() * vars_jacobian.clone();
        jacobian.copy_from(&j);
        // println!("J1: {}", self.jacobians.borrow());
    }

    fn dim(&self) -> usize {
        self.b_prior.len()
    }

    fn keys(&self) -> &[Vkey] {
        self.ordering.keys()
    }

    fn loss_function(&self) -> Option<&Self::L> {
        None
    }

    fn jacobian_shape(&self) -> (Self::JRows, Self::JCols) {
        (Dyn(3), Dyn(3))
    }
}

pub fn try_invert_symmetric_positive_semidefinite_matrix<R>(
    assume_full_rank: bool,
    m: DMatrixView<R>,
) -> Option<DMatrix<R>>
where
    R: RealField,
{
    assert_eq!(m.nrows(), m.ncols());
    if assume_full_rank {
        todo!()
    } else {
        m.svd(true, true).pseudo_inverse(R::default_epsilon()).ok()
        // let n = m.nrows();
        // m.svd(true, true)
        //     .solve(&DMatrix::<R>::identity(n, n), R::default_epsilon())
        //     .ok()
    }
}
pub fn symmetrize_with_mean<R>(mut m: DMatrixViewMut<R>)
where
    R: Real,
{
    assert_eq!(m.nrows(), m.ncols());
    for i in 0..m.nrows() {
        for j in i + 1..m.ncols() {
            let mean = R::from_f64(0.5).unwrap() * (m[(i, j)] + m[(j, i)]);
            m[(i, j)] = mean;
            m[(j, i)] = mean;
        }
    }
}
#[allow(non_snake_case)]
pub fn reorder_eigen_decomp<R>(U: &mut DMatrix<R>, d: &mut DVector<R>)
where
    R: RealField,
{
    const VALUE_PROCESSED: usize = usize::MAX;

    // Collect the singular values with their original index, ...
    let mut i_eigenvalues = d.map_with_location(|r, _, e| (e, r));
    assert_ne!(
        i_eigenvalues.len(),
        VALUE_PROCESSED,
        "Too many singular values"
    );

    // ... sort the singular values, ...
    i_eigenvalues
        .as_mut_slice()
        .sort_unstable_by(|(a, _), (b, _)| b.partial_cmp(a).expect("Singular value was NaN"));

    // ... and store them.
    d.zip_apply(&i_eigenvalues, |value, (new_value, _)| {
        value.clone_from(&new_value)
    });

    // Calculate required permutations given the sorted indices.
    // We need to identify all circles to calculate the required swaps.
    let mut permutations = PermutationSequence::identity_generic(i_eigenvalues.data.shape().0);

    for i in 0..i_eigenvalues.len() {
        let mut index_1 = i;
        let mut index_2 = i_eigenvalues[i].1;

        // Check whether the value was already visited ...
        while index_2 != VALUE_PROCESSED // ... or a "double swap" must be avoided.
                && i_eigenvalues[index_2].1 != VALUE_PROCESSED
        {
            // Add the permutation ...
            permutations.append_permutation(index_1, index_2);
            // ... and mark the value as visited.
            i_eigenvalues[index_1].1 = VALUE_PROCESSED;

            index_1 = index_2;
            index_2 = i_eigenvalues[index_1].1;
        }
    }

    // Permute the optional components

    permutations.permute_columns(U);
}
#[allow(non_snake_case)]
pub struct EVD<R>
where
    R: RealField,
{
    /// eigen vectors
    pub U: DMatrix<R>,
    /// eigen values
    pub v: DVector<R>,
    pub condition_number: R,
    pub perturbation: R,
}
#[allow(non_snake_case)]
impl<R> EVD<R>
where
    R: RealField,
{
    fn new(U: DMatrix<R>, v: DVector<R>, condition_number: R, perturbation: R) -> Self {
        EVD {
            U,
            v,
            condition_number,
            perturbation,
        }
    }
}

#[allow(non_snake_case)]
pub fn try_eigen_value_decomposition_unsorted<R>(
    P: DMatrixView<R>,
    rank_threshold: R,
) -> Option<EVD<R>>
where
    R: RealField,
{
    let es = P.into_owned().symmetric_eigen();
    let U = es.eigenvectors;
    let v = es.eigenvalues;
    // reorder_eigen_decomp(&mut U, &mut v);
    // Compute the rank.
    let mut rank = 0_usize;
    let mut perturbation_sq = R::zero();
    for i in 0..v.len() {
        if v[i] > rank_threshold {
            rank += 1;
        } else {
            perturbation_sq += v[i].clone().powi(2);
        }
    }
    if rank == 0 {
        return None;
    }
    let U = U.columns(0, rank);
    let v = v.rows(0, rank);
    let condition_number = v.abs().max() / v.abs().min();
    let perturbation = perturbation_sq.sqrt();
    Some(EVD::<R>::new(
        U.into(),
        v.into(),
        condition_number,
        perturbation,
    ))
}
#[allow(non_snake_case)]
pub fn try_eigen_value_decomposition<R>(P: DMatrixView<R>, rank_threshold: R) -> Option<EVD<R>>
where
    R: RealField,
{
    let es = P.into_owned().symmetric_eigen();
    let mut U = es.eigenvectors;
    let mut v = es.eigenvalues;
    reorder_eigen_decomp(&mut U, &mut v);
    // Compute the rank.
    let mut rank = 0_usize;
    let mut perturbation_sq = R::zero();
    for i in 0..v.len() {
        if v[i] > rank_threshold {
            rank += 1;
        } else {
            perturbation_sq += v[i].clone().powi(2);
        }
    }
    if rank == 0 {
        return None;
    }
    let U = U.columns(0, rank);
    let v = v.rows(0, rank);
    let condition_number = v.abs().max() / v.abs().min();
    let perturbation = perturbation_sq.sqrt();
    Some(EVD::<R>::new(
        U.into(),
        v.into(),
        condition_number,
        perturbation,
    ))
}
fn rc_permutate<F>(dims: &Vec<usize>, cols: &Vec<usize>, mut p: F)
where
    F: FnMut(usize, usize, usize),
{
    assert_eq!(dims.len(), cols.len());
    let mut fcol = 0;
    for i in 0..dims.len() {
        let tdim = dims[i];
        let tcol = cols[i];
        p(fcol, tcol, tdim);
        fcol += tdim;
    }
}
#[allow(non_snake_case)]
pub fn linear_system_reorder<VC, R>(
    mut A: DMatrixViewMut<R>,
    mut b: DVectorViewMut<R>,
    variables: &Variables<VC, R>,
    src_ordering: &VariableOrdering,
    dst_ordering: &VariableOrdering,
) where
    VC: VariablesContainer<R>,
    R: Real,
{
    let src_dims = src_ordering
        .keys()
        .iter()
        .map(|k| variables.dim_at(*k).unwrap())
        .collect::<Vec<usize>>();
    let src_col = src_dims
        .iter()
        .scan(0_usize, |c, &d| {
            let c0 = *c;
            *c += d;
            Some(c0)
        })
        .collect::<Vec<usize>>();

    let mut from_keys = Vec::<Vkey>::from(src_ordering.keys());
    let mut from_idxs: Vec<usize> = (0..from_keys.len()).collect();
    let mut from_dims = src_dims.clone();
    let mut from_cols = src_col.clone();
    for (to_idx, k) in dst_ordering.keys().iter().enumerate() {
        let from_idx = from_keys[to_idx..from_idxs.len()]
            .iter()
            .position(|ki| k == ki)
            .unwrap()
            + to_idx;
        from_keys.swap(from_idx, to_idx);
        from_idxs.swap(from_idx, to_idx);
        from_dims.swap(from_idx, to_idx);
        from_cols.swap(from_idx, to_idx);
    }
    let mut row_tmp_h = DMatrix::<R>::zeros(A.nrows(), A.ncols());
    rc_permutate(&from_dims, &from_cols, |fcol, tcol, tdim| {
        row_tmp_h
            .rows_mut(fcol, tdim)
            .copy_from(&A.rows(tcol, tdim));
    });
    rc_permutate(&from_dims, &from_cols, |fcol, tcol, tdim| {
        A.columns_mut(fcol, tdim)
            .copy_from(&row_tmp_h.columns(tcol, tdim));
    });
    let mut b_tmp = DVector::<R>::zeros(b.nrows());
    rc_permutate(&from_dims, &from_cols, |fcol, tcol, tdim| {
        b_tmp.rows_mut(fcol, tdim).copy_from(&b.rows(tcol, tdim));
    });
    b.copy_from(&b_tmp);
}

#[allow(non_snake_case)]
pub fn marginalize<FC, VC, R>(
    variables_keys_to_marginalize: &[Vkey],
    factors: &Factors<FC, R>,
    variables: &Variables<VC, R>,
) -> Option<DenseMarginalizationPriorFactor<VC, R>>
where
    FC: FactorsContainer<R>,
    VC: VariablesContainer<R>,
    R: Real,
{
    let m_keys = variables_keys_to_marginalize;
    let n_keys = factors.neighborhood_variables(variables_keys_to_marginalize);
    if n_keys.is_empty() {
        println!("nothing to marginalize");
        return None;
    }
    let mn_keys = [m_keys, n_keys.as_slice()].concat();

    let mn_factors = Factors::from_connected_factors(factors, m_keys);
    let mn_variables = Variables::from_variables(variables, &mn_keys);

    let sparsity = construct_hessian_sparsity(
        &mn_factors,
        &mn_variables,
        &VariableOrdering::new(&mn_keys),
        HessianTriangle::Upper,
    );
    let mut H = Vec::<R>::with_capacity(sparsity.total_nnz_AtA_cols);
    H.resize(sparsity.total_nnz_AtA_cols, R::zero());
    let mut b = DVector::<R>::zeros(sparsity.base.A_cols);
    linearization_hessian(&mn_factors, &mn_variables, &sparsity, &mut H, &mut b);
    let csc_pattern = SparsityPattern::try_from_offsets_and_indices(
        sparsity.base.A_cols,
        sparsity.base.A_cols,
        sparsity.outer_index.clone(),
        sparsity.inner_index.clone(),
    )
    .unwrap();
    let H = CscMatrix::try_from_pattern_and_values(csc_pattern.clone(), H.clone())
        .expect("CSC data must conform to format specifications");
    let mut H = DMatrix::<R>::from(&H);
    for r in 0..H.nrows() {
        for c in r..H.nrows() {
            if r == c {
                continue;
            }
            H[(c, r)] = H[(r, c)];
        }
    }

    let m_f = 0_usize;
    let m_cnt = m_keys
        .iter()
        .map(|k| variables.dim_at(*k).unwrap())
        .sum::<usize>();
    let n_f = m_cnt;
    let n_cnt = n_keys
        .iter()
        .map(|k| variables.dim_at(*k).unwrap())
        .sum::<usize>();
    let Hm = H.view((m_f, m_f), (m_cnt, m_cnt)).to_owned();
    let Hn = H.view((n_f, n_f), (n_cnt, n_cnt)).to_owned();
    let Hmn = H.view((n_f, m_f), (n_cnt, m_cnt)).to_owned();
    let bm = b.rows(m_f, m_cnt);
    let bn = b.rows(n_f, n_cnt);

    let Hm_pinv = try_invert_symmetric_positive_semidefinite_matrix(false, Hm.as_view()).unwrap();

    let mut Ht = Hn - Hmn * Hm_pinv.clone() * Hmn.transpose();
    let bt = bn - Hmn * Hm_pinv * bm;
    symmetrize_with_mean(Ht.as_view_mut());

    let rank_threshold = R::from_f64(1e-10).unwrap();
    let evd = try_eigen_value_decomposition(Ht.as_view(), rank_threshold).unwrap();
    let U = evd.U;
    let v = evd.v;

    let mut d_sqrt = DVector::<R>::zeros(v.len());
    let mut d_sqrt_inv = DVector::<R>::zeros(v.len());
    for i in 0..d_sqrt.len() {
        let lambda = v[i];
        assert!(lambda > R::zero());
        d_sqrt[i] = ComplexField::sqrt(lambda);
        d_sqrt_inv[i] = R::one() / d_sqrt[i];
    }
    let D_sqrt = DMatrix::from_diagonal(&d_sqrt);
    let D_sqrt_inv = DMatrix::from_diagonal(&d_sqrt_inv);

    let A_prior = D_sqrt * U.transpose();
    let b_prior = D_sqrt_inv * U.transpose() * bt.clone();

    let n_variables = Variables::from_variables(variables, &n_keys);
    Some(DenseMarginalizationPriorFactor::new(
        A_prior,
        b_prior,
        n_variables,
        &n_keys,
    ))
}

pub fn add_dense_marginalize_prior_factor<FC, VC, R>(
    _variables_container: &VC,
    factors_container: FC,
) -> impl FactorsContainer<R>
where
    FC: FactorsContainer<R> + 'static,
    VC: VariablesContainer<R> + 'static,
    R: Real,
{
    factors_container.and_factor::<DenseMarginalizationPriorFactor<VC, R>>()
}
#[cfg(test)]
pub(crate) mod tests {
    use matrixcompare::assert_matrix_eq;
    use nalgebra::{DMatrix, DVector};
    use rand::{distributions::Uniform, prelude::Distribution, seq::SliceRandom};

    use crate::{
        core::{
            factor::tests::RandomBlockFactor, factors::Factors,
            factors_container::FactorsContainer, key::Vkey, variable::tests::RandomVariable,
            variable_ordering::VariableOrdering, variables::Variables,
            variables_container::VariablesContainer,
        },
        fixedlag::marginalization::linear_system_reorder,
        nonlinear::{
            linearization::linearzation_jacobian, sparsity_pattern::construct_jacobian_sparsity,
        },
    };

    #[test]
    #[allow(non_snake_case)]
    fn full_lin_sys_reorder() {
        type Real = f64;
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
        let src_ordering = variables.default_variable_ordering();
        let pattern = construct_jacobian_sparsity(&factors, &variables, &src_ordering);
        let mut A = DMatrix::<Real>::zeros(pattern.base.A_rows, pattern.base.A_cols);
        let mut b = DVector::<Real>::zeros(pattern.base.A_rows);
        linearzation_jacobian(&factors, &variables, &pattern, &mut A, &mut b);
        let mut JAtA = A.transpose() * A.clone();
        let mut JAtb = A.transpose() * b.clone();
        let mut dst_keys = Vec::from(src_ordering.keys());
        dst_keys.shuffle(&mut rng);
        let dst_ordering = VariableOrdering::new(&dst_keys);
        linear_system_reorder(
            JAtA.as_view_mut(),
            JAtb.as_view_mut(),
            &variables,
            &src_ordering,
            &dst_ordering,
        );

        let pattern = construct_jacobian_sparsity(&factors, &variables, &dst_ordering);
        let mut A = DMatrix::<Real>::zeros(pattern.base.A_rows, pattern.base.A_cols);
        let mut b = DVector::<Real>::zeros(pattern.base.A_rows);
        linearzation_jacobian(&factors, &variables, &pattern, &mut A, &mut b);
        let ref_JAtA = A.transpose() * A.clone();
        let ref_JAtb = A.transpose() * b.clone();

        assert_matrix_eq!(ref_JAtA, JAtA, comp = abs, tol = 1e-9);
        assert_matrix_eq!(ref_JAtb, JAtb, comp = abs, tol = 1e-9);
    }
    #[test]
    #[allow(non_snake_case)]
    fn partial_lin_sys_reorder() {
        type Real = f64;
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
        let src_ordering = variables.default_variable_ordering();
        let pattern = construct_jacobian_sparsity(&factors, &variables, &src_ordering);
        let mut A = DMatrix::<Real>::zeros(pattern.base.A_rows, pattern.base.A_cols);
        let mut b = DVector::<Real>::zeros(pattern.base.A_rows);
        linearzation_jacobian(&factors, &variables, &pattern, &mut A, &mut b);
        let mut JAtA = A.transpose() * A.clone();
        let mut JAtb = A.transpose() * b.clone();
        let mut dst_keys = Vec::from(src_ordering.keys());
        dst_keys.shuffle(&mut rng);
        let dst_ordering = VariableOrdering::new(&dst_keys[0..dst_keys.len() / 2]);
        linear_system_reorder(
            JAtA.as_view_mut(),
            JAtb.as_view_mut(),
            &variables,
            &src_ordering,
            &dst_ordering,
        );
        let dst_dim: usize = dst_ordering
            .keys()
            .iter()
            .map(|k| variables.dim_at(*k).unwrap())
            .sum();

        let dst_ordering = VariableOrdering::new(&dst_keys);
        let pattern = construct_jacobian_sparsity(&factors, &variables, &dst_ordering);
        let mut A = DMatrix::<Real>::zeros(pattern.base.A_rows, pattern.base.A_cols);
        let mut b = DVector::<Real>::zeros(pattern.base.A_rows);
        linearzation_jacobian(&factors, &variables, &pattern, &mut A, &mut b);
        let ref_JAtA = A.transpose() * A.clone();
        let ref_JAtb = A.transpose() * b.clone();

        assert_matrix_eq!(
            ref_JAtA.view((0, 0), (dst_dim, dst_dim)),
            JAtA.view((0, 0), (dst_dim, dst_dim)),
            comp = abs,
            tol = 1e-9
        );
        assert_matrix_eq!(
            ref_JAtb.rows(0, dst_dim),
            JAtb.rows(0, dst_dim),
            comp = abs,
            tol = 1e-9
        );
    }
}
