use nalgebra::{DMatrixViewMut, DVector, DVectorViewMut, RealField};

use crate::{core::variables::Variables, nonlinear::sparsity_pattern::HessianSparsityPattern};
use core::marker::PhantomData;

use super::{
    factor::Factor,
    factors_container::{
        get_factor, get_factor_mut, get_factor_vec, get_factor_vec_mut, FactorsContainer,
    },
    key::Vkey,
    variables_container::VariablesContainer,
    Real,
};
/// Representation of factors set with addition/removing bypassing operations.
#[derive(Clone)]
pub struct Factors<C, R = f64>
where
    C: FactorsContainer<R>,
    R: Real,
{
    container: C,
    __marker: PhantomData<R>,
}
impl<C, R> Factors<C, R>
where
    C: FactorsContainer<R>,
    R: Real,
{
    pub fn new(container: C) -> Self {
        Factors::<C, R> {
            container,
            __marker: PhantomData,
        }
    }
    /// Creates new `Factors` from subset of `factors` which connected with `keys`.
    pub fn from_connected_factors(factors: &Factors<C, R>, keys: &[Vkey]) -> Self {
        // let mut new_factors = factors.clone();
        // new_factors.retain_conneted_factors(keys);
        // new_factors

        let mut indexes = Vec::<usize>::default();
        let mut new_factors = Factors::new(factors.container.empty_clone());
        factors
            .container
            .add_connected_factor_to(&mut new_factors, keys, &mut indexes, 0);
        new_factors
    }
    /// Returns internal indexes of factors connected with `keys`.
    pub fn connected_factors_indexes(&self, keys: &[Vkey]) -> Vec<usize> {
        let mut indexes = Vec::<usize>::default();
        let mut new_factors = Factors::new(self.container.empty_clone());
        //TODO: just get indexes of connected factors
        self.container
            .add_connected_factor_to(&mut new_factors, keys, &mut indexes, 0);
        indexes
    }
    /// Returns count of factors.
    pub fn len(&self) -> usize {
        self.container.len(0)
    }
    /// No factors.
    pub fn is_empty(&self) -> bool {
        self.container.is_empty()
    }
    /// Returns sum of factors `dim`.
    pub fn dim(&self) -> usize {
        self.container.dim(0)
    }
    /// Returns `dim` of factor with index.
    pub fn dim_at(&self, index: usize) -> Option<usize> {
        self.container.dim_at(index, 0)
    }
    /// Returns `keys` of factor with index.
    pub fn keys_at(&self, index: usize) -> Option<&[Vkey]> {
        self.container.keys_at(index, 0)
    }
    /// Performs weighting (whitening transformation) of error vector and jacobians matrix
    /// of factor with index.
    pub fn weight_jacobians_error_in_place_at<VC>(
        &self,
        variables: &Variables<VC, R>,
        error: DVectorViewMut<R>,
        jacobians: DMatrixViewMut<R>,
        index: usize,
    ) where
        VC: VariablesContainer<R>,
    {
        self.container
            .weight_jacobian_error_in_place_at(variables, error, jacobians, index, 0)
    }
    /// Performs weighting (whitening transformation) of error vector matrix
    /// of factor with index.
    pub fn weight_error_in_place_at<VC>(
        &self,
        variables: &Variables<VC, R>,
        error: DVectorViewMut<R>,
        index: usize,
    ) where
        VC: VariablesContainer<R>,
    {
        self.container
            .weight_error_in_place_at(variables, error, index, 0)
    }
    /// Computes jacobian and error of factor with index
    /// of factor with index.
    pub fn jacobian_error_at<VC>(
        &self,
        variables: &Variables<VC, R>,
        jacobian: DMatrixViewMut<R>,
        error: DVectorViewMut<R>,
        index: usize,
    ) -> bool
    where
        VC: VariablesContainer<R>,
    {
        self.container
            .jacobian_error_at(variables, jacobian, error, index, 0)
    }

    // /// Returns weighted jacobian matrix and error vector of factor with index.
    // /// NOTE: probably weighted word here wrong.
    // pub fn weighted_jacobians_error_at<VC>(
    //     &self,
    //     variables: &Variables<VC, R>,
    //     index: usize,
    // ) -> Option<JacobiansErrorReturn<'_, R>>
    // where
    //     VC: VariablesContainer<R>,
    // {
    //     self.container.jacobians_error_at(variables, index, 0)
    // }

    /// Returns error vector of factor with index.
    pub fn error_at<VC>(
        &self,
        variables: &Variables<VC, R>,
        error: DVectorViewMut<R>,
        index: usize,
    ) -> bool
    where
        VC: VariablesContainer<R>,
    {
        self.container.error_at(variables, error, index, 0)
    }
    pub fn error<VC>(&self, _variables: &Variables<VC, R>) -> DVector<R>
    where
        VC: VariablesContainer<R>,
    {
        todo!()
    }
    /// Computes value of cost function:
    /// $$F(\textbf{x})=\frac{1}{2}\sum_{i=0}^N \lvert\rvert f_i(\textbf{x}) \lvert\rvert^2$$
    pub fn error_squared_norm<VC>(&self, variables: &Variables<VC, R>) -> f64
    where
        VC: VariablesContainer<R>,
    {
        let mut err_squared_norm: R = R::zero();
        for f_index in 0..self.len() {
            let mut error = DVector::<R>::zeros(self.dim_at(f_index).unwrap());
            assert!(self.error_at(variables, error.as_view_mut(), f_index));
            self.weight_error_in_place_at(variables, error.as_view_mut(), f_index);
            err_squared_norm += error.norm_squared();
        }
        0.5 * err_squared_norm.to_f64().unwrap()
    }
    pub fn add<F>(&mut self, f: F)
    where
        F: Factor<R> + 'static,
        R: RealField,
    {
        #[cfg(not(debug_assertions))]
        {
            self.container.get_mut::<F>().unwrap().push(f)
        }
        #[cfg(debug_assertions)]
        {
            self.container
                .get_mut::<F>()
                .unwrap_or_else(|| panic!("type {} should be registered in factors container. use ().and_factor::<{}>()",
                    tynm::type_name::<F>(),
                    tynm::type_name::<F>()))
                .push(f)
        }
    }
    pub fn get_vec<F>(&self) -> &Vec<F>
    where
        F: Factor<R> + 'static,
    {
        get_factor_vec(&self.container)
    }
    pub fn get<F>(&self, index: usize) -> Option<&F>
    where
        F: Factor<R> + 'static,
    {
        get_factor(&self.container, index)
    }
    pub fn get_vec_mut<F>(&mut self) -> &mut Vec<F>
    where
        F: Factor<R> + 'static,
    {
        get_factor_vec_mut(&mut self.container)
    }
    pub fn get_mut<F>(&mut self, index: usize) -> Option<&mut F>
    where
        F: Factor<R> + 'static,
    {
        get_factor_mut(&mut self.container, index)
    }
    /// Remove factors connected with `key`.
    pub fn remove_conneted_factors(&mut self, key: Vkey) -> usize {
        self.container.remove_conneted_factors(key, 0)
    }
    /// Remove factors disconnected with `key`.
    pub fn retain_conneted_factors(&mut self, keys: &[Vkey]) -> usize {
        self.container.retain_conneted_factors(keys, 0)
    }
    /// Returns type name of factor with index.
    pub fn type_name_at(&self, index: usize) -> Option<String> {
        self.container.type_name_at(index, 0)
    }
    /// Returns count of unconnected variables.
    /// Optimization will not work with any unconnected variables.
    pub fn unused_variables_count<VC>(&self, variables: &Variables<VC, R>) -> usize
    where
        VC: VariablesContainer<R>,
    {
        let mut counter = 0_usize;
        for f_idx in 0..self.len() {
            counter += self
                .keys_at(f_idx)
                .unwrap()
                .iter()
                .filter(|key| !variables.default_variable_ordering().keys().contains(key))
                .count();
        }
        counter
    }
    /// Returns keys of neighbor variables.
    /// Variables contained in factor `keys` of factor connected with key in `variables_keys`
    /// excluding duplications and self containing (keys from `variables_keys` exludes from returned result).
    pub fn neighborhood_variables(&self, variables_keys: &[Vkey]) -> Vec<Vkey> {
        let mut neighborhood = Vec::<Vkey>::new();
        for vk in variables_keys {
            for f_idx in 0..self.len() {
                let fkeys = self.keys_at(f_idx).unwrap();
                // variable connected with factor
                if fkeys.contains(vk) {
                    for fk in fkeys {
                        if fk == vk {
                            continue; //exclude self
                        }
                        //neighborhood can't be in marginalized list
                        if variables_keys.contains(fk) {
                            continue;
                        }
                        if !neighborhood.contains(fk) {
                            neighborhood.push(*fk);
                        }
                    }
                }
            }
        }
        neighborhood
    }
    pub fn linearize_hessian<VC>(
        &self,
        variables: &Variables<VC, R>,
        sparsity: &HessianSparsityPattern,
        hessian_values: &mut [R],
        gradient: &mut DVector<R>,
    ) where
        VC: VariablesContainer<R>,
    {
        self.container
            .linearize_hessian(variables, sparsity, hessian_values, gradient);
    }
}
#[cfg(test)]
mod tests {

    use nalgebra::DVector;

    use crate::core::{
        factor::{
            tests::{FactorA, FactorB},
            Factor,
        },
        factors_container::{get_factor, FactorsContainer},
        key::Vkey,
        variable::tests::{VariableA, VariableB},
        variables::Variables,
        variables_container::VariablesContainer,
    };

    use super::Factors;

    #[test]
    fn add() {
        type Real = f64;
        let container = ().and_factor::<FactorA<Real>>().and_factor::<FactorB<Real>>();
        let mut factors = Factors::new(container);
        factors.add(FactorA::new(1.0, None, Vkey(0), Vkey(1)));
        factors.add(FactorB::new(2.0, None, Vkey(0), Vkey(1)));
        let f0: &FactorA<Real> = get_factor(&factors.container, 0).unwrap();
        assert_eq!(f0.orig, DVector::<Real>::from_element(3, 1.0));
        let f1: &FactorB<Real> = get_factor(&factors.container, 0).unwrap();
        assert_eq!(f1.orig, DVector::<Real>::from_element(3, 2.0));
    }
    #[test]
    fn len() {
        type Real = f64;
        let container = ().and_factor::<FactorA<Real>>().and_factor::<FactorB<Real>>();
        let mut factors = Factors::new(container);
        factors.add(FactorA::new(1.0, None, Vkey(0), Vkey(1)));
        factors.add(FactorB::new(2.0, None, Vkey(0), Vkey(1)));
        assert_eq!(factors.len(), 2);
    }
    #[test]
    fn dim() {
        type Real = f64;
        let container = ().and_factor::<FactorA<Real>>().and_factor::<FactorB<Real>>();
        let mut factors = Factors::new(container);
        factors.add(FactorA::new(1.0, None, Vkey(0), Vkey(1)));
        factors.add(FactorB::new(2.0, None, Vkey(0), Vkey(1)));
        assert_eq!(factors.dim(), 6);
    }

    #[test]
    fn dim_at() {
        type Real = f64;
        let container = ().and_factor::<FactorA<Real>>().and_factor::<FactorB<Real>>();
        let mut factors = Factors::new(container);
        factors.add(FactorA::new(1.0, None, Vkey(0), Vkey(1)));
        factors.add(FactorB::new(2.0, None, Vkey(0), Vkey(1)));
        assert_eq!(factors.dim_at(0).unwrap(), 3);
        assert_eq!(factors.dim_at(1).unwrap(), 3);
        assert!(factors.dim_at(2).is_none());
    }
    #[test]
    fn keys_at() {
        type Real = f64;
        let container = ().and_factor::<FactorA<Real>>().and_factor::<FactorB<Real>>();
        let mut factors = Factors::new(container);
        factors.add(FactorA::new(1.0, None, Vkey(0), Vkey(1)));
        factors.add(FactorB::new(2.0, None, Vkey(0), Vkey(1)));
        let keys = vec![Vkey(0), Vkey(1)];
        assert_eq!(factors.keys_at(0).unwrap(), keys);
        assert_eq!(factors.keys_at(1).unwrap(), keys);
        assert!(factors.keys_at(4).is_none());
        assert!(factors.keys_at(5).is_none());
    }
    #[test]
    fn squered_error() {
        type Real = f64;
        let container = ().and_factor::<FactorA<Real>>().and_factor::<FactorB<Real>>();
        let mut factors = Factors::new(container);
        factors.add(FactorA::new(1.0, None, Vkey(0), Vkey(1)));
        factors.add(FactorB::new(2.0, None, Vkey(0), Vkey(1)));
        let container = ().and_variable::<VariableA<Real>>().and_variable::<VariableB<Real>>();
        let mut variables = Variables::new(container);
        variables.add(Vkey(0), VariableA::<Real>::new(0.0));
        variables.add(Vkey(1), VariableB::<Real>::new(0.0));
        let mut error = DVector::<Real>::zeros(3);
        FactorA::new(1.0, None, Vkey(0), Vkey(1)).error(&variables, error.as_view_mut());
        assert_eq!(error.norm_squared(), 3.0);
        assert_eq!(factors.error_squared_norm(&variables), 7.5);
    }
    #[test]
    fn is_empty() {
        type Real = f64;
        let container = ().and_factor::<FactorA<Real>>().and_factor::<FactorB<Real>>();
        let mut factors = Factors::new(container);
        assert!(factors.is_empty());
        factors.add(FactorA::new(1.0, None, Vkey(0), Vkey(1)));
        factors.add(FactorB::new(2.0, None, Vkey(0), Vkey(1)));
        assert!(!factors.is_empty());
    }
}
