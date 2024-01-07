use crate::core::factor::Factor;
use crate::nonlinear::linearization::linearize_hessian_single_factor;
use crate::nonlinear::sparsity_pattern::HessianSparsityPattern;
use core::any::TypeId;

use core::mem;
use nalgebra::allocator::Allocator;
use nalgebra::{DMatrixViewMut, DVectorViewMut, DefaultAllocator};

use super::factors::Factors;
use super::key::Vkey;
use super::loss_function::LossFunction;
use super::variables::Variables;
use super::variables_container::VariablesContainer;
use super::Real;
use nalgebra::DVector;

pub trait FactorsKey<T = f64>: Clone
where
    T: Real,
{
    type Value: 'static + Factor<T>;
}

/// The building block trait for recursive variadics.
pub trait FactorsContainer<T = f64>: Clone + Default
where
    T: Real,
{
    /// Try to get the value for N.
    fn get<N: FactorsKey<T>>(&self) -> Option<&Vec<N::Value>>;
    /// Try to get the value for N mutably.
    fn get_mut<N: FactorsKey<T>>(&mut self) -> Option<&mut Vec<N::Value>>;
    /// Add the default value for N
    fn and_factor<N: FactorsKey<T>>(self) -> FactorsEntry<N, Self, T>
    where
        Self: Sized,
        N::Value: FactorsKey<T>,
    {
        match self.get::<N::Value>() {
            Some(_) => panic!(
                "type {} already present in FactorsContainer",
                tynm::type_name::<N::Value>()
            ),
            None => FactorsEntry {
                data: Vec::<N::Value>::default(),
                parent: self,
            },
        }
    }
    /// sum of factors dim
    fn dim(&self, init: usize) -> usize;
    /// sum of factors vecs len
    fn len(&self, init: usize) -> usize;
    fn is_empty(&self) -> bool;
    /// factor dim by index
    fn dim_at(&self, index: usize, init: usize) -> Option<usize>;
    /// factor keys by index
    fn keys_at(&self, index: usize, init: usize) -> Option<&[Vkey]>;
    /// factor jacobians error by index
    fn jacobian_error_at<C>(
        &self,
        variables: &Variables<C, T>,
        jacobian: DMatrixViewMut<T>,
        error: DVectorViewMut<T>,
        index: usize,
        init: usize,
    ) -> bool
    where
        C: VariablesContainer<T>;
    /// weight factor error and jacobians in-place
    fn weight_jacobian_error_in_place_at<C>(
        &self,
        variables: &Variables<C, T>,
        error: DVectorViewMut<T>,
        jacobians: DMatrixViewMut<T>,
        index: usize,
        init: usize,
    ) where
        C: VariablesContainer<T>;
    /// weight factor error in-place
    fn weight_error_in_place_at<C>(
        &self,
        variables: &Variables<C, T>,
        error: DVectorViewMut<T>,
        index: usize,
        init: usize,
    ) where
        C: VariablesContainer<T>;
    /// factor weighted error by index
    fn error_at<C>(
        &self,
        variables: &Variables<C, T>,
        error: DVectorViewMut<T>,
        index: usize,
        init: usize,
    ) -> bool
    where
        C: VariablesContainer<T>;
    /// Returns `true` if factor with index has custom loss
    fn has_loss_at<C>(&self, variables: &Variables<C, T>, index: usize, init: usize) -> bool
    where
        C: VariablesContainer<T>;
    /// factor type name used for debugging
    fn type_name_at(&self, index: usize, init: usize) -> Option<String>;
    /// Remove factors connected with variable with key
    /// # Arguments
    /// * `key` - A key of variable to remove
    /// * `init` - An initial value for removed factors counter
    /// # Returns
    /// Removed factors count
    fn remove_conneted_factors(&mut self, key: Vkey, init: usize) -> usize;

    /// Remove factors not connected with variables with keys
    /// # Arguments
    /// * `keys` - A keys of variables to retain
    /// * `init` - An initial value for removed factors counter
    /// # Returns
    /// Removed factors count
    fn retain_conneted_factors(&mut self, keys: &[Vkey], init: usize) -> usize;
    fn empty_clone(&self) -> Self;
    fn add_connected_factor_to<C>(
        &self,
        factors: &mut Factors<C, T>,
        keys: &[Vkey],
        indexes: &mut Vec<usize>,
        init: usize,
    ) where
        C: FactorsContainer<T>;
    fn linearize_hessian<C>(
        &self,
        variables: &Variables<C, T>,
        sparsity: &HessianSparsityPattern,
        hessian_values: &mut [T],
        gradient: &mut DVector<T>,
    ) where
        C: VariablesContainer<T>;
}

/// The base case for recursive variadics: no fields.
pub type FactorsEmpty = ();
impl<T> FactorsContainer<T> for FactorsEmpty
where
    T: Real,
{
    fn get<N: FactorsKey<T>>(&self) -> Option<&Vec<N::Value>> {
        None
    }
    fn get_mut<N: FactorsKey<T>>(&mut self) -> Option<&mut Vec<N::Value>> {
        None
    }
    fn dim(&self, init: usize) -> usize {
        init
    }
    fn len(&self, init: usize) -> usize {
        init
    }
    fn is_empty(&self) -> bool {
        true
    }
    fn dim_at(&self, _index: usize, _init: usize) -> Option<usize> {
        None
    }
    fn keys_at(&self, _index: usize, _init: usize) -> Option<&[Vkey]> {
        None
    }

    fn jacobian_error_at<C>(
        &self,
        _variables: &Variables<C, T>,
        _jacobian: DMatrixViewMut<T>,
        _error: DVectorViewMut<T>,
        _index: usize,
        _init: usize,
    ) -> bool
    where
        C: VariablesContainer<T>,
    {
        false
    }
    fn weight_jacobian_error_in_place_at<C>(
        &self,
        _variables: &Variables<C, T>,
        _error: DVectorViewMut<T>,
        _jacobians: DMatrixViewMut<T>,
        _index: usize,
        _init: usize,
    ) where
        C: VariablesContainer<T>,
    {
    }
    fn weight_error_in_place_at<C>(
        &self,
        _variables: &Variables<C, T>,
        _error: DVectorViewMut<T>,
        _index: usize,
        _init: usize,
    ) where
        C: VariablesContainer<T>,
    {
    }
    fn error_at<C>(
        &self,
        _variables: &Variables<C, T>,
        _error: DVectorViewMut<T>,
        _index: usize,
        _init: usize,
    ) -> bool
    where
        C: VariablesContainer<T>,
    {
        false
    }

    fn type_name_at(&self, _index: usize, _init: usize) -> Option<String> {
        None
    }

    fn remove_conneted_factors(&mut self, _key: Vkey, init: usize) -> usize {
        init
    }

    fn retain_conneted_factors(&mut self, _keys: &[Vkey], init: usize) -> usize {
        init
    }
    fn empty_clone(&self) -> Self {}
    fn add_connected_factor_to<C>(
        &self,
        _factors: &mut Factors<C, T>,
        _keys: &[Vkey],
        _indexes: &mut Vec<usize>,
        _init: usize,
    ) where
        C: FactorsContainer<T>,
    {
    }

    fn has_loss_at<C>(&self, _variables: &Variables<C, T>, _index: usize, _init: usize) -> bool
    where
        C: VariablesContainer<T>,
    {
        false
    }

    fn linearize_hessian<C>(
        &self,
        _variables: &Variables<C, T>,
        _sparsity: &HessianSparsityPattern,
        _hessian_values: &mut [T],
        _gradient: &mut DVector<T>,
    ) where
        C: VariablesContainer<T>,
    {
    }
}

/// Wraps some field data and a parent, which is either another Entry or Empty
#[derive(Clone)]
pub struct FactorsEntry<K, P, T>
where
    K: FactorsKey<T>,
    T: Real,
{
    data: Vec<K::Value>,
    parent: P,
}
impl<K, P, T> Default for FactorsEntry<K, P, T>
where
    K: FactorsKey<T>,
    P: FactorsContainer<T> + Default,
    T: Real,
{
    fn default() -> Self {
        FactorsEntry::<K, P, T> {
            data: Vec::<K::Value>::default(),
            parent: P::default(),
        }
    }
}

impl<K, P, T> FactorsContainer<T> for FactorsEntry<K, P, T>
where
    K: FactorsKey<T>,
    P: FactorsContainer<T> + Default,
    T: Real,
    DefaultAllocator: Allocator<T, <<K as FactorsKey<T>>::Value as Factor<T>>::JRows>,
    DefaultAllocator: Allocator<T, <<K as FactorsKey<T>>::Value as Factor<T>>::JCols>,
    DefaultAllocator: Allocator<
        T,
        <<K as FactorsKey<T>>::Value as Factor<T>>::JRows,
        <<K as FactorsKey<T>>::Value as Factor<T>>::JCols,
    >,
    DefaultAllocator: Allocator<
        T,
        <<K as FactorsKey<T>>::Value as Factor<T>>::JCols,
        <<K as FactorsKey<T>>::Value as Factor<T>>::JRows,
    >,
    DefaultAllocator: Allocator<
        T,
        <<K as FactorsKey<T>>::Value as Factor<T>>::JCols,
        <<K as FactorsKey<T>>::Value as Factor<T>>::JCols,
    >,
{
    fn get<N: FactorsKey<T>>(&self) -> Option<&Vec<N::Value>> {
        if TypeId::of::<N::Value>() == TypeId::of::<K::Value>() {
            Some(unsafe { mem::transmute(&self.data) })
        } else {
            self.parent.get::<N>()
        }
    }
    fn get_mut<N: FactorsKey<T>>(&mut self) -> Option<&mut Vec<N::Value>> {
        if TypeId::of::<N::Value>() == TypeId::of::<K::Value>() {
            Some(unsafe { mem::transmute(&mut self.data) })
        } else {
            self.parent.get_mut::<N>()
        }
    }
    fn dim(&self, init: usize) -> usize {
        let mut d = init;
        for f in self.data.iter() {
            d += f.dim();
        }
        self.parent.dim(d)
    }
    fn len(&self, init: usize) -> usize {
        let l = init + self.data.len();
        self.parent.len(l)
    }
    fn is_empty(&self) -> bool {
        if self.data.is_empty() {
            self.parent.is_empty()
        } else {
            false
        }
    }
    fn dim_at(&self, index: usize, init: usize) -> Option<usize> {
        if (init..(init + self.data.len())).contains(&index) {
            Some(self.data[index - init].dim())
        } else {
            self.parent.dim_at(index, init + self.data.len())
        }
    }
    fn keys_at(&self, index: usize, init: usize) -> Option<&[Vkey]> {
        if (init..(init + self.data.len())).contains(&index) {
            Some(self.data[index - init].keys())
        } else {
            self.parent.keys_at(index, init + self.data.len())
        }
    }
    fn jacobian_error_at<C>(
        &self,
        variables: &Variables<C, T>,
        jacobian: DMatrixViewMut<T>,
        error: DVectorViewMut<T>,
        index: usize,
        init: usize,
    ) -> bool
    where
        C: VariablesContainer<T>,
    {
        if (init..(init + self.data.len())).contains(&index) {
            self.data[index - init].jacobian_error(variables, jacobian, error);
            true
        } else {
            self.parent
                .jacobian_error_at(variables, jacobian, error, index, init + self.data.len())
        }
    }
    fn weight_jacobian_error_in_place_at<C>(
        &self,
        variables: &Variables<C, T>,
        error: DVectorViewMut<T>,
        jacobians: DMatrixViewMut<T>,
        index: usize,
        init: usize,
    ) where
        C: VariablesContainer<T>,
    {
        if (init..(init + self.data.len())).contains(&index) {
            let loss = self.data[index - init].loss_function();
            if let Some(loss) = loss {
                loss.weight_jacobians_error_in_place(error, jacobians);
            }
        } else {
            self.parent.weight_jacobian_error_in_place_at(
                variables,
                error,
                jacobians,
                index,
                init + self.data.len(),
            )
        }
    }
    fn weight_error_in_place_at<C>(
        &self,
        variables: &Variables<C, T>,
        error: DVectorViewMut<T>,
        index: usize,
        init: usize,
    ) where
        C: VariablesContainer<T>,
    {
        if (init..(init + self.data.len())).contains(&index) {
            let loss = self.data[index - init].loss_function();
            if let Some(loss) = loss {
                loss.weight_error_in_place(error);
            }
        } else {
            self.parent
                .weight_error_in_place_at(variables, error, index, init + self.data.len())
        }
    }
    fn error_at<C>(
        &self,
        variables: &Variables<C, T>,
        error: DVectorViewMut<T>,
        index: usize,
        init: usize,
    ) -> bool
    where
        C: VariablesContainer<T>,
    {
        if (init..(init + self.data.len())).contains(&index) {
            self.data[index - init].error(variables, error);
            true
        } else {
            self.parent
                .error_at(variables, error, index, init + self.data.len())
        }
    }

    fn has_loss_at<C>(&self, variables: &Variables<C, T>, index: usize, init: usize) -> bool
    where
        C: VariablesContainer<T>,
    {
        if (init..(init + self.data.len())).contains(&index) {
            self.data[index - init].loss_function().is_some()
        } else {
            self.parent
                .has_loss_at(variables, index, init + self.data.len())
        }
    }

    fn type_name_at(&self, index: usize, init: usize) -> Option<String> {
        if (init..(init + self.data.len())).contains(&index) {
            Some(tynm::type_name::<K::Value>())
        } else {
            self.parent.type_name_at(index, init + self.data.len())
        }
    }

    fn remove_conneted_factors(&mut self, key: Vkey, init: usize) -> usize {
        let removed = self.data.len();
        self.data.retain_mut(|f| {
            // if f.keys().contains(&key) {
            //     f.on_variable_remove(key)
            // } else {
            //     true
            // }
            !f.keys().contains(&key)
        });
        let removed = removed - self.data.len();
        self.parent.remove_conneted_factors(key, removed + init)
    }

    fn retain_conneted_factors(&mut self, keys: &[Vkey], init: usize) -> usize {
        let removed = self.data.len();
        self.data
            .retain(|f| keys.iter().any(|key| f.keys().contains(key)));
        let removed = removed - self.data.len();
        self.parent.retain_conneted_factors(keys, removed + init)
    }
    fn empty_clone(&self) -> Self {
        Self::default()
    }

    fn add_connected_factor_to<C>(
        &self,
        factors: &mut Factors<C, T>,
        keys: &[Vkey],
        indexes: &mut Vec<usize>,
        init: usize,
    ) where
        C: FactorsContainer<T>,
    {
        for (i, f) in self.data.iter().enumerate() {
            let index = i + init;
            if indexes.contains(&index) {
                continue;
            }
            if keys.iter().any(|key| f.keys().contains(key)) {
                factors.add(f.clone());
                indexes.push(index);
            }
        }
        self.parent
            .add_connected_factor_to(factors, keys, indexes, init + self.data.len())
    }

    fn linearize_hessian<C>(
        &self,
        variables: &Variables<C, T>,
        sparsity: &HessianSparsityPattern,
        hessian_values: &mut [T],
        gradient: &mut DVector<T>,
    ) where
        C: VariablesContainer<T>,
    {
        for f in &self.data {
            linearize_hessian_single_factor(f, variables, sparsity, hessian_values, gradient);
        }
        self.parent
            .linearize_hessian(variables, sparsity, hessian_values, gradient);
    }
}

impl<K, T> FactorsKey<T> for K
where
    K: 'static + Factor<T>,
    T: Real,
{
    type Value = K;
}

pub fn get_factor_vec<C, F, T>(container: &C) -> &Vec<F>
where
    C: FactorsContainer<T>,
    F: Factor<T> + 'static,
    T: Real,
{
    #[cfg(not(debug_assertions))]
    {
        container.get::<F>().unwrap()
    }
    #[cfg(debug_assertions)]
    {
        container.get::<F>().unwrap_or_else(|| {
            panic!(
                "type {} should be registered in factors container. use ().and_factor::<{}>()",
                tynm::type_name::<F>(),
                tynm::type_name::<F>()
            )
        })
    }
}
pub fn get_factor<C, F, T>(container: &C, index: usize) -> Option<&F>
where
    C: FactorsContainer<T>,
    F: Factor<T> + 'static,
    T: Real,
{
    get_factor_vec(container).get(index)
}
pub fn get_factor_vec_mut<C, F, T>(container: &mut C) -> &mut Vec<F>
where
    C: FactorsContainer<T>,
    F: Factor<T> + 'static,
    T: Real,
{
    #[cfg(not(debug_assertions))]
    {
        container.get_mut::<F>().unwrap()
    }
    #[cfg(debug_assertions)]
    {
        container.get_mut::<F>().unwrap_or_else(|| {
            panic!(
                "type {} should be registered in factors container. use ().and_factor::<{}>()",
                tynm::type_name::<F>(),
                tynm::type_name::<F>()
            )
        })
    }
}
pub fn get_factor_mut<C, F, T>(container: &mut C, index: usize) -> Option<&mut F>
where
    C: FactorsContainer<T>,
    F: Factor<T> + 'static,
    T: Real,
{
    get_factor_vec_mut(container).get_mut(index)
}
#[cfg(test)]
pub(crate) mod tests {

    use nalgebra::{DMatrix, DVector};

    use crate::core::{
        factor::tests::{FactorA, FactorB},
        factors_container::{get_factor, get_factor_mut, FactorsContainer},
        key::Vkey,
        variable::tests::{VariableA, VariableB},
        variables::Variables,
        variables_container::VariablesContainer,
    };

    #[test]
    fn make() {
        type Real = f64;
        let container = ().and_factor::<FactorA<Real>>().and_factor::<FactorB<Real>>();
        let _fc0 = container.get::<FactorA<Real>>().unwrap();
        let _fc1 = container.get::<FactorB<Real>>().unwrap();
    }
    #[test]
    fn get() {
        type Real = f64;
        let mut container = ().and_factor::<FactorA<Real>>().and_factor::<FactorB<Real>>();
        {
            let fc0 = container.get_mut::<FactorA<Real>>().unwrap();
            fc0.push(FactorA::new(2.0, None, Vkey(0), Vkey(1)));
            fc0.push(FactorA::new(1.0, None, Vkey(0), Vkey(1)));
        }
        {
            let fc1 = container.get_mut::<FactorB<Real>>().unwrap();
            fc1.push(FactorB::new(2.0, None, Vkey(0), Vkey(1)));
        }
        let fc0 = container.get::<FactorA<Real>>().unwrap();
        assert_eq!(
            fc0.get(0).unwrap().orig,
            DVector::<Real>::from_element(3, 2.0)
        );
        assert_eq!(
            fc0.get(1).unwrap().orig,
            DVector::<Real>::from_element(3, 1.0)
        );
        let fc1 = container.get::<FactorB<Real>>().unwrap();
        assert_eq!(
            fc1.get(0).unwrap().orig,
            DVector::<Real>::from_element(3, 2.0)
        );
        let f0: &FactorA<_> = get_factor(&container, 0).unwrap();
        let f1: &FactorA<_> = get_factor(&container, 1).unwrap();
        assert_eq!(f0.orig, DVector::<Real>::from_element(3, 2.0));
        assert_eq!(f1.orig, DVector::<Real>::from_element(3, 1.0));
    }
    #[test]
    fn get_mut() {
        type Real = f64;
        let mut container = ().and_factor::<FactorA<Real>>().and_factor::<FactorB<Real>>();
        {
            let fc0 = container.get_mut::<FactorA<Real>>().unwrap();
            fc0.push(FactorA::new(2.0, None, Vkey(0), Vkey(1)));
            fc0.push(FactorA::new(1.0, None, Vkey(0), Vkey(1)));
        }
        {
            let fc1 = container.get_mut::<FactorB<Real>>().unwrap();
            fc1.push(FactorB::new(2.0, None, Vkey(0), Vkey(1)));
        }
        {
            let f: &mut FactorA<_> = get_factor_mut(&mut container, 0).unwrap();
            f.orig = DVector::<Real>::from_element(3, 3.0);
            let f: &mut FactorA<_> = get_factor_mut(&mut container, 1).unwrap();
            f.orig = DVector::<Real>::from_element(3, 4.0);
        }
        {
            let f: &mut FactorB<_> = get_factor_mut(&mut container, 0).unwrap();
            f.orig = DVector::<Real>::from_element(3, 5.0);
        }
        let fc0 = container.get::<FactorA<Real>>().unwrap();
        assert_eq!(
            fc0.get(0).unwrap().orig,
            DVector::<Real>::from_element(3, 3.0)
        );
        assert_eq!(
            fc0.get(1).unwrap().orig,
            DVector::<Real>::from_element(3, 4.0)
        );
        let fc1 = container.get::<FactorB<Real>>().unwrap();
        assert_eq!(
            fc1.get(0).unwrap().orig,
            DVector::<Real>::from_element(3, 5.0)
        );
        let f0: &FactorA<_> = get_factor(&container, 0).unwrap();
        let f1: &FactorA<_> = get_factor(&container, 1).unwrap();
        assert_eq!(f0.orig, DVector::<Real>::from_element(3, 3.0));
        assert_eq!(f1.orig, DVector::<Real>::from_element(3, 4.0));
    }
    #[test]
    fn len() {
        type Real = f64;
        let mut container = ().and_factor::<FactorA<Real>>().and_factor::<FactorB<Real>>();
        {
            let fc0 = container.get_mut::<FactorA<Real>>().unwrap();
            fc0.push(FactorA::new(2.0, None, Vkey(0), Vkey(1)));
            fc0.push(FactorA::new(1.0, None, Vkey(0), Vkey(1)));
        }
        {
            let fc1 = container.get_mut::<FactorB<Real>>().unwrap();
            fc1.push(FactorB::new(2.0, None, Vkey(0), Vkey(1)));
        }
        assert_eq!(container.len(0), 3);
    }
    #[test]
    fn dim_at() {
        type Real = f64;
        let mut container = ().and_factor::<FactorA<Real>>().and_factor::<FactorB<Real>>();
        {
            let fc0 = container.get_mut::<FactorA<Real>>().unwrap();
            fc0.push(FactorA::new(2.0, None, Vkey(0), Vkey(1)));
            fc0.push(FactorA::new(1.0, None, Vkey(0), Vkey(1)));
        }
        {
            let fc1 = container.get_mut::<FactorB<Real>>().unwrap();
            fc1.push(FactorB::new(2.0, None, Vkey(0), Vkey(1)));
        }
        assert_eq!(container.dim_at(0, 0).unwrap(), 3);
        assert_eq!(container.dim_at(1, 0).unwrap(), 3);
        assert_eq!(container.dim_at(2, 0).unwrap(), 3);
        assert!(container.dim_at(4, 0).is_none());
        assert!(container.dim_at(5, 0).is_none());
    }
    #[test]
    fn dim() {
        type Real = f64;
        let mut container = ().and_factor::<FactorA<Real>>().and_factor::<FactorB<Real>>();
        {
            let fc0 = container.get_mut::<FactorA<Real>>().unwrap();
            fc0.push(FactorA::new(2.0, None, Vkey(0), Vkey(1)));
            fc0.push(FactorA::new(1.0, None, Vkey(0), Vkey(1)));
        }
        {
            let fc1 = container.get_mut::<FactorB<Real>>().unwrap();
            fc1.push(FactorB::new(2.0, None, Vkey(0), Vkey(1)));
        }
        assert_eq!(container.dim(0), 9);
    }
    #[test]
    fn keys_at() {
        type Real = f64;
        let mut container = ().and_factor::<FactorA<Real>>().and_factor::<FactorB<Real>>();
        {
            let fc0 = container.get_mut::<FactorA<Real>>().unwrap();
            fc0.push(FactorA::new(2.0, None, Vkey(0), Vkey(1)));
            fc0.push(FactorA::new(1.0, None, Vkey(0), Vkey(1)));
        }
        {
            let fc1 = container.get_mut::<FactorB<Real>>().unwrap();
            fc1.push(FactorB::new(2.0, None, Vkey(0), Vkey(1)));
        }
        let keys = vec![Vkey(0), Vkey(1)];
        assert_eq!(container.keys_at(0, 0).unwrap(), keys);
        assert_eq!(container.keys_at(1, 0).unwrap(), keys);
        assert_eq!(container.keys_at(2, 0).unwrap(), keys);
        assert!(container.keys_at(4, 0).is_none());
        assert!(container.keys_at(5, 0).is_none());
    }
    #[test]
    fn jacobian_at() {
        type Real = f64;

        let container = ().and_variable::<VariableA<Real>>().and_variable::<VariableB<Real>>();
        let mut variables = Variables::new(container);
        variables.add(Vkey(0), VariableA::<Real>::new(1.0));
        variables.add(Vkey(1), VariableB::<Real>::new(2.0));

        let mut container = ().and_factor::<FactorA<Real>>().and_factor::<FactorB<Real>>();
        {
            let fc0 = container.get_mut::<FactorA<Real>>().unwrap();
            fc0.push(FactorA::new(2.0, None, Vkey(0), Vkey(1)));
            fc0.push(FactorA::new(1.0, None, Vkey(0), Vkey(1)));
        }
        {
            let fc1 = container.get_mut::<FactorB<Real>>().unwrap();
            fc1.push(FactorB::new(2.0, None, Vkey(0), Vkey(1)));
        }
        let mut jacobians = DMatrix::<Real>::zeros(3, 3 * 2);
        jacobians.column_mut(0).fill(1.0);
        jacobians.column_mut(4).fill(2.0);

        let mut comp_error = DVector::<Real>::zeros(3);
        let mut comp_jacobians = DMatrix::<Real>::zeros(3, 3 * 2);
        assert!(container.jacobian_error_at(
            &variables,
            comp_jacobians.as_view_mut(),
            comp_error.as_view_mut(),
            0,
            0
        ));
        assert_eq!(comp_jacobians, jacobians);
        assert!(container.jacobian_error_at(
            &variables,
            comp_jacobians.as_view_mut(),
            comp_error.as_view_mut(),
            1,
            0
        ));
        assert_eq!(comp_jacobians, jacobians);
        assert!(container.jacobian_error_at(
            &variables,
            comp_jacobians.as_view_mut(),
            comp_error.as_view_mut(),
            2,
            0
        ));
        assert_eq!(comp_jacobians, jacobians);
        assert!(!container.jacobian_error_at(
            &variables,
            comp_jacobians.as_view_mut(),
            comp_error.as_view_mut(),
            4,
            0
        ));
        assert!(!container.jacobian_error_at(
            &variables,
            comp_jacobians.as_view_mut(),
            comp_error.as_view_mut(),
            5,
            0
        ));
    }
    #[test]
    fn weighted_error_at() {
        type Real = f64;

        let container = ().and_variable::<VariableA<Real>>().and_variable::<VariableB<Real>>();
        let mut variables = Variables::new(container);
        variables.add(Vkey(0), VariableA::<Real>::new(1.0));
        variables.add(Vkey(1), VariableB::<Real>::new(2.0));

        let mut container = ().and_factor::<FactorA<Real>>().and_factor::<FactorB<Real>>();
        {
            let fc0 = container.get_mut::<FactorA<Real>>().unwrap();
            fc0.push(FactorA::new(2.0, None, Vkey(0), Vkey(1)));
            fc0.push(FactorA::new(1.0, None, Vkey(0), Vkey(1)));
        }
        {
            let fc1 = container.get_mut::<FactorB<Real>>().unwrap();
            fc1.push(FactorB::new(2.0, None, Vkey(0), Vkey(1)));
        }
        let mut jacobians = Vec::<DMatrix<Real>>::with_capacity(2);
        jacobians.resize_with(2, || DMatrix::zeros(3, 3));
        jacobians[0].column_mut(0).fill(1.0);
        jacobians[1].column_mut(1).fill(2.0);
        // assert_eq!(
        //     container
        //         .weighted_error_at(&variables, 0, 0)
        //         .unwrap()
        //         .deref(),
        //     container
        //         .get::<FactorA<Real>>()
        //         .unwrap()
        //         .get(0)
        //         .unwrap()
        //         .weighted_error(&variables)
        // );
    }
    #[test]
    fn is_empty() {
        type Real = f64;
        let mut container = ().and_factor::<FactorA<Real>>().and_factor::<FactorB<Real>>();
        assert!(container.is_empty());
        let fc0 = container.get_mut::<FactorA<Real>>().unwrap();
        fc0.push(FactorA::new(2.0, None, Vkey(0), Vkey(1)));
        fc0.push(FactorA::new(1.0, None, Vkey(0), Vkey(1)));
        assert!(!container.is_empty());
    }
    #[test]
    fn empty_clone() {
        type Real = f64;
        let mut container = ().and_factor::<FactorA<Real>>().and_factor::<FactorB<Real>>();
        let fc0 = container.get_mut::<FactorA<Real>>().unwrap();
        fc0.push(FactorA::new(2.0, None, Vkey(0), Vkey(1)));
        fc0.push(FactorA::new(1.0, None, Vkey(0), Vkey(1)));
        let fc1 = container.get_mut::<FactorB<Real>>().unwrap();
        fc1.push(FactorB::new(2.0, None, Vkey(0), Vkey(1)));
        fc1.push(FactorB::new(1.0, None, Vkey(0), Vkey(1)));

        let mut container2 = container.empty_clone();
        assert!(container2.is_empty());
        let fc0 = container2.get_mut::<FactorA<Real>>().unwrap();
        fc0.push(FactorA::new(2.0, None, Vkey(0), Vkey(1)));
        fc0.push(FactorA::new(1.0, None, Vkey(0), Vkey(1)));
        let fc1 = container2.get_mut::<FactorB<Real>>().unwrap();
        fc1.push(FactorB::new(2.0, None, Vkey(0), Vkey(1)));
        fc1.push(FactorB::new(1.0, None, Vkey(0), Vkey(1)));
    }
}
