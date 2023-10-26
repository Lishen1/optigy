use crate::core::key::Vkey;
use crate::core::variable::Variable;
use crate::core::variable_ordering::VariableOrdering;
use crate::core::variables_container::{get_variable, get_variable_mut, VariablesContainer};
use hashbrown::HashMap;
use nalgebra::{DVector, DVectorView, RealField};
use num::Float;

use std::marker::PhantomData;

use super::factors::Factors;
use super::factors_container::FactorsContainer;
use super::variables_container::{get_map, get_map_mut};

#[derive(Clone)]
pub struct Variables<C, R = f64>
where
    C: VariablesContainer<R>,
    R: RealField + Float,
{
    // pub(crate) container: C,
    pub container: C,
    phantom: PhantomData<R>,
}

impl<C, R> Variables<C, R>
where
    C: VariablesContainer<R>,
    R: RealField + Float,
{
    pub fn new(container: C) -> Self {
        Variables::<C, R> {
            container,
            phantom: PhantomData,
        }
    }
    pub fn dim(&self) -> usize {
        self.container.dim(0)
    }

    pub fn len(&self) -> usize {
        self.container.len(0)
    }
    pub fn is_empty(&self) -> bool {
        self.container.is_empty()
    }

    pub fn retract(&mut self, delta: DVectorView<R>, variable_ordering: &VariableOrdering) {
        debug_assert_eq!(delta.nrows(), self.dim());
        let mut d: usize = 0;
        for i in 0..variable_ordering.len() {
            let key = variable_ordering.key(i).unwrap();
            d = self.container.retract(delta, key, d);
        }
    }
    pub fn retracted(&self, delta: DVectorView<R>, variable_ordering: &VariableOrdering) -> Self {
        debug_assert_eq!(delta.nrows(), self.dim());
        let mut variables = self.clone();
        let mut d: usize = 0;
        for i in 0..variable_ordering.len() {
            let key = variable_ordering.key(i).unwrap();
            d = variables.container.retract(delta, key, d);
        }
        variables
    }

    //TODO: return DVectorView
    pub fn local<VC>(
        &self,
        variables: &Variables<VC, R>,
        variable_ordering: &VariableOrdering,
    ) -> DVector<R>
    where
        VC: VariablesContainer<R>,
    {
        let mut delta = DVector::<R>::zeros(variables.dim());
        let mut d: usize = 0;
        for i in 0..variable_ordering.len() {
            let key = variable_ordering.key(i).unwrap();
            d = self.container.local(variables, delta.as_view_mut(), key, d);
        }
        debug_assert_eq!(delta.nrows(), d);
        delta
    }

    pub fn default_variable_ordering(&self) -> VariableOrdering {
        // VariableOrdering::new(&self.container.keys(Vec::new()))

        // need sort to repeated results
        // due to undetermined hashmap ordering
        let mut keys = self.container.keys(Vec::new());
        keys.sort();
        VariableOrdering::new(&keys)
    }

    pub fn get_map<V>(&self) -> &HashMap<Vkey, V>
    where
        V: Variable<R> + 'static,
    {
        get_map(&self.container)
    }

    pub fn get<V>(&self, key: Vkey) -> Option<&V>
    where
        V: Variable<R> + 'static,
    {
        get_variable(&self.container, key)
    }

    pub fn get_map_mut<V>(&mut self) -> &mut HashMap<Vkey, V>
    where
        V: Variable<R> + 'static,
    {
        get_map_mut(&mut self.container)
    }

    pub fn get_mut<V>(&mut self, key: Vkey) -> Option<&mut V>
    where
        V: Variable<R> + 'static,
    {
        get_variable_mut(&mut self.container, key)
    }

    pub fn add<V>(&mut self, key: Vkey, var: V)
    where
        V: Variable<R> + 'static,
    {
        #[cfg(not(debug_assertions))]
        {
            self.container.get_mut::<V>().unwrap().insert(key, var);
        }
        #[cfg(debug_assertions)]
        {
            self.container
            .get_mut::<V>()
            .unwrap_or_else(|| panic!("type {} should be registered in variables container. use ().and_variable::<{}>()",
                   tynm::type_name::<V>(),
                   tynm::type_name::<V>()))
            .insert(key, var);
        }
    }
    pub fn remove<FC>(&mut self, key: Vkey, factors: &mut Factors<FC, R>) -> (bool, usize)
    where
        FC: FactorsContainer<R>,
    {
        (
            self.container.remove(key),
            factors.remove_conneted_factors(key),
        )
    }
    pub fn dim_at(&self, key: Vkey) -> Option<usize> {
        self.container.dim_at(key)
    }
    pub fn from_variables(variables: &Self, keys: &[Vkey]) -> Self {
        let mut new_variables = Variables::new(variables.container.empty_clone());
        for key in keys {
            variables
                .container
                .add_variable_to(&mut new_variables, *key);
        }
        new_variables
    }
    pub fn type_name_at(&self, key: Vkey) -> Option<String> {
        self.container.type_name_at(key)
    }
}
#[cfg(test)]
mod tests {
    use matrixcompare::assert_matrix_eq;

    use crate::core::variable::tests::{VariableA, VariableB};

    use super::*;

    #[test]
    fn add_variable() {
        type Real = f64;
        let container = ().and_variable::<VariableA<Real>>().and_variable::<VariableB<Real>>();
        let mut variables = Variables::new(container);
        variables.add(Vkey(0), VariableA::<Real>::new(0.0));
        variables.add(Vkey(1), VariableB::<Real>::new(0.0));
    }

    #[test]
    fn get_variable() {
        type Real = f64;
        let container = ().and_variable::<VariableA<Real>>().and_variable::<VariableB<Real>>();
        let mut variables = Variables::new(container);
        variables.add(Vkey(0), VariableA::<Real>::new(1.0));
        variables.add(Vkey(1), VariableB::<Real>::new(2.0));
        let _var_0: &VariableA<_> = variables.get(Vkey(0)).unwrap();
        let _var_1: &VariableB<_> = variables.get(Vkey(1)).unwrap();
    }
    #[test]
    fn get_mut_variable() {
        type Real = f64;
        let container = ().and_variable::<VariableA<Real>>().and_variable::<VariableB<Real>>();
        let mut variables = Variables::new(container);
        variables.add(Vkey(0), VariableA::<Real>::new(0.0));
        variables.add(Vkey(1), VariableB::<Real>::new(0.0));
        {
            let var_0: &mut VariableA<_> = variables.get_mut(Vkey(0)).unwrap();
            var_0.val.fill(1.0);
        }
        {
            let var_1: &mut VariableB<_> = variables.get_mut(Vkey(1)).unwrap();
            var_1.val.fill(2.0);
        }
        let var_0: &VariableA<_> = variables.get(Vkey(0)).unwrap();
        let var_1: &VariableB<_> = variables.get(Vkey(1)).unwrap();
        assert_eq!(var_0.val, DVector::<Real>::from_element(3, 1.0));
        assert_eq!(var_1.val, DVector::<Real>::from_element(3, 2.0));
    }
    #[test]
    fn dim_at() {
        type Real = f64;
        let container = ().and_variable::<VariableA<Real>>().and_variable::<VariableB<Real>>();
        let mut variables = Variables::new(container);
        variables.add(Vkey(0), VariableA::<Real>::new(0.0));
        variables.add(Vkey(1), VariableB::<Real>::new(0.0));
        {
            let var_0: &mut VariableA<_> = variables.get_mut(Vkey(0)).unwrap();
            var_0.val.fill(1.0);
        }
        {
            let var_1: &mut VariableB<_> = variables.get_mut(Vkey(1)).unwrap();
            var_1.val.fill(2.0);
        }
        assert_eq!(variables.dim_at(Vkey(0)).unwrap(), 3);
        assert_eq!(variables.dim_at(Vkey(1)).unwrap(), 3);
    }
    #[test]
    fn local() {
        type Real = f64;
        let container = ().and_variable::<VariableA<Real>>().and_variable::<VariableB<Real>>();
        let mut variables = Variables::new(container);
        variables.add(Vkey(0), VariableA::<Real>::new(0.0));
        variables.add(Vkey(1), VariableB::<Real>::new(0.0));
        let orig_variables = variables.clone();
        {
            let var_0: &mut VariableA<_> = variables.get_mut(Vkey(0)).unwrap();
            var_0.val.fill(1.0);
        }
        {
            let var_1: &mut VariableB<_> = variables.get_mut(Vkey(1)).unwrap();
            var_1.val.fill(2.0);
        }
        let var_0: &VariableA<_> = variables.get(Vkey(0)).unwrap();
        let var_1: &VariableB<_> = variables.get(Vkey(1)).unwrap();
        assert_eq!(var_0.val, DVector::<Real>::from_element(3, 1.0));
        assert_eq!(var_1.val, DVector::<Real>::from_element(3, 2.0));

        let ordering = variables.default_variable_ordering();
        let delta = variables.local(&orig_variables, &ordering);

        let dim_0 = variables.get::<VariableA<_>>(Vkey(0)).unwrap().dim();
        let dim_1 = variables.get::<VariableB<_>>(Vkey(1)).unwrap().dim();

        assert_matrix_eq!(
            DVector::<Real>::from_element(dim_1, 1.0),
            delta.rows(ordering.search_key(Vkey(0)).unwrap(), dim_0)
        );
        assert_matrix_eq!(
            DVector::<Real>::from_element(dim_0, 2.0),
            delta.rows(ordering.search_key(Vkey(1)).unwrap() * dim_0, dim_1)
        );
    }

    #[test]
    fn retract() {
        type Real = f64;
        let container = ().and_variable::<VariableA<Real>>().and_variable::<VariableB<Real>>();
        let mut variables = Variables::new(container);
        variables.add(Vkey(0), VariableA::<Real>::new(0.0));
        variables.add(Vkey(1), VariableB::<Real>::new(0.0));
        let mut delta = DVector::<Real>::zeros(variables.dim());
        let dim_0 = variables.get::<VariableA<_>>(Vkey(0)).unwrap().dim();
        let dim_1 = variables.get::<VariableB<_>>(Vkey(1)).unwrap().dim();
        delta.fill(5.0);
        delta.fill(1.0);
        println!("delta {:?}", delta);
        let variable_ordering = variables.default_variable_ordering(); // reversed
        variables.retract(delta.as_view(), &variable_ordering);
        let v0: &VariableA<Real> = variables.get(Vkey(0)).unwrap();
        let v1: &VariableB<Real> = variables.get(Vkey(1)).unwrap();
        assert_eq!(v1.val, delta.rows(0, dim_0));
        assert_eq!(v0.val, delta.rows(dim_0, dim_1));
    }

    #[test]
    fn dim() {
        type Real = f64;
        let container = ().and_variable::<VariableA<Real>>().and_variable::<VariableB<Real>>();
        let mut variables = Variables::new(container);
        variables.add(Vkey(0), VariableA::<Real>::new(0.0));
        variables.add(Vkey(1), VariableB::<Real>::new(0.0));
        assert_eq!(variables.dim(), 6);
    }
    #[test]
    fn len() {
        type Real = f64;
        let container = ().and_variable::<VariableA<Real>>().and_variable::<VariableB<Real>>();
        let mut variables = Variables::new(container);
        variables.add(Vkey(0), VariableA::<Real>::new(0.0));
        variables.add(Vkey(1), VariableB::<Real>::new(0.0));
        assert_eq!(variables.len(), 2);
    }
    #[test]
    fn is_empty() {
        type Real = f64;
        let container = ().and_variable::<VariableA<Real>>().and_variable::<VariableB<Real>>();
        let mut variables = Variables::new(container);
        assert!(variables.is_empty());
        variables.add(Vkey(0), VariableA::<Real>::new(0.0));
        variables.add(Vkey(1), VariableB::<Real>::new(0.0));
        assert!(!variables.is_empty());
    }
}
