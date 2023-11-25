use std::marker::PhantomData;

use crate::{
    core::{HashMap, Real},
    fixedlag::marginalization::marginalize,
    prelude::{
        ExternVkey, Factor, Factors, FactorsContainer, NonlinearOptimizationError,
        NonlinearOptimizer, OptIterate, Variable, Variables, VariablesContainer, Vkey,
    },
};
/// Representation of optimizable factor graph.
/// Implements all general operations such addition/removing
/// variables and factors, consistency checks and optimization process.
pub struct FactorGraph<FC, VC, O, R = f64>
where
    FC: FactorsContainer<R> + 'static,
    VC: VariablesContainer<R> + 'static,
    O: OptIterate<R>,
    R: Real,
{
    factors: Factors<FC, R>,
    variables: Variables<VC, R>,
    optimizer: NonlinearOptimizer<O, R>,
    variables_to_marginalize: Vec<Vkey>,
    variables_keys_counter: usize,
    external_to_internal: HashMap<ExternVkey, Vkey>,
    internal_to_external: HashMap<Vkey, ExternVkey>,
}

impl<FC, VC, O, R> FactorGraph<FC, VC, O, R>
where
    FC: FactorsContainer<R>,
    VC: VariablesContainer<R>,
    O: OptIterate<R>,
    R: Real,
{
    pub fn new(factors_container: FC, varibles_container: VC, optimizer: O) -> Self {
        let variables = Variables::new(varibles_container);
        let factors = Factors::new(factors_container);
        FactorGraph {
            factors,
            variables,
            optimizer: NonlinearOptimizer::new(optimizer),
            variables_to_marginalize: Vec::default(),
            variables_keys_counter: 0usize,
            external_to_internal: HashMap::default(),
            internal_to_external: HashMap::default(),
        }
    }
    /// Returns factors.
    pub fn factors(&self) -> &Factors<FC, R> {
        &self.factors
    }
    /// Returns variables
    pub fn variables(&self) -> &Variables<VC, R> {
        &self.variables
    }
    /// Map external unique key into internal one.
    /// Returns corresponded mapped key.
    pub fn map_external_to_internal_mut(&mut self, key: ExternVkey) -> Vkey {
        if self.external_to_internal.contains_key(&key) {
            self.external_to_internal[&key]
        } else {
            let internal_key = self.next_variable_key();
            self.external_to_internal.insert(key, internal_key);
            self.internal_to_external.insert(internal_key, key);
            internal_key
        }
    }
    /// Returns corresponded mapped key.
    pub fn map_external_to_internal(&self, key: ExternVkey) -> Option<Vkey> {
        self.external_to_internal.get(&key).map(|k| k.to_owned())
    }
    /// Returns external to internal map
    pub fn external_to_internal(&self) -> &HashMap<ExternVkey, Vkey> {
        &self.external_to_internal
    }
    /// Returns internal to external map
    pub fn internal_to_extern(&self) -> &HashMap<Vkey, ExternVkey> {
        &self.internal_to_external
    }
    /// Add factor.
    pub fn add_factor<F>(&mut self, factor: F)
    where
        F: Factor<R> + 'static,
    {
        self.factors.add(factor);
    }
    /// Add variable.
    pub fn add_variable<V>(&mut self, variable: V) -> Vkey
    where
        V: Variable<R> + 'static,
    {
        let key = self.next_variable_key();
        self.variables.add(key, variable);
        key
    }
    /// Add variable with unique internal key.
    pub fn add_variable_with_key<V>(&mut self, key: Vkey, variable: V) -> Vkey
    where
        V: Variable<R> + 'static,
    {
        self.variables.add(key, variable);
        key
    }
    /// Add variable with unique external key.
    pub fn add_variable_with_external_key<V>(&mut self, key: ExternVkey, variable: V) -> Vkey
    where
        V: Variable<R> + 'static,
    {
        let external_key = self.map_external_to_internal_mut(key);
        self.variables.add(external_key, variable);
        external_key
    }
    /// Returns factor by index.
    pub fn get_factor<F>(&self, index: usize) -> Option<&F>
    where
        F: Factor<R> + 'static,
    {
        self.factors.get::<F>(index)
    }
    /// Returns mutable factor by index.
    pub fn get_factor_mut<F>(&mut self, index: usize) -> Option<&mut F>
    where
        F: Factor<R> + 'static,
    {
        self.factors.get_mut::<F>(index)
    }
    /// Returns variable by key.
    pub fn get_variable<V>(&self, key: Vkey) -> Option<&V>
    where
        V: Variable<R> + 'static,
    {
        self.variables.get::<V>(key)
    }
    /// Returns mutable variable by key.
    pub fn get_variable_mut<V>(&mut self, key: Vkey) -> Option<&mut V>
    where
        V: Variable<R> + 'static,
    {
        self.variables.get_mut::<V>(key)
    }
    /// Returns vector stored all factors of type `F`.
    pub fn get_factors<F>(&self) -> &Vec<F>
    where
        F: Factor<R> + 'static,
    {
        self.factors.get_vec::<F>()
    }
    /// Returns mutable vector stored all factors of type `F`.
    pub fn get_factors_mut<F>(&mut self) -> &Vec<F>
    where
        F: Factor<R> + 'static,
    {
        self.factors.get_vec_mut::<F>()
    }
    /// Returns map stored all variables of type `V`.
    pub fn get_variables<V>(&self) -> &HashMap<Vkey, V>
    where
        V: Variable<R> + 'static,
    {
        self.variables.get_map::<V>()
    }
    /// Returns mutable map stored all variables of type `V`.
    pub fn get_variables_mut<V>(&mut self) -> &mut HashMap<Vkey, V>
    where
        V: Variable<R> + 'static,
    {
        self.variables.get_map_mut::<V>()
    }
    /// Performs optimization with optimization parameters.
    pub fn optimize<CB>(
        &mut self,
        params: OptParams<FC, VC, R, CB>,
    ) -> Result<(), NonlinearOptimizationError>
    where
        CB: Fn(usize, f64, &Factors<FC, R>, &Variables<VC, R>),
    {
        self.process_pended_tasks();
        self.optimizer
            .optimize_with_callback(&self.factors, &mut self.variables, params.callback)
    }
    /// Returns count of unused variables (without connections).
    /// Must be zero before optimization call.
    pub fn unused_variables_count(&self) -> usize {
        let mut counter = 0_usize;
        for f_idx in 0..self.factors.len() {
            counter += self
                .factors
                .keys_at(f_idx)
                .unwrap()
                .iter()
                .filter(|key| {
                    !self
                        .variables
                        .default_variable_ordering()
                        .keys()
                        .contains(key)
                })
                .count();
        }
        counter
    }
    /// Remove variable, disconnect from related factors.
    fn remove_variable_immediately(
        key: Vkey,
        factors: &mut Factors<FC, R>,
        variables: &mut Variables<VC, R>,
    ) {
        let (ok, _) = variables.remove(key, factors);
        assert!(ok);
    }
    /// Remove variable. If `marginalize` flag is 'true' variable will be removed
    /// before optimization.
    pub fn remove_variable(&mut self, key: Vkey, marginalize: bool) {
        if marginalize {
            self.variables_to_marginalize.push(key);
        } else {
            Self::remove_variable_immediately(key, &mut self.factors, &mut self.variables);
        }
    }
    /// Performs marginalization of variables removed with  `marginalize=true`.
    fn perform_marginalization(&mut self) {
        let marg_prior = marginalize(
            &self.variables_to_marginalize,
            &self.factors,
            &self.variables,
        );
        for key in &self.variables_to_marginalize {
            Self::remove_variable_immediately(*key, &mut self.factors, &mut self.variables);
        }

        if let Some(marg_prior) = marg_prior {
            self.variables_to_marginalize.clear();
            self.factors.add(marg_prior);
        }
    }
    /// Process different pended tasks before optimization.
    fn process_pended_tasks(&mut self) {
        self.perform_marginalization();
    }
    /// Returns next unique variable key.
    /// Current implementation just increment key value. Maybe it should store
    /// keys history and do reindexing when needed.
    fn next_variable_key(&mut self) -> Vkey {
        let key = Vkey(self.variables_keys_counter);
        self.variables_keys_counter += 1;
        key
    }
}

pub struct OptParams<FC, VC, R, BC = fn(usize, f64, &Factors<FC, R>, &Variables<VC, R>) -> ()>
where
    FC: FactorsContainer<R>,
    VC: VariablesContainer<R>,
    R: Real,
    BC: Fn(usize, f64, &Factors<FC, R>, &Variables<VC, R>),
{
    callback: Option<BC>,
    __marker: PhantomData<(FC, VC, R)>,
}

impl<FC, VC, CB, R> OptParams<FC, VC, R, CB>
where
    FC: FactorsContainer<R>,
    VC: VariablesContainer<R>,
    R: Real,
    CB: Fn(usize, f64, &Factors<FC, R>, &Variables<VC, R>),
{
    pub fn builder() -> OptParamsBuilder<FC, VC, R, CB> {
        OptParamsBuilder::default()
        // OptParamsBuilder::new(None::<fn(usize, f64, &Factors<FC, R>, &Variables<VC, R>) -> ()>)
    }
}
impl<FC, VC, BC, R> Default for OptParams<FC, VC, R, BC>
where
    FC: FactorsContainer<R>,
    VC: VariablesContainer<R>,
    R: Real,
    BC: Fn(usize, f64, &Factors<FC, R>, &Variables<VC, R>),
{
    fn default() -> Self {
        Self::builder().build()
    }
}
pub struct OptParamsBuilder<
    FC,
    VC,
    R,
    CB = fn(usize, f64, &Factors<FC, R>, &Variables<VC, R>) -> (),
> where
    FC: FactorsContainer<R>,
    VC: VariablesContainer<R>,
    R: Real,
    CB: Fn(usize, f64, &Factors<FC, R>, &Variables<VC, R>),
{
    callback: Option<CB>,
    __marker: PhantomData<(FC, VC, R)>,
}

impl<FC, VC, CB, R> OptParamsBuilder<FC, VC, R, CB>
where
    FC: FactorsContainer<R>,
    VC: VariablesContainer<R>,
    R: Real,
    CB: Fn(usize, f64, &Factors<FC, R>, &Variables<VC, R>),
{
    pub fn callback(mut self, callback: CB) -> Self {
        self.callback = Some(callback);
        self
    }
    pub fn build(self) -> OptParams<FC, VC, R, CB> {
        OptParams {
            callback: self.callback,
            __marker: PhantomData,
        }
    }
}
impl<FC, VC, CB, R> Default for OptParamsBuilder<FC, VC, R, CB>
where
    FC: FactorsContainer<R>,
    VC: VariablesContainer<R>,
    R: Real,
    CB: Fn(usize, f64, &Factors<FC, R>, &Variables<VC, R>),
{
    fn default() -> Self {
        OptParamsBuilder {
            callback: None,
            __marker: PhantomData,
        }
    }
}
