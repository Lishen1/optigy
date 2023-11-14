use core::cell::Ref;

use crate::core::key::Vkey;
use crate::core::loss_function::LossFunction;
use crate::core::variables::Variables;
use nalgebra::DMatrix;

use nalgebra::DVector;

use super::variables_container::VariablesContainer;
use super::Real;
pub type JacobiansReturn<'a, R> = Ref<'a, DMatrix<R>>;
pub type ErrorReturn<'a, R> = Ref<'a, DVector<R>>;
pub type Jacobians<R> = DMatrix<R>;
pub struct JacobiansErrorReturn<'a, R>
where
    R: Real,
{
    pub jacobians: JacobiansReturn<'a, R>,
    pub error: ErrorReturn<'a, R>,
}
impl<'a, R> JacobiansErrorReturn<'a, R>
where
    R: Real,
{
    fn new(jacobians: JacobiansReturn<'a, R>, error: ErrorReturn<'a, R>) -> Self {
        JacobiansErrorReturn { jacobians, error }
    }
}
/// Represent factor $f_i(\textbf{x})$ of factor graph.
pub trait Factor<R = f64>: Clone
where
    R: Real,
{
    type L: LossFunction<R>;
    /// Returns value of factor function $f_i(\textbf{x})$.
    /// Dimension of $f_i(\textbf{x})$ must be equal to `dim()`.
    fn error<C>(&self, variables: &Variables<C, R>) -> ErrorReturn<R>
    where
        C: VariablesContainer<R>;
    /// Returns jacobians $\frac{\partial f_i(\textbf{x})}{\partial \textbf{x}_{keys}} \big|_x$
    fn jacobians<C>(&self, variables: &Variables<C, R>) -> JacobiansReturn<R>
    where
        C: VariablesContainer<R>;
    /// Returns pair of jacobians with factor function value.
    fn jacobians_error<C>(&self, variables: &Variables<C, R>) -> JacobiansErrorReturn<R>
    where
        C: VariablesContainer<R>,
    {
        JacobiansErrorReturn::new(self.jacobians(variables), self.error(variables))
    }
    /// Returns dimension $D$ of $f_i(\textbf{x}) \in \mathbb{R}^D$
    fn dim(&self) -> usize;
    /// Returns count of variables corresponded with this factor.
    fn len(&self) -> usize {
        self.keys().len()
    }
    /// No corresponded variables.
    fn is_empty(&self) -> bool {
        self.keys().is_empty()
    }
    /// Returns corresponded variables keys.
    fn keys(&self) -> &[Vkey];
    /// Returns noise model which performs whitening transformation of factor an its jacobians.
    fn loss_function(&self) -> Option<&Self::L>;
    /// Returns true if needed to keep this factor on variable remove.
    /// # Arguments
    /// * `key` - A key of variable going to remove.
    fn on_variable_remove(&mut self, _key: Vkey) -> bool {
        false
    }
}

/// Performs numerical differentiation of factor function.
pub fn compute_numerical_jacobians<V, F, R>(
    variables: &Variables<V, R>,
    factor: &F,
    jacobians: &mut DMatrix<R>,
) where
    V: VariablesContainer<R>,
    F: Factor<R>,
    R: Real,
{
    let mut factor_variables = Variables::new(variables.container.empty_clone());

    let mut offsets: Vec<usize> = vec![0; factor.len()];
    let mut offset: usize = 0;
    for (idx, key) in factor.keys().iter().enumerate() {
        variables
            .container
            .add_variable_to(&mut factor_variables, *key);
        offsets[idx] = offset;
        offset += variables.dim_at(*key).unwrap();
    }
    for (idx, key) in factor.keys().iter().enumerate() {
        variables.container.compute_jacobian_for(
            factor,
            &mut factor_variables,
            *key,
            offsets[idx],
            jacobians.as_view_mut(),
        );
    }
}
#[cfg(test)]
pub(crate) mod tests {
    use super::{ErrorReturn, Factor, Jacobians, JacobiansReturn};
    use crate::core::{
        key::Vkey,
        loss_function::GaussianLoss,
        variable::tests::{RandomVariable, VariableA, VariableB},
        variables::Variables,
        variables_container::VariablesContainer,
        Real,
    };
    use core::cell::RefCell;

    use nalgebra::{DMatrix, DVector, Matrix3};

    #[derive(Clone)]
    pub struct FactorA<R>
    where
        R: Real,
    {
        pub orig: DVector<R>,
        pub loss: Option<GaussianLoss<R>>,
        pub error: RefCell<DVector<R>>,
        pub jacobians: RefCell<DMatrix<R>>,
        pub keys: Vec<Vkey>,
    }
    impl<R> FactorA<R>
    where
        R: Real,
    {
        pub fn new(v: R, loss: Option<GaussianLoss<R>>, var0: Vkey, var1: Vkey) -> Self {
            let keys = vec![var0, var1];
            let jacobians = DMatrix::<R>::zeros(3, 3 * keys.len());
            FactorA {
                orig: DVector::from_element(3, v),
                loss,
                error: RefCell::new(DVector::zeros(3)),
                jacobians: RefCell::new(jacobians),
                keys,
            }
        }
    }

    impl<R> Factor<R> for FactorA<R>
    where
        R: Real,
    {
        type L = GaussianLoss<R>;
        fn error<C>(&self, variables: &Variables<C, R>) -> ErrorReturn<R>
        where
            C: VariablesContainer<R>,
        {
            let v0: &VariableA<R> = variables.get(self.keys()[0]).unwrap();
            let v1: &VariableB<R> = variables.get(self.keys()[1]).unwrap();
            {
                *self.error.borrow_mut() = v0.val.clone() - v1.val.clone() + self.orig.clone();
            }
            self.error.borrow()
        }

        fn jacobians<C>(&self, variables: &Variables<C, R>) -> JacobiansReturn<R>
        where
            C: VariablesContainer<R>,
        {
            let v0: &VariableA<R> = variables.get(Vkey(0)).unwrap();
            let v1: &VariableB<R> = variables.get(Vkey(1)).unwrap();
            {
                let mut js = self.jacobians.borrow_mut();
                js.column_mut(0).copy_from(&v0.val);
                js.column_mut(4).copy_from(&v1.val);
            }
            self.jacobians.borrow()
        }

        fn dim(&self) -> usize {
            3
        }

        fn keys(&self) -> &[Vkey] {
            &self.keys
        }

        fn loss_function(&self) -> Option<&Self::L> {
            self.loss.as_ref()
        }
    }
    // pub type FactorB<R> = FactorA<R>;

    #[derive(Clone)]
    pub struct FactorB<R>
    where
        R: Real,
    {
        pub orig: DVector<R>,
        pub loss: Option<GaussianLoss<R>>,
        pub error: RefCell<DVector<R>>,
        pub jacobians: RefCell<DMatrix<R>>,
        pub keys: Vec<Vkey>,
    }
    impl<R> FactorB<R>
    where
        R: Real,
    {
        pub fn new(v: R, loss: Option<GaussianLoss<R>>, var0: Vkey, var1: Vkey) -> Self {
            let _jacobians = Vec::<DMatrix<R>>::with_capacity(2);
            let keys = vec![var0, var1];
            let jacobians = DMatrix::<R>::zeros(3, 3 * keys.len());
            FactorB {
                orig: DVector::from_element(3, v),
                loss,
                error: RefCell::new(DVector::zeros(3)),
                jacobians: RefCell::new(jacobians),
                keys,
            }
        }
    }
    impl<R> Factor<R> for FactorB<R>
    where
        R: Real,
    {
        type L = GaussianLoss<R>;
        fn error<C>(&self, variables: &Variables<C, R>) -> ErrorReturn<R>
        where
            C: VariablesContainer<R>,
        {
            let v0: &VariableA<R> = variables.get(Vkey(0)).unwrap();
            let v1: &VariableB<R> = variables.get(Vkey(1)).unwrap();
            {
                *self.error.borrow_mut() = v0.val.clone() - v1.val.clone() + self.orig.clone();
            }
            self.error.borrow()
        }
        fn jacobians<C>(&self, variables: &Variables<C, R>) -> JacobiansReturn<R>
        where
            C: VariablesContainer<R>,
        {
            let v0: &VariableA<R> = variables.get(Vkey(0)).unwrap();
            let v1: &VariableB<R> = variables.get(Vkey(1)).unwrap();
            {
                let mut js = self.jacobians.borrow_mut();
                js.column_mut(0).copy_from(&v0.val);
                js.column_mut(4).copy_from(&v1.val);
            }
            self.jacobians.borrow()
        }

        fn dim(&self) -> usize {
            3
        }

        fn keys(&self) -> &[Vkey] {
            &self.keys
        }

        fn loss_function(&self) -> Option<&Self::L> {
            self.loss.as_ref()
        }
    }
    #[derive(Clone)]
    pub struct RandomBlockFactor<R>
    where
        R: Real,
    {
        pub loss: Option<GaussianLoss<R>>,
        pub error: RefCell<DVector<R>>,
        pub jacobians: RefCell<Jacobians<R>>,
        pub keys: Vec<Vkey>,
    }
    impl<R> RandomBlockFactor<R>
    where
        R: Real,
    {
        pub fn new(var0: Vkey, var1: Vkey) -> Self {
            let _rng = rand::thread_rng();
            let _jacobians = Vec::<DMatrix<R>>::with_capacity(2);
            let keys = vec![var0, var1];
            let jacobians = DMatrix::<R>::from_fn(3, 3 * keys.len(), |_i, _j| R::one());
            RandomBlockFactor {
                loss: None,
                error: RefCell::new(DVector::zeros(3)),
                jacobians: RefCell::new(jacobians),
                keys,
            }
        }
    }
    impl<R> Factor<R> for RandomBlockFactor<R>
    where
        R: Real,
    {
        type L = GaussianLoss<R>;
        fn error<C>(&self, variables: &Variables<C, R>) -> ErrorReturn<R>
        where
            C: VariablesContainer<R>,
        {
            let v0: &RandomVariable<R> = variables.get(self.keys()[0]).unwrap();
            let v1: &RandomVariable<R> = variables.get(self.keys()[1]).unwrap();
            {
                *self.error.borrow_mut() = v0.val.clone() - v1.val.clone();
            }
            self.error.borrow()
        }

        fn jacobians<C>(&self, _variables: &Variables<C, R>) -> JacobiansReturn<R>
        where
            C: VariablesContainer<R>,
        {
            self.jacobians.borrow()
        }

        fn dim(&self) -> usize {
            3
        }

        fn keys(&self) -> &[Vkey] {
            &self.keys
        }

        fn loss_function(&self) -> Option<&Self::L> {
            self.loss.as_ref()
        }
    }
    #[test]
    fn error() {
        type Real = f64;
        let container = ().and_variable::<VariableA<Real>>().and_variable::<VariableB<Real>>();
        let mut variables = Variables::new(container);
        variables.add(Vkey(0), VariableA::<Real>::new(4.0));
        variables.add(Vkey(1), VariableB::<Real>::new(2.0));
        let loss = GaussianLoss::information(Matrix3::identity().as_view());
        let f0 = FactorA::new(1.0, Some(loss), Vkey(0), Vkey(1));
        {
            let e0 = f0.error(&variables).to_owned();
            assert_eq!(e0, DVector::<Real>::from_element(3, 3.0));
        }
        let v0: &mut VariableA<Real> = variables.get_mut(Vkey(0)).unwrap();
        v0.val.fill(3.0);
        {
            let e0 = f0.error(&variables).to_owned();
            assert_eq!(e0, DVector::<Real>::from_element(3, 2.0));
        }
    }
    #[test]
    fn factor_impl() {}
}
