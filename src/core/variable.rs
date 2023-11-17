use std::cell::Ref;

use nalgebra::{DVector, DVectorView, RealField};

use crate::prelude::JacobianReturn;
/// Represent variable $\textbf{x}_i$ of factor graph.
pub type TangentReturn<'a, R> = Ref<'a, DVector<R>>;
pub trait Variable<R>: Clone
where
    R: RealField,
{
    /// Returns local tangent such: $\textbf{x}_i \boxminus \breve{\textbf{x}}_i$
    /// where $\breve{\textbf{x}}_i$ is linearization point in case of marginalization.
    fn local(&self, linearization_point: &Self) -> TangentReturn<R>;

    /// Retract (perturbate) $\textbf{x}_i$ by `delta` such:
    /// $\textbf{x}_i=\textbf{x}_i \boxplus \delta \textbf{x}_i$
    fn retract(&mut self, delta: DVectorView<R>);

    /// Returns retracted copy of `self`.
    fn retracted(&self, delta: DVectorView<R>) -> Self {
        let mut var = self.clone();
        var.retract(delta);
        var
    }
    /// Returns dimension $D$ of $\delta{\textbf{x}_i} \in \mathbb{R}^D$
    fn dim(&self) -> usize;
    /// Returns jacobian of local(tangent) of variable perturbation
    /// with respect of perturbation delta:
    /// $$\frac{\partial (\textbf{x}_i \boxplus \delta) \boxminus \breve{\textbf{x}}_i}{\partial \delta} \Big|_x$$
    fn retract_local_jacobian(&self, linearization_point: &Self) -> JacobianReturn<R>;
}
#[cfg(test)]
pub(crate) mod tests {
    use std::cell::RefCell;

    use rand::Rng;

    use super::*;

    #[derive(Debug, Clone)]
    pub struct VariableA<R>
    where
        R: RealField,
    {
        pub val: DVector<R>,
        local: RefCell<DVector<R>>,
    }

    impl<R> Variable<R> for VariableA<R>
    where
        R: RealField,
    {
        fn local(&self, value: &Self) -> TangentReturn<R>
        where
            R: RealField,
        {
            *self.local.borrow_mut() = self.val.clone() - value.val.clone();
            self.local.borrow()
        }

        fn retract(&mut self, delta: DVectorView<R>)
        where
            R: RealField,
        {
            self.val = self.val.clone() + delta;
        }

        fn dim(&self) -> usize {
            3
        }
    }
    #[derive(Debug, Clone)]
    pub struct VariableB<R>
    where
        R: RealField,
    {
        pub val: DVector<R>,
        local: RefCell<DVector<R>>,
    }

    impl<R> Variable<R> for VariableB<R>
    where
        R: RealField,
    {
        fn local(&self, value: &Self) -> TangentReturn<R>
        where
            R: RealField,
        {
            *self.local.borrow_mut() = self.val.clone() - value.val.clone();
            self.local.borrow()
        }

        fn retract(&mut self, delta: DVectorView<R>)
        where
            R: RealField,
        {
            self.val = self.val.clone() + delta.clone();
        }

        fn dim(&self) -> usize {
            3
        }
    }

    impl<R> VariableA<R>
    where
        R: RealField,
    {
        pub fn new(v: R) -> Self {
            VariableA {
                val: DVector::<R>::from_element(3, v),
                local: RefCell::new(DVector::<R>::zeros(3)),
            }
        }
    }
    impl<R> VariableB<R>
    where
        R: RealField,
    {
        pub fn new(v: R) -> Self {
            VariableB {
                val: DVector::<R>::from_element(3, v),
                local: RefCell::new(DVector::<R>::zeros(3)),
            }
        }
    }

    #[derive(Debug, Clone)]
    pub struct RandomVariable<R>
    where
        R: RealField,
    {
        pub val: DVector<R>,
        local: RefCell<DVector<R>>,
    }

    impl<R> Variable<R> for RandomVariable<R>
    where
        R: RealField,
    {
        fn local(&self, value: &Self) -> TangentReturn<R>
        where
            R: RealField,
        {
            *self.local.borrow_mut() = self.val.clone() - value.val.clone();
            self.local.borrow()
        }

        fn retract(&mut self, delta: DVectorView<R>)
        where
            R: RealField,
        {
            self.val = self.val.clone() + delta.clone();
        }

        fn dim(&self) -> usize {
            3
        }
    }
    impl<R> Default for RandomVariable<R>
    where
        R: RealField,
    {
        fn default() -> Self {
            let mut rng = rand::thread_rng();
            RandomVariable {
                val: DVector::from_fn(3, |_, _| R::from_f64(rng.gen::<f64>()).unwrap()),
                local: RefCell::new(DVector::<R>::zeros(3)),
            }
        }
    }
}
