use nalgebra::{DVector, DVectorView, RealField};
/// Represent variable $\textbf{x}_i$ of factor graph.
pub trait Variable<R>: Clone
where
    R: RealField,
{
    /// Returns local tangent such: $\textbf{x}_i \boxminus \breve{\textbf{x}}_i$
    /// where $\breve{\textbf{x}}_i$ is linearization point in case of marginalization.
    fn local(&self, linearization_point: &Self) -> DVector<R>
    where
        R: RealField;
    /// Retract (perturbate) $\textbf{x}_i$ by `delta` such:
    /// $\textbf{x}_i=\textbf{x}_i \boxplus \delta \textbf{x}_i$
    fn retract(&mut self, delta: DVectorView<R>)
    where
        R: RealField;
    /// Returns retracted copy of `self`.
    fn retracted(&self, delta: DVectorView<R>) -> Self
    where
        R: RealField,
    {
        let mut var = self.clone();
        var.retract(delta);
        var
    }
    /// Returns dimension $D$ of $\delta{\textbf{x}_i} \in \mathbb{R}^D$
    fn dim(&self) -> usize;
}
#[cfg(test)]
pub(crate) mod tests {
    use rand::Rng;

    use super::*;

    #[derive(Debug, Clone)]
    pub struct VariableA<R>
    where
        R: RealField,
    {
        pub val: DVector<R>,
    }

    impl<R> Variable<R> for VariableA<R>
    where
        R: RealField,
    {
        fn local(&self, value: &Self) -> DVector<R>
        where
            R: RealField,
        {
            self.val.clone() - value.val.clone()
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
    }

    impl<R> Variable<R> for VariableB<R>
    where
        R: RealField,
    {
        fn local(&self, value: &Self) -> DVector<R>
        where
            R: RealField,
        {
            self.val.clone() - value.val.clone()
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
            }
        }
    }

    #[derive(Debug, Clone)]
    pub struct RandomVariable<R>
    where
        R: RealField,
    {
        pub val: DVector<R>,
    }

    impl<R> Variable<R> for RandomVariable<R>
    where
        R: RealField,
    {
        fn local(&self, value: &Self) -> DVector<R>
        where
            R: RealField,
        {
            self.val.clone() - value.val.clone()
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
            }
        }
    }
}
