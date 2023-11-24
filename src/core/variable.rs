use nalgebra::{DMatrixViewMut, DVector, DVectorView, DVectorViewMut};

use crate::prelude::{Variables, VariablesContainer};

use super::Real;

/// Represent variable $\textbf{x}_i$ of factor graph.
pub trait Variable<R>: Clone
where
    R: Real,
{
    /// Computes local tangent such: $\textbf{x}_i \boxminus \breve{\textbf{x}}_i$
    /// where $\breve{\textbf{x}}_i$ is linearization point in case of marginalization.
    fn local(&self, linearization_point: &Self, tangent: DVectorViewMut<R>);

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
    /// Computes jacobian of local(tangent) of variable perturbation
    /// with respect of perturbation delta:
    /// $$\frac{\partial (\textbf{x}_i \boxplus \delta) \boxminus \breve{\textbf{x}}_i}{\partial \delta} \Big|_x$$
    fn retract_local_jacobian(&self, linearization_point: &Self, jacobian: DMatrixViewMut<R>) {
        compute_variable_numerical_jacobian(linearization_point, self, jacobian);
    }
    /// Updates some variable data vased on other variables.
    /// Needed for coordinate frame transformation for example.
    fn update<VC>(&mut self, _variables: &Variables<VC, R>)
    where
        VC: VariablesContainer<R>,
    {
    }
}

/// Performs numerical differentiation of variable local(retact(dx)).
pub fn compute_variable_numerical_jacobian<V, R>(
    linearization_point: &V,
    variable: &V,
    mut jacobian: DMatrixViewMut<R>,
) where
    V: Variable<R>,
    R: Real,
{
    //central difference
    let delta = R::from_f64(1e-9).unwrap();
    let mut dx = DVector::<R>::zeros(variable.dim());
    let mut dy0 = DVector::<R>::zeros(variable.dim());
    let mut dy1 = DVector::<R>::zeros(variable.dim());
    for i in 0..variable.dim() {
        dx[i] = delta.clone();
        variable
            .retracted(dx.as_view())
            .local(linearization_point, dy0.as_view_mut());
        dx[i] = -delta.clone();
        variable
            .retracted(dx.as_view())
            .local(linearization_point, dy1.as_view_mut());
        jacobian.column_mut(i).copy_from(
            &((dy0.clone() - dy1.clone()) / (R::from_f64(2.0).unwrap() * delta.clone())),
        );
        dx[i] = R::zero();
    }
}

#[cfg(test)]
pub(crate) mod tests {

    use nalgebra::DVector;
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
        R: Real,
    {
        fn local(&self, value: &Self, mut tangent: DVectorViewMut<R>)
        where
            R: RealField,
        {
            tangent.copy_from(&(self.val.clone() - value.val.clone()));
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

        // fn retract_local_jacobian(&self, linearization_point: &Self, jacobian: DMatrixViewMut<R>) {
        //     compute_variable_numerical_jacobian(linearization_point, self, jacobian);
        // }
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
        R: Real,
    {
        fn local(&self, value: &Self, mut tangent: DVectorViewMut<R>)
        where
            R: RealField,
        {
            tangent.copy_from(&(self.val.clone() - value.val.clone()));
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

        fn retract_local_jacobian(
            &self,
            _linearization_point: &Self,
            mut jacobian: DMatrixViewMut<R>,
        ) {
            jacobian.fill_with_identity();
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
        R: Real,
    {
        fn local(&self, value: &Self, mut tangent: DVectorViewMut<R>)
        where
            R: RealField,
        {
            tangent.copy_from(&(self.val.clone() - value.val.clone()));
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

        fn retract_local_jacobian(
            &self,
            _linearization_point: &Self,
            mut jacobian: DMatrixViewMut<R>,
        ) {
            jacobian.fill_with_identity();
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
