use criterion::{criterion_group, criterion_main, Criterion};
use nalgebra::DVector;
use optigy::{
    core::{
        factors::Factors, factors_container::FactorsContainer, loss_function::GaussianLoss,
        variables_container::VariablesContainer,
    },
    nonlinear::{
        linearization::linearization_hessian,
        sparsity_pattern::{construct_hessian_sparsity, HessianTriangle},
    },
    prelude::{Variables, Vkey},
    // slam::{between_factor::BetweenFactor, se3::SE2},
};
use slam_common::{between_factor::BetweenFactor, se2::SE2};

fn lower_hessian_sparsity(c: &mut Criterion) {
    let container = ().and_variable::<SE2>();
    let mut variables = Variables::new(container);
    let vcnt = 5000;

    for i in 0..vcnt {
        variables.add(Vkey(i), SE2::new(0.0, 0.0, 0.0));
    }

    let container = ().and_factor::<BetweenFactor>();
    let mut factors = Factors::new(container);

    for i in 0..vcnt {
        for j in 1..6 {
            factors.add(BetweenFactor::new(
                Vkey(i),
                Vkey((i + j) % vcnt),
                0.0,
                0.0,
                0.0,
                Option::<GaussianLoss>::None,
            ));
        }
    }
    let variable_ordering = variables.default_variable_ordering();
    c.bench_function("lower_hessian_sparsity", |b| {
        b.iter(|| {
            construct_hessian_sparsity(
                &factors,
                &variables,
                &variable_ordering,
                HessianTriangle::Upper,
            );
        })
    });
}

fn lower_hessian_linearization(c: &mut Criterion) {
    let container = ().and_variable::<SE2>();
    let mut variables = Variables::new(container);
    let vcnt = 200;

    for i in 0..vcnt {
        variables.add(Vkey(i), SE2::new(0.0, 0.0, 0.0));
    }

    let container = ().and_factor::<BetweenFactor>();
    let mut factors = Factors::new(container);

    for i in 0..vcnt {
        for j in 1..3000 {
            factors.add(BetweenFactor::new(
                Vkey(i),
                Vkey((i + j) % vcnt),
                0.0,
                0.0,
                0.0,
                Option::<GaussianLoss>::None,
            ));
        }
    }
    let variable_ordering = variables.default_variable_ordering();
    let tri = HessianTriangle::Upper;
    let sparsity = construct_hessian_sparsity(&factors, &variables, &variable_ordering, tri);
    let mut a_values = Vec::<f64>::new();
    let a_rows = sparsity.base.A_cols;
    // A_cols = sparsity.base.A_cols;
    a_values.resize(sparsity.total_nnz_AtA_cols, 0.0);
    let mut atb: DVector<f64> = DVector::zeros(a_rows);
    c.bench_function("lower_hessian_linearization", |b| {
        b.iter(|| linearization_hessian(&factors, &variables, &sparsity, &mut a_values, &mut atb))
    });
}
criterion_group!(benches, lower_hessian_sparsity, lower_hessian_linearization);
criterion_main!(benches);
