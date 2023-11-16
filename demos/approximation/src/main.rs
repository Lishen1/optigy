use core::cell::RefCell;
use std::fs::File;
use std::io::Write;

use color_print::cprintln;
use nalgebra::vector;
use nalgebra::DMatrixView;
use nalgebra::DVectorView;

use nalgebra::Vector2;
use nalgebra::{DMatrix, DVector};
use num::Float;

use optigy::fixedlag::marginalization::marginalize;
use optigy::fixedlag::marginalization::symmetrize_with_mean;
use optigy::fixedlag::marginalization::try_invert_symmetric_positive_semidefinite_matrix;
use optigy::nonlinear::linearization::linearzation_jacobian;
use optigy::nonlinear::sparsity_pattern::construct_jacobian_sparsity;
use optigy::prelude::DiagonalLoss;
use optigy::prelude::ErrorReturn;
use optigy::prelude::Factors;
use optigy::prelude::FactorsContainer;

use optigy::prelude::JacobiansReturn;
use optigy::prelude::LevenbergMarquardtOptimizer;
use optigy::prelude::LevenbergMarquardtOptimizerParams;
use optigy::prelude::NonlinearOptimizer;

use optigy::prelude::NonlinearOptimizerVerbosityLevel;
use optigy::prelude::Real;
use optigy::prelude::VariableOrdering;
use optigy::prelude::VariablesContainer;
use optigy::prelude::{Factor, Jacobians, ScaleLoss, Variables, Vkey};
use rgb::RGB8;
use slam_common::between_factor::BetweenFactor;
use slam_common::se2::SE2;
use textplots::Chart;
use textplots::ColorPlot;
use textplots::Shape;
#[derive(Clone)]
struct GPSPositionFactor<R = f64>
where
    R: Real,
{
    pub error: RefCell<DVector<R>>,
    pub jacobians: RefCell<Jacobians<R>>,
    pub keys: Vec<Vkey>,
    pub pose: Vector2<R>,
    pub loss: DiagonalLoss<R>,
}
impl<R> GPSPositionFactor<R>
where
    R: Real,
{
    pub fn new(key: Vkey, pose: Vector2<R>, sigmas: Vector2<R>) -> Self {
        let keys = vec![key];
        let jacobians = DMatrix::identity(2, 3 * keys.len());
        GPSPositionFactor {
            error: RefCell::new(DVector::zeros(2)),
            jacobians: RefCell::new(jacobians),
            keys,
            pose,
            loss: DiagonalLoss::sigmas(&sigmas.as_view()),
        }
    }
}
impl<R> Factor<R> for GPSPositionFactor<R>
where
    R: Real,
{
    type L = DiagonalLoss<R>;
    fn error<C>(&self, variables: &Variables<C, R>) -> ErrorReturn<R>
    where
        C: VariablesContainer<R>,
    {
        let v0: &SE2<R> = variables.get(self.keys()[0]).unwrap();
        {
            let pose = v0.origin.params();
            let pose = vector![pose[0], pose[1]];
            let d = pose.cast::<R>() - self.pose;
            self.error.borrow_mut().copy_from(&d);
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
        2
    }

    fn keys(&self) -> &[Vkey] {
        &self.keys
    }

    fn loss_function(&self) -> Option<&Self::L> {
        Some(&self.loss)
    }
}

/**
 * A simple 2D pose-graph SLAM with 'GPS' measurement
 * The robot moves from x1 to x3, with odometry information between each pair.
 * each step has an associated 'GPS' measurement by GPSPose2Factor
 * The graph strcuture is shown:
 *
 *  g1   g2   g3
 *  |    |    |
 *  x1 - x2 - x3
 *
 * The GPS factor has error function
 *     e = pose.translation() - measurement
 */

#[allow(non_snake_case)]
fn quadratic<R>(H: DMatrixView<R>, b: DVectorView<R>, dx: DVectorView<R>, f0: R) -> R
where
    R: Real,
{
    let f = dx.transpose() * H * dx * R::from_f64(0.5).unwrap() + b.transpose() * dx;
    f[(0, 0)] + f0
}
fn linspace<T: Float + std::convert::From<u16>>(l: T, h: T, n: usize) -> Vec<T> {
    let size: T = (n as u16 - 1)
        .try_into()
        .expect("too many elements: max is 2^16");
    let dx = (h - l) / size;

    (1..=n)
        .scan(l, |a, _| {
            *a = *a + dx;
            Some(*a)
        })
        .collect()
}
#[allow(non_snake_case)]
fn main() {
    let container = ().and_variable::<SE2>();
    let mut variables = Variables::new(container);

    let container = ()
        .and_factor::<GPSPositionFactor>()
        // .and_factor::<BetweenFactor<GaussianLoss>>()
        .and_factor::<BetweenFactor<ScaleLoss>>();
    let mut factors = Factors::new(container);

    factors.add(GPSPositionFactor::new(
        Vkey(1),
        Vector2::new(0.0, 0.0),
        Vector2::new(2.0, 2.0),
    ));
    factors.add(GPSPositionFactor::new(
        Vkey(2),
        Vector2::new(5.0, 0.0),
        Vector2::new(2.0, 2.0),
    ));
    factors.add(GPSPositionFactor::new(
        Vkey(3),
        Vector2::new(10.0, 0.0),
        Vector2::new(2.0, 2.0),
    ));
    factors.add(GPSPositionFactor::new(
        Vkey(4),
        Vector2::new(12.0, 0.0),
        Vector2::new(2.0, 2.0),
    ));
    factors.add(GPSPositionFactor::new(
        Vkey(5),
        Vector2::new(15.0, 0.0),
        Vector2::new(2.0, 2.0),
    ));
    factors.add(BetweenFactor::new(
        Vkey(1),
        Vkey(2),
        5.0,
        0.0,
        0.0,
        // Some(GaussianLoss {}),
        Some(ScaleLoss::scale(1e1)),
    ));
    factors.add(BetweenFactor::new(
        Vkey(2),
        Vkey(3),
        5.0,
        0.0,
        0.0,
        Some(ScaleLoss::scale(1e1)),
    ));
    factors.add(BetweenFactor::new(
        Vkey(3),
        Vkey(4),
        5.0,
        0.0,
        0.0,
        Some(ScaleLoss::scale(1e1)),
    ));
    factors.add(BetweenFactor::new(
        Vkey(4),
        Vkey(5),
        5.0,
        0.0,
        0.0,
        Some(ScaleLoss::scale(1e1)),
    ));

    variables.add(Vkey(1), SE2::new(0.2, -0.3, 0.2));
    variables.add(Vkey(2), SE2::new(5.1, 0.3, -0.1));
    variables.add(Vkey(3), SE2::new(9.9, -0.1, -0.2));
    variables.add(Vkey(4), SE2::new(12.2, -0.1, -0.2));
    variables.add(Vkey(5), SE2::new(15.3, -0.1, -0.2));

    // let mut optimizer = NonlinearOptimizer::default();
    let mut params = LevenbergMarquardtOptimizerParams::default();
    params.base.verbosity_level = NonlinearOptimizerVerbosityLevel::Subiteration;
    // params.base.max_iterations = 2;
    let mut optimizer =
        NonlinearOptimizer::<_, f64>::new(LevenbergMarquardtOptimizer::<f64>::with_params(params));

    let opt_res = optimizer.optimize(&factors, &mut variables);
    println!("opt_res {:?}", opt_res);
    println!("final error {}", factors.error_squared_norm(&variables));

    //marginalization test
    let binding = vec![Vkey(2), Vkey(3)];
    let m_keys = binding.as_slice();
    let n_keys = factors.neighborhood_variables(m_keys);
    let mn_keys = [m_keys, n_keys.as_slice()].concat();

    let mn_factors = Factors::from_connected_factors(&factors, m_keys);
    let mn_variables = Variables::from_variables(&variables, &mn_keys);
    let mut r_factors = factors.clone();
    let mut nr_keys: Vec<Vkey> = variables.default_variable_ordering().keys().into();
    for key in m_keys {
        nr_keys.retain(|k| *k != *key);
    }
    let nr_variables = Variables::from_variables(&variables, &nr_keys);
    for key in m_keys {
        r_factors.remove_conneted_factors(*key);
    }
    let m_dim: usize = m_keys.iter().map(|k| variables.dim_at(*k).unwrap()).sum();
    let n_dim: usize = n_keys.iter().map(|k| variables.dim_at(*k).unwrap()).sum();
    println!("n_keys {:?}", n_keys);
    println!("m_keys {:?}", m_keys);
    println!("mn_keys {:?}", mn_keys);
    println!("factors len {}", factors.len());
    println!("mn_factors len {}", mn_factors.len());
    println!("r_factors len {}", r_factors.len());

    let jsparsity =
        construct_jacobian_sparsity(&mn_factors, &mn_variables, &VariableOrdering::new(&mn_keys));
    let mut jA = DMatrix::<f64>::zeros(jsparsity.base.A_rows, jsparsity.base.A_cols);
    let mut jb = DVector::<f64>::zeros(jsparsity.base.A_rows);
    linearzation_jacobian(&mn_factors, &mn_variables, &jsparsity, &mut jA, &mut jb);

    let H = jA.transpose() * jA.clone();
    let b = -jA.transpose() * jb.clone();
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
    // let Hm_pinv = Hm.try_inverse().unwrap();

    let mut Ht = Hn - Hmn * Hm_pinv.clone() * Hmn.transpose();
    let bt = bn - Hmn * Hm_pinv * bm;
    symmetrize_with_mean(Ht.as_view_mut());

    // println!("Ht {}", Ht);
    // println!("U {}", U);
    // println!("v {}", v);
    // println!("ref Ht {}", Ht);
    // println!(
    //     "recomp Ht {}",
    //     U.clone() * DMatrix::<f64>::from_diagonal(&v) * U.clone().transpose()
    // );

    let marg_prior = marginalize(m_keys, &factors, &variables).unwrap();
    let A_prior = marg_prior.A_prior;
    let b_prior = marg_prior.b_prior;

    cprintln!("<rgb(0,255,0)>===============================</>");

    let F_x0 = factors.error_squared_norm(&variables);
    println!("F(x) at x0 {}", F_x0);
    let F_mn_x0 = mn_factors.error_squared_norm(&mn_variables);
    let F_r_x0 = r_factors.error_squared_norm(&nr_variables);
    let F_x0mnr = F_mn_x0 + F_r_x0;
    println!("F(xm, xn)+F(xn, xr) at x0 {}", F_x0mnr);
    println!("F(xm, xn) at x0 {}", F_mn_x0);

    fn ref_F_xmxn<FC, VC, R>(
        x: R,
        dx0: DVectorView<R>,
        mn_factors: &Factors<FC, R>,
        mn_variables: &Variables<VC, R>,
    ) -> R
    where
        FC: FactorsContainer<R>,
        VC: VariablesContainer<R>,
        R: Real,
    {
        let d = DVector::<R>::from_element(dx0.len(), x);
        let dx = dx0.to_owned() + d;
        let F_mn_dx = mn_factors.error_squared_norm(
            &mn_variables.retracted(dx.as_view(), &mn_variables.default_variable_ordering()),
        );
        R::from_f64(F_mn_dx).unwrap()
        // R::from_f64(F_mn_dx).unwrap()
    }
    fn m_est_err<R>(
        x: R,
        dx0: DVectorView<R>,
        H: DMatrixView<R>,
        b: DVectorView<R>,
        m_dim: usize,
    ) -> R
    where
        R: Real,
    {
        let d = DVector::<R>::from_element(dx0.len(), x);
        let dx = dx0.to_owned() + d;

        let n_dim = H.ncols() - m_dim;
        let m_cnt = m_dim;
        let n_cnt = n_dim;
        let m_f = 0_usize;
        let n_f = m_cnt;

        let Hm = H.view((m_f, m_f), (m_cnt, m_cnt)).to_owned();
        let Hn = H.view((n_f, n_f), (n_cnt, n_cnt)).to_owned();
        let Hmn = H.view((n_f, m_f), (n_cnt, m_cnt)).to_owned();
        let bm = b.rows(m_f, m_cnt);
        let bn = b.rows(n_f, n_cnt);

        let Hm_pinv =
            try_invert_symmetric_positive_semidefinite_matrix(false, Hm.as_view()).unwrap();

        let mut Ht = Hn - Hmn * Hm_pinv.clone() * Hmn.transpose();
        let _bt = bn - Hmn * Hm_pinv.clone() * bm;
        symmetrize_with_mean(Ht.as_view_mut());

        let dn = dx.rows(n_f, n_cnt).to_owned();
        let ref_dm = dx.rows(0, m_cnt).to_owned();
        let dm = -Hm_pinv * (bm + Hmn.transpose() * dn);

        // println!("est dm {} ref dm {}", dm, ref_dm);

        (dm - ref_dm).norm()
    }
    fn approx_F_xmxn<R>(x: R, dx0: DVectorView<R>, H: DMatrixView<R>, b: DVectorView<R>, f0: R) -> R
    where
        R: Real,
    {
        let d = DVector::<R>::from_element(dx0.len(), x);
        let dx = dx0.to_owned() + d;
        let e = quadratic(H.as_view(), b.as_view(), dx.as_view(), f0);
        e
    }
    fn approx_F_xn<R>(x: R, dx0: DVectorView<R>, H: DMatrixView<R>, b: DVectorView<R>, f0: R) -> R
    where
        R: Real,
    {
        let dx = dx0.to_owned() + DVector::<R>::from_element(dx0.len(), x);
        let e = quadratic(H.as_view(), b.as_view(), dx.as_view(), f0);
        e
    }
    fn global_approx_F_xn<R>(
        x: R,
        dx0: DVectorView<R>,
        Omega: DMatrixView<R>,
        bt: DVectorView<R>,
        f0: R,
    ) -> R
    where
        R: Real,
    {
        let dx = dx0.to_owned() + DVector::<R>::from_element(dx0.len(), x);
        let r = dx + Omega.try_inverse().unwrap() * bt;
        let f = r.transpose() * Omega * r * R::from_f64(0.5).unwrap();
        f[(0, 0)] + f0
    }
    fn approx_prior_F_xn<R>(
        x: R,
        dx0: DVectorView<R>,
        A: DMatrixView<R>,
        b: DVectorView<R>,
        f0: R,
    ) -> R
    where
        R: Real,
    {
        let dx = dx0.to_owned() + DVector::<R>::from_element(dx0.len(), x);
        let r = A * dx + b;
        R::from_f64(0.5).unwrap() * r.norm_squared() + f0
    }

    let dx_mn = -H.clone().try_inverse().unwrap() * b.clone();

    let dx_Fmn = dx_mn.clone();
    let dx_Fn = DVector::<f64>::from(dx_mn.rows(m_dim, n_dim));
    let dx_Fn_prior = DVector::<f64>::from(dx_mn.rows(m_dim, n_dim));
    let dx_m_est_err = dx_mn.clone();
    let vars_Fmn = mn_variables.clone();
    let facts_Fmn = mn_factors.clone();
    let eH = H.clone();
    let eb = b.clone();

    let mut file = File::create("approx.txt").unwrap();
    let mut min_y = f64::MAX;
    let mut max_y = 0.0f64;
    let min_x = -0.1;
    let max_x = 0.1;

    for x in linspace(min_x, max_x, 1000) {
        let y0 = ref_F_xmxn(x as f64, dx_Fmn.as_view(), &facts_Fmn, &vars_Fmn);
        let y1 = approx_F_xmxn(x as f64, dx_mn.as_view(), H.as_view(), b.as_view(), F_mn_x0);
        let y2 = approx_F_xn(
            x as f64,
            dx_Fn.as_view(),
            Ht.as_view(),
            bt.as_view(),
            F_mn_x0 * 0.0,
        );
        let y3 = approx_prior_F_xn(
            x as f64,
            dx_Fn_prior.as_view(),
            A_prior.as_view(),
            b_prior.as_view(),
            F_mn_x0,
        );
        let omega = Ht.clone();
        let y4 = global_approx_F_xn(
            x as f64,
            dx_Fn.as_view(),
            omega.as_view(),
            bt.as_view(),
            F_mn_x0 * 0.0,
        );
        min_y = min_y.min(y0);
        max_y = max_y.max(y0);
        min_y = min_y.min(y1);
        max_y = max_y.max(y1);
        min_y = min_y.min(y2);
        max_y = max_y.max(y2);
        min_y = min_y.min(y3);
        max_y = max_y.max(y3);
        min_y = min_y.min(y4);
        max_y = max_y.max(y4);

        file.write_fmt(format_args!("{} {} {} {} {} {}\n", x, y0, y1, y2, y3, y4))
            .unwrap();
    }
    file.flush().unwrap();
    cprintln!("<rgb(0,0,255)><bg:white>F(xm, xn) = 0.5*sum(||f(xm, xn)||^2)</>");
    cprintln!("<rgb(255,0,0)><bg:white>F(xm0+dm, xn0+dn) = 0.5*dx.T*H*dx+b.T*dx+f0 where dx = [dm, dn]</>");
    cprintln!("<rgb(0,255,0)>F(xn0+dn) = 0.5*dn.T*Ht*dn+bt.T*dn+f0</>");
    cprintln!("<rgb(255,255,0)>F(xn0+dn) = 0.5*||A_prior*dn+b_prior||^2</>");
    Chart::new_with_y_range(100, 80, min_x, max_x, min_y as f32, max_y as f32)
        .linecolorplot(
            &Shape::Continuous(Box::new(move |x| {
                ref_F_xmxn(x as f64, dx_Fmn.as_view(), &facts_Fmn, &vars_Fmn) as f32
            })),
            RGB8 { r: 0, g: 0, b: 255 },
        )
        .linecolorplot(
            &Shape::Continuous(Box::new(move |x| {
                approx_F_xmxn(
                    x as f64 + 0.0002 * 0.0,
                    dx_mn.as_view(),
                    H.as_view(),
                    b.as_view(),
                    F_mn_x0,
                ) as f32
            })),
            RGB8 { r: 255, g: 0, b: 0 },
        )
        .linecolorplot(
            &Shape::Continuous(Box::new(move |x| {
                approx_F_xn(
                    x as f64 + 0.0004 * 0.0,
                    dx_Fn.as_view(),
                    Ht.as_view(),
                    bt.as_view(),
                    F_mn_x0,
                ) as f32
            })),
            RGB8 { r: 0, g: 255, b: 0 },
        )
        .linecolorplot(
            &Shape::Continuous(Box::new(move |x| {
                approx_prior_F_xn(
                    x as f64 + 0.0006 * 0.0,
                    dx_Fn_prior.as_view(),
                    A_prior.as_view(),
                    b_prior.as_view(),
                    F_mn_x0,
                ) as f32
            })),
            RGB8 {
                r: 255,
                g: 255,
                b: 0,
            },
        )
        .nice();

    let mut min_y = f64::MAX;
    let mut max_y = 0.0f64;

    for x in linspace(min_x, max_x, 1000) {
        let y = m_est_err(
            x as f64,
            dx_m_est_err.as_view(),
            eH.as_view(),
            eb.as_view(),
            m_dim,
        );
        min_y = min_y.min(y);
        max_y = max_y.max(y);
    }
    println!("min max marg error: |dm - dm*| {} {}", min_y, max_y);
    Chart::new_with_y_range(100, 50, min_x, max_x, min_y as f32, max_y as f32)
        .linecolorplot(
            &Shape::Continuous(Box::new(move |x| {
                m_est_err(
                    x as f64 + 0.0002 * 0.0,
                    dx_m_est_err.as_view(),
                    eH.as_view(),
                    eb.as_view(),
                    m_dim,
                ) as f32
            })),
            RGB8 { r: 255, g: 0, b: 0 },
        )
        .nice();
}
