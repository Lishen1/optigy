use lazy_static::lazy_static;
use std::cell::RefCell;

use std::collections::{HashMap, VecDeque};

use std::error::Error;
use std::fs::File;
use std::io::Write;

use std::time::Instant;
use std::{env::current_dir, fs::read_to_string};

use clap::Parser;
use nalgebra::{
    dvector, matrix, vector, DMatrix, DMatrixView, DVector, DVectorView, Matrix2, RealField,
    Vector2, Vector3,
};
use num::Float;
use optigy::core::factor::ErrorReturn;
use optigy::core::loss_function::{DiagonalLoss, ScaleLoss};

use optigy::fixedlag::marginalization::{add_dense_marginalize_prior_factor, marginalize};

use optigy::prelude::{
    Factor, FactorGraph, Factors, FactorsContainer, GaussianLoss, JacobiansReturn,
    LevenbergMarquardtOptimizer, LevenbergMarquardtOptimizerParams,
    NonlinearOptimizerVerbosityLevel, OptParams, Variable, Variables, VariablesContainer, Vkey,
};
use optigy::slam::between_factor::BetweenFactor;
use optigy::slam::prior_factor::PriorFactor;
use optigy::slam::se3::SE2;
use optigy::viz::graph_viz::{FactorGraphViz, HighlightFactorsGroup, HighlightVariablesGroup};
use plotters::coord::types::RangedCoordf64;
use plotters::coord::Shift;
use plotters::prelude::*;
use plotters::style::full_palette::{BLACK, GREEN};
use random_color::RandomColor;
#[derive(Clone)]
struct GPSPositionFactor<R = f64>
where
    R: RealField + Float,
{
    pub error: RefCell<DVector<R>>,
    jacobians: RefCell<DMatrix<R>>,
    pub keys: Vec<Vkey>,
    pub pose: Vector2<R>,
    pub loss: DiagonalLoss<R>,
}
impl<R> GPSPositionFactor<R>
where
    R: RealField + Float,
{
    pub fn new(key: Vkey, pose: Vector2<R>, sigmas: Vector2<R>) -> Self {
        let keys = vec![key];
        let jacobians = DMatrix::identity(2, 3);
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
    R: RealField + Float,
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
            let d = self.pose - pose.cast::<R>();
            self.error.borrow_mut().copy_from(&d);
        }
        self.error.borrow()
    }

    fn jacobians<C>(&self, variables: &Variables<C, R>) -> JacobiansReturn<R>
    where
        C: VariablesContainer<R>,
    {
        // compute_numerical_jacobians(variables, self, &mut self.jacobians.borrow_mut());
        // println!("J {}", self.jacobians.borrow());
        let v0: &SE2<R> = variables.get(self.keys()[0]).unwrap();
        let th = -R::from_f64(v0.origin.log()[2]).unwrap();
        let R_inv =
            -matrix![Float::cos(th), -Float::sin(th); Float::sin(th), Float::cos(th) ].transpose();
        {
            self.jacobians
                .borrow_mut()
                .view_mut((0, 0), (2, 2))
                .copy_from(&R_inv);
        }
        // println!("R {}", R_inv);
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

#[derive(Debug, Clone)]
pub struct E2<R = f64>
where
    R: RealField,
{
    pub val: Vector2<R>,
}

impl<R> Variable<R> for E2<R>
where
    R: RealField,
{
    fn local(&self, linearization_point: &Self) -> DVector<R>
    where
        R: RealField,
    {
        let d = self.val.clone() - linearization_point.val.clone();
        let l = DVector::<R>::from_column_slice(d.as_slice());
        l
    }

    fn retract(&mut self, delta: DVectorView<R>)
    where
        R: RealField,
    {
        self.val = self.val.clone() + delta
    }

    fn dim(&self) -> usize {
        2
    }
}
impl<R> E2<R>
where
    R: RealField,
{
    pub fn new(x: f64, y: f64) -> Self {
        E2 {
            val: Vector2::new(R::from_f64(x).unwrap(), R::from_f64(y).unwrap()),
        }
    }
}

#[derive(Clone)]
struct VisionFactor<R = f64>
where
    R: RealField + Float,
{
    keys: [Vkey; 2],
    ray: Vector2<R>,
    error: RefCell<DVector<R>>,
    jacobians: RefCell<DMatrix<R>>,
    loss: GaussianLoss<R>,
}
impl<R> VisionFactor<R>
where
    R: RealField + Float,
{
    const LANDMARK_KEY: usize = 0;
    const POSE_KEY: usize = 1;
    fn new(landmark_id: Vkey, pose_id: Vkey, ray: Vector2<R>, cov: DMatrixView<R>) -> Self {
        VisionFactor {
            keys: [landmark_id, pose_id],
            ray,
            error: RefCell::new(DVector::<R>::zeros(2)),
            jacobians: RefCell::new(DMatrix::<R>::identity(2, 5)),
            loss: GaussianLoss::<R>::covariance(cov.as_view()),
        }
    }
}
impl<R> Factor<R> for VisionFactor<R>
where
    R: RealField + Float,
{
    type L = GaussianLoss<R>;

    fn error<C>(&self, variables: &Variables<C, R>) -> ErrorReturn<R>
    where
        C: VariablesContainer<R>,
    {
        let landmark_v: &E2<R> = variables
            .get(self.keys[VisionFactor::<R>::LANDMARK_KEY])
            .unwrap();
        let pose_v: &SE2<R> = variables
            .get(self.keys[VisionFactor::<R>::POSE_KEY])
            .unwrap();
        // let R_inv = pose_v.origin.inverse().matrix();
        // let R_inv = R_inv.fixed_view::<2, 2>(0, 0).to_owned();
        let th = R::from_f64(pose_v.origin.log()[2]).unwrap();
        let R_inv =
            matrix![Float::cos(th), -Float::sin(th); Float::sin(th), Float::cos(th) ].transpose();
        let p = pose_v.origin.params().fixed_rows::<2>(0);
        let l = landmark_v.val;
        // let l0 = pose_v.origin.inverse().transform(&landmark_v.val);
        let l0 = R_inv * (l - vector![R::from_f64(p[0]).unwrap(), R::from_f64(p[1]).unwrap()]);

        let r = l0.normalize();

        {
            self.error.borrow_mut().copy_from(&(r - self.ray));
        }

        // println!("err comp {}", self.error.borrow().norm());

        self.error.borrow()
    }

    fn jacobians<C>(&self, variables: &Variables<C, R>) -> JacobiansReturn<R>
    where
        C: VariablesContainer<R>,
    {
        let landmark_v: &E2<R> = variables.get(self.keys[0]).unwrap();
        let pose_v: &SE2<R> = variables.get(self.keys[1]).unwrap();
        // let R_inv = pose_v.origin.inverse().matrix();
        // let R_inv = R_inv.fixed_view::<2, 2>(0, 0).to_owned();

        let th = R::from_f64(pose_v.origin.log()[2]).unwrap();
        let R_inv =
            matrix![Float::cos(th), -Float::sin(th); Float::sin(th), Float::cos(th) ].transpose();

        let l = landmark_v.val;
        let p = pose_v.origin.params().fixed_rows::<2>(0);
        let l0 = R_inv * (l - vector![R::from_f64(p[0]).unwrap(), R::from_f64(p[1]).unwrap()]);
        // let l0 = R_inv * l - R_inv * p;

        let r = l0.normalize();
        let J_norm = (Matrix2::identity() - r * r.transpose()) / l0.norm();
        {
            self.jacobians
                .borrow_mut()
                .columns_mut(0, 2)
                .copy_from(&(J_norm * R_inv));
        }
        {
            self.jacobians
                .borrow_mut()
                .columns_mut(2, 2)
                .copy_from(&(-J_norm));
        }
        {
            let th = R::from_f64(pose_v.origin.log()[2]).unwrap();
            let x = landmark_v.val[0] - R::from_f64(pose_v.origin.params()[0]).unwrap();
            let y = landmark_v.val[1] - R::from_f64(pose_v.origin.params()[1]).unwrap();

            self.jacobians.borrow_mut().columns_mut(4, 1).copy_from(
                &(J_norm
                    * Vector2::new(
                        -x * Float::sin(th) + y * Float::cos(th),
                        -x * Float::cos(th) - y * Float::sin(th),
                    )),
            );
        }
        // println!("an J: {}", self.jacobians.borrow());
        // compute_numerical_jacobians(variables, self, &mut self.jacobians.borrow_mut());
        // println!("num J: {}", self.jacobians.borrow());
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

struct Landmark {
    id: Vkey,
    obs_cnt: usize,
    poses_keys: Vec<Vkey>,
    triangulated: bool,
    factors: Vec<VisionFactor>,
}
impl Landmark {
    fn new<C>(variables: &mut Variables<C>, id: Vkey) -> Self
    where
        C: VariablesContainer,
    {
        Landmark {
            id,
            obs_cnt: 0,
            poses_keys: Vec::new(),
            triangulated: false,
            factors: Vec::new(),
        }
    }
    fn add_observation<C>(
        &mut self,
        factors: &mut Factors<C>,
        pose_id: Vkey,
        ray: Vector2<f64>,
        cov: &Matrix2<f64>,
    ) where
        C: FactorsContainer,
    {
        // return;
        let vf = VisionFactor::new(self.id, pose_id, ray, cov.as_view());
        if self.triangulated {
            factors.add(vf);
        } else {
            self.factors.push(vf);
        }
        self.obs_cnt += 1;
        self.poses_keys.push(pose_id);
        // self.triangulated = false;
    }
    fn need_to_remove<FC, VC>(
        &mut self,
        pose_key: Vkey,
        factors: &mut Factors<FC>,
        variables: &mut Variables<VC>,
    ) -> bool
    where
        FC: FactorsContainer,
        VC: VariablesContainer,
    {
        let mut rem_poses = 0_usize;
        if let Some(_idx) = self.poses_keys.iter().position(|v| *v == pose_key) {
            rem_poses += 1;
        }
        self.poses_keys.len() == rem_poses
    }
    fn remove_pose<FC, VC>(
        &mut self,
        pose_key: Vkey,
        factors: &mut Factors<FC>,
        variables: &mut Variables<VC>,
    ) -> bool
    where
        FC: FactorsContainer,
        VC: VariablesContainer,
    {
        // if !self.triangulated {
        //     return false;
        // }
        if let Some(idx) = self.poses_keys.iter().position(|v| *v == pose_key) {
            self.poses_keys.remove(idx);
            let fcnt = self
                .factors
                .iter()
                .filter(|f| {
                    (f.keys()[VisionFactor::<f64>::POSE_KEY] == pose_key
                        && f.keys()[VisionFactor::<f64>::LANDMARK_KEY] == self.id)
                })
                .count();
            println!("fcnt {}", fcnt);
            self.factors.retain(|f| {
                // factors.retain(|f| {
                !(f.keys()[VisionFactor::<f64>::POSE_KEY] == pose_key
                    && f.keys()[VisionFactor::<f64>::LANDMARK_KEY] == self.id)
            });
            factors.get_vec_mut::<VisionFactor<f64>>().retain(|f| {
                !(f.keys()[VisionFactor::<f64>::POSE_KEY] == pose_key
                    && f.keys()[VisionFactor::<f64>::LANDMARK_KEY] == self.id)
            });
        }
        if self.need_to_remove(pose_key, factors, variables) {
            if self.triangulated {
                // assert!(variables.get_map_mut::<E2>().remove(&self.id).is_some());
                variables.remove(self.id, factors);
            }
            // self.triangulated = false;
            return true;
        }
        return false;
    }
    fn triangulate<FC, VC>(&mut self, factors: &mut Factors<FC>, variables: &mut Variables<VC>)
    where
        FC: FactorsContainer,
        VC: VariablesContainer,
    {
        if self.obs_cnt < 2 || self.triangulated {
            return;
        }
        let mut rays = Vec::<Vector2<f64>>::new();
        for vf in &self.factors {
            let r = vf.ray;
            rays.push(r);
        }
        let mut max_ang = 0.0;
        for i in 0..rays.len() {
            for j in 0..rays.len() {
                if i == j {
                    continue;
                }
                let ri = &rays[i];
                let rj = &rays[j];
                let ang = ri.dot(rj).acos().to_degrees().abs();

                if ang > max_ang {
                    max_ang = ang;
                }
            }
        }
        // println!("max_ang {}", max_ang);
        if max_ang < 5.0 {
            return;
        }

        let mut A = Matrix2::<f64>::zeros();
        let mut b = Vector2::<f64>::zeros();
        let mut rays = Vec::<Vector2<f64>>::new();
        let mut poses = Vec::<Vector2<f64>>::new();
        for p_key in &self.poses_keys {
            let pose: &SE2 = variables.get(*p_key).unwrap();
            let th = pose.origin.log()[2];
            let R = matrix![th.cos(), -th.sin(); th.sin(), th.cos() ];
            for f_idx in 0..self.factors.len() {
                let vf = self.factors.get(f_idx).unwrap();
                if vf.keys()[1] == *p_key {
                    let p = pose.origin.params().fixed_rows::<2>(0);
                    let r = R * vf.ray;
                    let Ai = Matrix2::<f64>::identity() - r * r.transpose();
                    A += Ai;
                    b += Ai * p;
                    rays.push(r);
                    poses.push(Vector2::<f64>::new(p[0], p[1]));
                }
            }
        }
        let chol = A.cholesky();
        if chol.is_some() {
            let coord = chol.unwrap().solve(&b);
            for i in 0..poses.len() {
                let p = &poses[i];
                let r = &rays[i];
                let nr = (coord - p).normalize();
                let ang = r.dot(&nr).acos().to_degrees().abs();
                if ang > 0.1 {
                    return;
                }
            }
            for f in &self.factors {
                factors.add(f.clone());
            }
            self.factors.clear();
            variables.add(self.id, E2::new(coord[0], coord[1]));
            self.triangulated = true;
        }
    }
}
/// Simple program to greet a person
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// make gif animation
    #[arg(short, long, action)]
    do_viz: bool,
}
fn fmt_f64(num: f64, width: usize, precision: usize, exp_pad: usize) -> String {
    let mut num = format!("{:.precision$e}", num, precision = precision);
    // Safe to `unwrap` as `num` is guaranteed to contain `'e'`
    let exp = num.split_off(num.find('e').unwrap());

    let (sign, exp) = if exp.starts_with("e-") {
        ('-', &exp[2..])
    } else {
        ('+', &exp[1..])
    };
    num.push_str(&format!("e{}{:0>pad$}", sign, exp, pad = exp_pad));

    format!("{:>width$}", num, width = width)
}
fn generate_colors(count: usize) -> Vec<RGBColor> {
    println!("generate_colors");
    (0..count)
        .into_iter()
        .map(|_| {
            let c = RandomColor::new().to_rgb_array();
            RGBColor(c[0], c[1], c[2])
        })
        .collect()
}
lazy_static! {
    static ref RANDOM_COLORS: Vec<RGBColor> = generate_colors(1000);
}
#[allow(non_snake_case)]
fn draw<FC, VC, T>(
    step: usize,
    iteration: usize,
    error: f64,
    factors: &Factors<FC>,
    variables: &Variables<VC>,
    xrange: (f64, f64),
    yrange: (f64, f64),
    img_size: (i32, i32),
    root_screen: &DrawingArea<T, Shift>,
    win_size: usize,
    id: usize,
    gt_poses: &Vec<Vector3<f64>>,
    poses_keys: &VecDeque<Vkey>,
    landmarks_cnt: usize,
    poses_history: &Vec<Vector2<f64>>,
) where
    FC: FactorsContainer,
    VC: VariablesContainer,
    T: DrawingBackend,
{
    let (min_x, max_x) = xrange;
    let (min_y, max_y) = yrange;
    let scene_w = max_x - min_x;
    let scene_h = max_y - min_y;
    let (img_w, img_h) = img_size;
    let marg = 0;
    let root = if scene_h > scene_w {
        let scale = img_h as f64 / scene_h;
        let sc_w = ((max_x - min_x) * scale) as i32;
        root_screen
            .margin(marg, marg, marg, marg)
            .apply_coord_spec(Cartesian2d::<RangedCoordf64, RangedCoordf64>::new(
                min_x..max_y,
                max_y..min_y,
                (img_w / 2 - sc_w / 2..sc_w * 2, 0..img_h),
            ))
    } else {
        let scale = img_h as f64 / scene_h;
        let sc_h = ((max_y - min_y) * scale) as i32;
        root_screen
            .margin(marg, marg, marg, marg)
            .apply_coord_spec(Cartesian2d::<RangedCoordf64, RangedCoordf64>::new(
                min_x..max_y,
                max_y..min_y,
                (0..img_w, img_h / 2 - sc_h / 2..sc_h * 2),
            ))
    };

    root.fill(&BLACK).unwrap();
    for i in 1..poses_history.len() {
        let p0 = poses_history[i - 1];
        let p1 = poses_history[i];
        root.draw(&PathElement::new(
            vec![(p0[0], p0[1]), (p1[0], p1[1])],
            GREEN,
        ))
        .unwrap();
    }
    if id >= win_size - 1 {
        for wi in 0..win_size {
            let p_id = Vkey(id + landmarks_cnt - win_size + wi + 1);
            // let p_id = Key(id + landmarks_init.len());
            let v = variables.get::<SE2>(p_id).unwrap();
            let th = v.origin.log()[2];
            let R = matrix![th.cos(), -th.sin(); th.sin(), th.cos() ];
            for vf in factors.get_vec::<VisionFactor>() {
                // if vf.keys()[VisionFactor::LANDMARK_KEY].0 != 64 {
                //     continue;
                // }
                if vf.keys()[VisionFactor::<f64>::POSE_KEY] == p_id {
                    let l = variables.get::<E2>(vf.keys()[VisionFactor::<f64>::LANDMARK_KEY]);
                    if l.is_some() {
                        let l = l.unwrap();
                        let p0 = v.origin.params().fixed_rows::<2>(0);
                        let r = (R * vf.ray).normalize();
                        assert!(r.norm() > 0.99 && r.norm() < 1.000001);
                        let p1 = p0 + r * (l.val - p0).norm();
                        root.draw(&PathElement::new(
                            vec![(p0[0], p0[1]), (p1[0], p1[1])],
                            RANDOM_COLORS[vf.keys()[VisionFactor::<f64>::LANDMARK_KEY].0
                                % RANDOM_COLORS.len()],
                        ))
                        .unwrap();
                    }
                }
            }
        }
    }
    for (_k, v) in variables.get_map::<SE2>().iter() {
        root.draw(&Circle::new(
            (v.origin.params()[0], v.origin.params()[1]),
            3,
            Into::<ShapeStyle>::into(RGBColor(0, 255, 0)).filled(),
        ))
        .unwrap();
    }
    for (k, v) in variables.get_map::<E2>().iter() {
        root.draw(&Circle::new(
            (v.val[0], v.val[1]),
            2,
            Into::<ShapeStyle>::into(RANDOM_COLORS[k.0 % RANDOM_COLORS.len()]).filled(),
        ))
        .unwrap();
        root.draw(&Text::new(
            format!("{}", k.0),
            (v.val[0], v.val[1]),
            ("sans-serif", 15.0)
                .into_font()
                .color(&RGBColor(255, 255, 255)),
        ))
        .unwrap();
    }
    for (i, f) in factors
        .get_vec::<GPSPositionFactor<f64>>()
        .iter()
        .enumerate()
    {
        root.draw(&Circle::new(
            (f.pose[0], f.pose[1]),
            4,
            Into::<ShapeStyle>::into(RANDOM_COLORS[i % RANDOM_COLORS.len()]).filled(),
        ))
        .unwrap();
    }

    for idx in 0..gt_poses.len() - 1 {
        let p0 = gt_poses[idx];
        let p1 = gt_poses[idx + 1];
        let p0 = p0.fixed_rows::<2>(0);
        let p1 = p1.fixed_rows::<2>(0);
        root.draw(&PathElement::new(
            vec![(p0[0], p0[1]), (p1[0], p1[1])],
            BLUE,
        ))
        .unwrap();
    }
    for idx in 0..poses_keys.len() - 1 {
        let key_0 = poses_keys[idx];
        let key_1 = poses_keys[idx + 1];
        let v0 = variables.get::<SE2>(key_0);
        let v1 = variables.get::<SE2>(key_1);
        if v0.is_some() && v1.is_some() {
            let p0 = v0.unwrap().origin.params();
            let p1 = v1.unwrap().origin.params();
            root.draw(&PathElement::new(
                vec![(p0[0], p0[1]), (p1[0], p1[1])],
                RGBColor(0, 255, 0),
            ))
            .unwrap();
        }
    }
    for idx in 0..poses_keys.len() {
        let key_0 = poses_keys[idx];
        let v0 = variables.get::<SE2>(key_0);
        if v0.is_some() {
            let p0 = v0.unwrap().origin.params();
            let R = v0.unwrap().origin.matrix();
            let R = R.fixed_view::<2, 2>(0, 0).to_owned();
            let th = v0.unwrap().origin.log()[2];
            let R = matrix![th.cos(), -th.sin(); th.sin(), th.cos()];
            // println!("R det {}", R.determinant());
            let len = 0.6;
            let ux = R * Vector2::<f64>::new(len, 0.0);
            let uy = R * Vector2::<f64>::new(0.0, len);
            root.draw(&PathElement::new(
                vec![(p0[0], p0[1]), (p0[0] + ux[0], p0[1] + ux[1])],
                Into::<ShapeStyle>::into(RGBColor(255, 0, 0)).stroke_width(2),
            ))
            .unwrap();
            root.draw(&PathElement::new(
                vec![(p0[0], p0[1]), (p0[0] + uy[0], p0[1] + uy[1])],
                Into::<ShapeStyle>::into(RGBColor(0, 255, 0)).stroke_width(2),
            ))
            .unwrap();
        }
    }
    root_screen
        .draw(&Rectangle::new(
            [(6, 3), (410, 25)],
            ShapeStyle {
                color: RGBAColor(255, 165, 0, 1.0),
                filled: true,
                stroke_width: 2,
            },
        ))
        .unwrap();
    root_screen
        .draw(&Text::new(
            format!(
                "step: {} iteration: {} error: {}",
                step,
                iteration,
                fmt_f64(error, 10, 3, 2)
            ),
            (10, 5),
            ("sans-serif", 25.0).into_font(),
        ))
        .unwrap();
    root.present().unwrap();
}
fn write_mat<R>(mat: DMatrixView<R>, path: &str)
where
    R: RealField + Float,
{
    let mut file = File::create(path).unwrap();
    for r in 0..mat.nrows() {
        for c in 0..mat.ncols() {
            file.write_fmt(format_args!("{} ", mat[(r, c)])).unwrap();
        }
        file.write_fmt(format_args!("\n")).unwrap();
    }
}
fn write_vec<R>(vec: DVectorView<R>, path: &str)
where
    R: RealField + Float,
{
    let mut file = File::create(path).unwrap();
    for r in 0..vec.nrows() {
        file.write_fmt(format_args!("{}\n", vec[r])).unwrap();
    }
}

#[allow(non_snake_case)]
fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    let variables_container = ().and_variable::<SE2>().and_variable::<E2>();
    let factors_container =
        ().and_factor::<BetweenFactor<GaussianLoss>>()
            .and_factor::<PriorFactor<ScaleLoss>>()
            .and_factor::<VisionFactor>()
            .and_factor::<GPSPositionFactor>()
            .and_factor::<BetweenFactor<DiagonalLoss>>();
    let factors_container =
        add_dense_marginalize_prior_factor(&variables_container, factors_container);
    // let mut params = GaussNewtonOptimizerParams::default();
    // params.base.verbosity_level = NonlinearOptimizerVerbosityLevel::Iteration;
    // let mut optimizer = NonlinearOptimizer::new(GaussNewtonOptimizer::with_params(params));
    let mut params = LevenbergMarquardtOptimizerParams::default();
    params.base.verbosity_level = NonlinearOptimizerVerbosityLevel::Iteration;
    let mut factor_graph = FactorGraph::new(
        factors_container,
        variables_container,
        LevenbergMarquardtOptimizer::with_params(params),
    );
    let factor_graph_viz = FactorGraphViz::default();
    // println!("current dir {:?}", current_dir().unwrap());
    let landmarks_filename = current_dir().unwrap().join("data").join("landmarks.txt");
    let observations_filename = current_dir().unwrap().join("data").join("observations.txt");
    let odometry_filename = current_dir().unwrap().join("data").join("odometry.txt");
    let gt_filename = current_dir().unwrap().join("data").join("gt.txt");
    let gps_filename = current_dir().unwrap().join("data").join("gps.txt");
    let mut landmarks_init = Vec::<Vector2<f64>>::new();
    let mut poses_history = Vec::<Vector2<f64>>::new();

    for (_id, line) in read_to_string(landmarks_filename)
        .unwrap()
        .lines()
        .enumerate()
    {
        let mut l = line.split_whitespace();
        let x = l.next().unwrap().parse::<f64>()?;
        let y = l.next().unwrap().parse::<f64>()?;
        // variables.add(Key(id), E2::new(x, y));
        landmarks_init.push(Vector2::new(x, y));
    }
    let mut landmarks = HashMap::<Vkey, Landmark>::default();

    let mut poses_keys = VecDeque::<Vkey>::new();
    let mut var_id: usize = 0;
    let mut odom_lines: Vec<String> = read_to_string(odometry_filename)
        .unwrap()
        .lines()
        .map(|s| String::from(s))
        .collect();
    let mut gps_lines: Vec<String> = read_to_string(gps_filename)
        .unwrap()
        .lines()
        .map(|s| String::from(s))
        .collect();

    let mut gt_poses = Vec::<Vector3<f64>>::new();
    for (_id, line) in read_to_string(gt_filename).unwrap().lines().enumerate() {
        let mut l = line.split_whitespace();
        let x = l.next().unwrap().parse::<f64>()?;
        let y = l.next().unwrap().parse::<f64>()?;
        let th = l.next().unwrap().parse::<f64>()?;
        gt_poses.push(Vector3::new(x, y, th));
    }
    let mut min_x = f64::MAX;
    let mut max_x = f64::MIN;
    let mut min_y = f64::MAX;
    let mut max_y = f64::MIN;
    for gt in &gt_poses {
        min_x = min_x.min(gt[0]);
        max_x = max_x.max(gt[0]);
        min_y = min_y.min(gt[1]);
        max_y = max_y.max(gt[1]);
    }

    const OUTPUT_GIF: &str = "2d-slam.gif";
    let img_w = 1024_i32;
    let img_h = 768_i32;
    let root_screen = BitMapBackend::gif(OUTPUT_GIF, (img_w as u32, img_h as u32), 100)
        .unwrap()
        .into_drawing_area();

    let mut step: usize = 0;

    for (id, line) in read_to_string(observations_filename)
        .unwrap()
        .lines()
        .enumerate()
    {
        let mut l = line.split_whitespace();
        let x = l.next().unwrap().parse::<f64>()?;
        let y = l.next().unwrap().parse::<f64>()?;
        let th = l.next().unwrap().parse::<f64>()?;
        let pose_id = Vkey(id + landmarks_init.len());

        factor_graph.add_variable(pose_id, SE2::new(x, y, th));
        if id > 0 {
            let mut l = odom_lines[id - 1].split_whitespace();
            let dx = l.next().unwrap().parse::<f64>()?;
            let dy = l.next().unwrap().parse::<f64>()?;
            let dth = l.next().unwrap().parse::<f64>()?;
            let sigx = l.next().unwrap().parse::<f64>()?;
            let sigy = l.next().unwrap().parse::<f64>()?;
            let sigth = l.next().unwrap().parse::<f64>()?;

            let dse2 = SE2::<f64>::new(dx, dy, dth);
            let pose0 = factor_graph
                .get_variable::<SE2>(Vkey(id + landmarks_init.len() - 1))
                .unwrap()
                .origin
                .clone();
            let pose1: &mut SE2 = factor_graph
                .get_variable_mut(Vkey(id + landmarks_init.len()))
                .unwrap();
            pose1.origin = pose0.multiply(&dse2.origin);
            // pose1.origin = pose0;
            factor_graph.add_factor(BetweenFactor::new(
                Vkey(id + landmarks_init.len() - 1),
                Vkey(id + landmarks_init.len()),
                dx,
                dy,
                dth,
                Some(DiagonalLoss::sigmas(&dvector![sigx, sigy, sigth].as_view())),
            ));
        }
        {
            let mut l = gps_lines[id].split_whitespace();
            let gpsx = l.next().unwrap().parse::<f64>()?;
            let gpsy = l.next().unwrap().parse::<f64>()?;
            let sigx = l.next().unwrap().parse::<f64>()?;
            let sigy = l.next().unwrap().parse::<f64>()?;
            // factor_graph.add_factor(GPSPositionFactor::new(
            //     Vkey(id + landmarks_init.len()),
            //     vector![gpsx, gpsy],
            //     vector![sigx, sigy],
            // ));
        }

        poses_keys.push_back(pose_id);
        let last_pose_key = *poses_keys.front().unwrap();
        let last_pose: &SE2 = factor_graph.get_variable(last_pose_key).unwrap();
        let lx = last_pose.origin.params()[0];
        let ly = last_pose.origin.params()[1];
        let lth = last_pose.origin.log()[2];

        if id == 0 {
            factor_graph.add_factor(PriorFactor::new(
                *poses_keys.front().unwrap(),
                lx,
                ly,
                lth,
                Some(ScaleLoss::scale(1e5)),
            ));
        }
        let mut sA = Matrix2::<f64>::zeros();
        let mut sb = Vector2::<f64>::zeros();
        let rays_cnt = l.next().unwrap().parse::<usize>()?;

        let R = matrix![th.cos(), -th.sin(); th.sin(), th.cos() ];
        for _ in 0..rays_cnt {
            let id = l.next().unwrap().parse::<usize>()?;
            let rx = l.next().unwrap().parse::<f64>()?;
            let ry = l.next().unwrap().parse::<f64>()?;
            let sx = l.next().unwrap().parse::<f64>()?;
            let sy = l.next().unwrap().parse::<f64>()?;
            let sxy = l.next().unwrap().parse::<f64>()?;
            landmarks
                .entry(Vkey(id))
                .or_insert_with(|| Landmark::new(&mut factor_graph.variables, Vkey(id)));

            landmarks.get_mut(&Vkey(id)).unwrap().add_observation(
                &mut factor_graph.factors,
                pose_id,
                Vector2::<f64>::new(rx, ry),
                &Matrix2::new(sx, sxy, sxy, sy),
            );

            // if let Some(l) = variables.get::<E2>(Key(id)) {
            //     let r = R * Vector2::<f64>::new(rx, ry);
            //     let I = Matrix2::identity();
            //     let A = I - r * r.transpose();
            //     let l = l.val;
            //     sA += A.transpose() * A;
            //     sb += A.transpose() * A * l;
            //     pcnt += 1;
            // }
        }

        for l in landmarks.values_mut().into_iter() {
            l.triangulate(&mut factor_graph.factors, &mut factor_graph.variables);
        }
        // let pose: &mut SE2 = variables.get_mut(Key(id + landmarks_init.len())).unwrap();
        // let mut pp = pose.origin.params().clone();

        // if pcnt > 6 {
        //     let chol = sA.cholesky();
        //     if chol.is_some() {
        //         let coord = chol.unwrap().solve(&sb);
        //         // let lp = pose.origin.log();
        //         // coord =
        //         // println!("old params {}", pp);
        //         pp[0] = coord[0];
        //         pp[1] = coord[1];
        //         // println!("new params {}", pp);
        //         // println!("new lp {}", lp);
        //         pose.origin.set_params(&pp);
        //     }
        // }

        assert_eq!(factor_graph.unused_variables_count(), 0);
        let start = Instant::now();
        let win_size = 6;
        let opt_res = if args.do_viz {
            let opt_params = OptParams::builder()
                .callback(
                    |iteration, error, factors: &Factors<_, _>, variables: &Variables<_, _>| {
                        draw(
                            step,
                            iteration,
                            error,
                            factors,
                            variables,
                            (min_x, max_x),
                            (min_y, max_y),
                            (img_w, img_h),
                            &root_screen,
                            win_size,
                            id,
                            &gt_poses,
                            &poses_keys,
                            landmarks_init.len(),
                            &poses_history,
                        )
                    },
                )
                .build();
            let res = factor_graph.optimize(opt_params);
            root_screen
                .present()
                .expect("Unable to write result to file");
            println!("{} saved!", OUTPUT_GIF);

            res
        } else {
            let opt_params = <OptParams<_, _, _>>::builder().build();
            factor_graph.optimize(opt_params)
        };

        if poses_keys.len() >= win_size {
            let key = poses_keys.pop_front().unwrap();
            let lkeys = landmarks
                .iter_mut()
                .map(|(k, l)| {
                    (
                        *k,
                        l.need_to_remove(
                            key,
                            &mut factor_graph.factors,
                            &mut factor_graph.variables,
                        ) && l.triangulated,
                    )
                })
                .collect::<Vec<(Vkey, bool)>>();
            let mut keys_to_marg = Vec::<Vkey>::default();
            for (k, b) in lkeys {
                if b {
                    keys_to_marg.push(k);
                }
            }
            keys_to_marg.push(key);
            println!("keys to marg {:?}", keys_to_marg);
            let marg_prior = marginalize(
                &keys_to_marg,
                &factor_graph.factors,
                &factor_graph.variables,
            )
            .unwrap();
            let n_variables = factor_graph.factors.neighborhood_variables(&keys_to_marg);
            factor_graph_viz.generate_dot(
                &factor_graph,
                Some(vec![
                    HighlightVariablesGroup::new(keys_to_marg.clone(), "m-variables"),
                    HighlightVariablesGroup::new(n_variables, "n-variables"),
                ]),
                // None,
                // None,
                Some(vec![HighlightFactorsGroup::new(
                    factor_graph
                        .factors
                        .connected_factors_indexes(&keys_to_marg),
                    "m-connected-factors",
                )]),
            );
            // panic!("exit");
            factor_graph.add_factor(marg_prior);
            // if !variables.get_map_mut::<E2>().is_empty() {
            //     let l_keys = variables
            //         .get_map::<E2>()
            //         .keys()
            //         .map(|kv| *kv)
            //         .collect::<Vec<Vkey>>();
            //     let marg_prior = marginalize(l_keys.as_slice(), &factors, &variables);
            //     if let Some(marg_prior) = marg_prior {
            //         factors.add(marg_prior);
            //     } else {
            //         println!("no marg");
            //     }
            //     variables.get_map_mut::<E2>().clear();
            //     factors.get_vec_mut::<VisionFactor>().clear();
            //     landmarks.clear();
            // }
            let pose: &SE2 = factor_graph.get_variable(key).unwrap();
            let pose = pose.origin.params();
            poses_history.push(vector![pose[0], pose[1]]);
            let (rok, _fcnt) = factor_graph
                .variables
                .remove(key, &mut factor_graph.factors);
            assert!(rok);

            landmarks.retain(|_, l| {
                !l.remove_pose(key, &mut factor_graph.factors, &mut factor_graph.variables)
            });
            // for l in landmarks.values() {
            //     if l.obs_cnt < 2 {
            //         println!("low obs cnt");
            //     }
            // }

            // let bf = factors.get_vec_mut::<BetweenFactor<DiagonalLoss>>();
            println!("pose id {:?}", key);
            // bf.retain(|f| !(f.keys()[0] == key));
            // let removed_factors_cnt = factors.remove_conneted_factors(key);
            // assert_eq!(removed_factors_cnt, 0);
        }
        // let mut optimizer = NonlinearOptimizer::new(GaussNewtonOptimizer::default());
        // let start = Instant::now();
        let duration = start.elapsed();
        println!("optimize time: {:?}", duration);
        println!("opt_res {:?}", opt_res);
        step += 1;
    }
    Ok(())
}
