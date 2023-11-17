use lazy_static::lazy_static;
use state_variables::E2;

use core::time;
use macroquad::prelude::*;
use std::collections::VecDeque;

use std::error::Error;
use std::fs::{self, File};
use std::io::Write;

use std::thread;
use std::time::Instant;
use std::{env::current_dir, fs::read_to_string};

use clap::Parser;
use nalgebra::{
    dvector, matrix, vector, DMatrixView, DVectorView, Matrix2, RealField, Vector2, Vector3,
};
use num::Float;

use optigy::prelude::{
    add_dense_marginalize_prior_factor, DiagonalLoss, Factor, FactorGraph, Factors,
    FactorsContainer, GaussianLoss, LevenbergMarquardtOptimizer, LevenbergMarquardtOptimizerParams,
    NonlinearOptimizerVerbosityLevel, OptParams, ScaleLoss, Variables, VariablesContainer, Vkey,
};
use optigy::viz::graph_viz::FactorGraphViz;
use random_color::RandomColor;
pub mod gps_factor;
pub mod landmarks;
pub mod state_variables;
pub mod vision_factor;
use gps_factor::GPSPositionFactor;
use vision_factor::VisionFactor;

use crate::landmarks::Landmarks;

use slam_common::between_factor::BetweenFactor;
use slam_common::prior_factor::PriorFactor;
use slam_common::se2::SE2;

/// Simple program to greet a person
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// make gif animation
    #[arg(short, long, action)]
    do_viz: bool,
    #[arg(short, long, action)]
    write_gif: bool,
    #[arg(short, long, action)]
    marginalize: bool,
    #[arg(short, long, action)]
    use_gps: bool,
    #[arg(short, long, action)]
    use_vision: bool,
}
fn fmt_f64(num: f64, width: usize, precision: usize, exp_pad: usize) -> String {
    let mut num = format!("{:.precision$e}", num, precision = precision);
    // Safe to `unwrap` as `num` is guaranteed to contain `'e'`
    let exp = num.split_off(num.find('e').unwrap());
    //  let (sign, exp) = if exp.starts_with("e-") {
    //         ('-', &exp[2..])
    let (sign, exp) = if let Some(stripped) = exp.strip_prefix("e-") {
        ('-', stripped)
    } else {
        ('+', &exp[1..])
    };
    num.push_str(&format!("e{}{:0>pad$}", sign, exp, pad = exp_pad));

    format!("{:>width$}", num, width = width)
}
fn generate_colors(count: usize) -> Vec<Color> {
    println!("generate_colors");
    (0..count)
        .map(|_| {
            let c = RandomColor::new().to_rgb_array();
            Color::from_rgba(c[0], c[1], c[2], 255)
        })
        .collect()
}
lazy_static! {
    static ref RANDOM_COLORS: Vec<Color> = generate_colors(1000);
}
#[derive(Default)]
struct ScreenCamera {
    pub render_target: Option<RenderTarget>,
}

impl Camera for ScreenCamera {
    fn matrix(&self) -> Mat4 {
        let (width, height) = miniquad::window::screen_size();
        let dpi = miniquad::window::dpi_scale();
        glam::Mat4::orthographic_rh_gl(0., width / dpi, 0., height / dpi, 1., -1.)
    }

    fn depth_enabled(&self) -> bool {
        false
    }

    fn render_pass(&self) -> Option<miniquad::RenderPass> {
        self.render_target.as_ref().map(|rt| rt.render_pass)
    }

    fn viewport(&self) -> Option<(i32, i32, i32, i32)> {
        None
    }
}
#[allow(non_snake_case)]
fn draw<FC, VC>(
    step: usize,
    iteration: usize,
    error: f64,
    factors: &Factors<FC>,
    variables: &Variables<VC>,
    xrange: (f64, f64),
    yrange: (f64, f64),
    win_size: usize,
    id: usize,
    gt_poses: &Vec<Vector3<f64>>,
    poses_keys: &VecDeque<Vkey>,
    poses_history: &Vec<Vector2<f64>>,
    render_target: RenderTarget,
    write_gif: bool,
) where
    FC: FactorsContainer,
    VC: VariablesContainer,
{
    let (min_x, max_x) = xrange;
    let (min_y, max_y) = yrange;
    let w = max_x - min_x;
    let h = max_y - min_y;
    let sr = h / w;
    let rect = if screen_width() > screen_height() {
        let r = (screen_width() / screen_height()) as f64 * sr;
        let mx = min_x + w / 2.0;
        let min_x = (min_x - mx) * r + mx;
        let max_x = (max_x - mx) * r + mx;
        let w = max_x - min_x;
        let h = max_y - min_y;
        Rect::new(min_x as f32, min_y as f32, w as f32, h as f32)
    } else {
        let r = 1.0 / (screen_width() / screen_height()) as f64 / sr;
        let my = min_y + h / 2.0;
        let min_y = (min_y - my) * r + my;
        let max_y = (max_y - my) * r + my;
        let w = max_x - min_x;
        let h = max_y - min_y;
        Rect::new(min_x as f32, min_y as f32, w as f32, h as f32)
    };
    let mut camera = Camera2D::from_display_rect(rect);
    let mut screen_camera = ScreenCamera::default();
    if write_gif {
        camera.render_target = Some(render_target.clone());
        screen_camera.render_target = Some(render_target.clone());
    }

    set_camera(&camera);

    clear_background(BLACK);
    let ws = screen_width().min(screen_height());
    let thickness = 20.0f32 / ws;
    let ray_thickness = 10.0f32 / ws;
    let pose_rad = 60.0f32 / ws;
    let gps_rad = 70.0f32 / ws;
    let landmark_rad = 50.0f32 / ws;
    for i in 1..poses_history.len() {
        let p0 = poses_history[i - 1];
        let p1 = poses_history[i];
        draw_line(
            p0[0] as f32,
            p0[1] as f32,
            p1[0] as f32,
            p1[1] as f32,
            thickness,
            GREEN,
        );
    }
    if id >= win_size - 1 {
        for wi in 0..win_size {
            let p_id = poses_keys[wi];
            let v = variables.get::<SE2>(p_id).unwrap();
            let th = v.origin.log()[2];
            let R = matrix![th.cos(), -th.sin(); th.sin(), th.cos() ];
            for vf in factors.get_vec::<VisionFactor>() {
                if vf.keys()[VisionFactor::<f64>::POSE_KEY] == p_id {
                    let l = variables.get::<E2>(vf.keys()[VisionFactor::<f64>::LANDMARK_KEY]);
                    if l.is_some() {
                        let l = l.unwrap();
                        let p0 = v.origin.params().fixed_rows::<2>(0);
                        let r = (R * vf.ray()).normalize();
                        assert!(r.norm() > 0.99 && r.norm() < 1.000001);
                        let p1 = p0 + r * (l.val - p0).norm();
                        draw_line(
                            p0[0] as f32,
                            p0[1] as f32,
                            p1[0] as f32,
                            p1[1] as f32,
                            ray_thickness,
                            RANDOM_COLORS[vf.keys()[VisionFactor::<f64>::LANDMARK_KEY].0
                                % RANDOM_COLORS.len()],
                        );
                    }
                }
            }
        }
    }
    for (_k, v) in variables.get_map::<SE2>().iter() {
        draw_circle(
            v.origin.params()[0] as f32,
            v.origin.params()[1] as f32,
            landmark_rad,
            GREEN,
        );
    }
    for (k, v) in variables.get_map::<E2>().iter() {
        draw_circle(
            v.val[0] as f32,
            v.val[1] as f32,
            pose_rad,
            RANDOM_COLORS[k.0 % RANDOM_COLORS.len()],
        );
    }
    if write_gif {
        set_camera(&screen_camera);
    } else {
        set_default_camera();
    }
    for (k, v) in variables.get_map::<E2>().iter() {
        let wc = camera.world_to_screen(Vec2::new(v.val[0] as f32, v.val[1] as f32));
        // draw_text(format!("{}", k.0).as_str(), wc.x, wc.y, 35.0, WHITE)
        let ty = if write_gif {
            screen_height() - wc.y
        } else {
            wc.y
        };
        draw_text(format!("{}", k.0).as_str(), wc.x, ty, 35.0, WHITE)
    }
    set_camera(&camera);
    for (_i, f) in factors
        .get_vec::<GPSPositionFactor<f64>>()
        .iter()
        .enumerate()
    {
        draw_circle_lines(
            f.pose[0] as f32,
            f.pose[1] as f32,
            gps_rad,
            thickness,
            RANDOM_COLORS[f.keys()[0].0 % RANDOM_COLORS.len()],
        );
    }

    for idx in 0..gt_poses.len() - 1 {
        let p0 = gt_poses[idx];
        let p1 = gt_poses[idx + 1];
        let p0 = p0.fixed_rows::<2>(0);
        let p1 = p1.fixed_rows::<2>(0);
        draw_line(
            p0[0] as f32,
            p0[1] as f32,
            p1[0] as f32,
            p1[1] as f32,
            thickness,
            BLUE,
        );
    }
    for idx in 0..poses_keys.len() - 1 {
        let key_0 = poses_keys[idx];
        let key_1 = poses_keys[idx + 1];
        let v0 = variables.get::<SE2>(key_0);
        let v1 = variables.get::<SE2>(key_1);
        if v0.is_some() && v1.is_some() {
            let p0 = v0.unwrap().origin.params();
            let p1 = v1.unwrap().origin.params();
            draw_line(
                p0[0] as f32,
                p0[1] as f32,
                p1[0] as f32,
                p1[1] as f32,
                thickness,
                GREEN,
            );
        }
    }
    for idx in 0..poses_keys.len() {
        let key_0 = poses_keys[idx];
        let v0 = variables.get::<SE2>(key_0);
        if let Some(v0) = v0 {
            let p0 = v0.origin.params();
            let R = v0.origin.matrix();
            let _R = R.fixed_view::<2, 2>(0, 0).to_owned();
            let th = v0.origin.log()[2];
            let R = matrix![th.cos(), -th.sin(); th.sin(), th.cos()];
            // println!("R det {}", R.determinant());
            let len = 0.6;
            let ux = R * Vector2::<f64>::new(len, 0.0);
            let uy = R * Vector2::<f64>::new(0.0, len);
            draw_line(
                p0[0] as f32,
                p0[1] as f32,
                (p0[0] + ux[0]) as f32,
                (p0[1] + ux[1]) as f32,
                thickness,
                RED,
            );
            draw_line(
                p0[0] as f32,
                p0[1] as f32,
                (p0[0] + uy[0]) as f32,
                (p0[1] + uy[1]) as f32,
                thickness,
                GREEN,
            );
        }
    }
    if write_gif {
        set_camera(&screen_camera);
    } else {
        set_default_camera();
    }
    draw_rectangle(6.0f32, 3.0f32, 530.0f32, 30.0f32, ORANGE);
    draw_text(
        format!(
            "step: {} iteration: {} error: {}",
            step,
            iteration,
            fmt_f64(error, 10, 3, 2)
        )
        .as_str(),
        10.0f32,
        25.0f32,
        30.0f32,
        BLACK,
    );
    set_default_camera();
    // screen_camera.render_target = None;
    // set_camera(&screen_camera);
    draw_texture_ex(
        &render_target.texture,
        0.,
        0.,
        WHITE,
        DrawTextureParams {
            dest_size: Some(vec2(screen_width(), screen_height())),
            ..Default::default()
        },
    );

    if !write_gif {
        thread::sleep(time::Duration::from_millis(20));
    }
}
pub fn write_mat<R>(mat: DMatrixView<R>, path: &str)
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
pub fn write_vec<R>(vec: DVectorView<R>, path: &str)
where
    R: RealField + Float,
{
    let mut file = File::create(path).unwrap();
    for r in 0..vec.nrows() {
        file.write_fmt(format_args!("{}\n", vec[r])).unwrap();
    }
}

fn window_conf() -> Conf {
    Conf {
        window_title: "SLAM 2D".to_owned(),
        fullscreen: true,
        sample_count: 8,
        ..Default::default()
    }
}

#[allow(non_snake_case)]
#[macroquad::main(window_conf)]
#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
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
    let mut params = LevenbergMarquardtOptimizerParams::default();
    params.base.verbosity_level = NonlinearOptimizerVerbosityLevel::Iteration;
    let mut factor_graph = FactorGraph::new(
        factors_container,
        variables_container,
        LevenbergMarquardtOptimizer::with_params(params),
    );
    let mut factor_graph_viz = FactorGraphViz::default();
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
    let mut landmarks = Landmarks::default();

    let mut poses_keys = VecDeque::<Vkey>::new();
    let _var_id: usize = 0;
    let odom_lines: Vec<String> = read_to_string(odometry_filename)
        .unwrap()
        .lines()
        .map(String::from)
        .collect();
    let gps_lines: Vec<String> = read_to_string(gps_filename)
        .unwrap()
        .lines()
        .map(String::from)
        .collect();

    let mut gt_poses = Vec::<Vector3<f64>>::new();
    for (_id, line) in read_to_string(gt_filename).unwrap().lines().enumerate() {
        let mut l = line.split_whitespace();
        let x = l.next().unwrap().parse::<f64>()?;
        let y = l.next().unwrap().parse::<f64>()?;
        let th = l.next().unwrap().parse::<f64>()?;
        gt_poses.push(Vector3::new(x, y, th));
    }

    const OUTPUT_GIF: &str = "2d-slam.gif";
    let img_w = 800_i32;
    let img_h = 600_i32;

    let write_pdf = false;
    let image: Option<fs::File>;
    let mut encoder: Option<gif::Encoder<fs::File>> = None;
    if args.write_gif || args.do_viz {
        image = Some(File::create(OUTPUT_GIF).unwrap());
        encoder = Some(gif::Encoder::new(image.unwrap(), img_w as u16, img_h as u16, &[]).unwrap());
    }

    let mut prev_pose_id = Vkey(0);
    for (step, (id, line)) in read_to_string(observations_filename)
        .unwrap()
        .lines()
        .enumerate()
        .enumerate()
    {
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
        for l in factor_graph.variables().get_map::<E2>().values() {
            min_x = min_x.min(l.val.x);
            max_x = max_x.max(l.val.x);
            min_y = min_y.min(l.val.y);
            max_y = max_y.max(l.val.y);
        }
        let mut l = line.split_whitespace();
        let x = l.next().unwrap().parse::<f64>()?;
        let y = l.next().unwrap().parse::<f64>()?;
        let th = l.next().unwrap().parse::<f64>()?;
        let curr_pose_id = factor_graph.add_variable(SE2::new(x, y, th));
        if id > 0 {
            let mut l = odom_lines[id - 1].split_whitespace();
            let dx = l.next().unwrap().parse::<f64>()?;
            let dy = l.next().unwrap().parse::<f64>()?;
            let dth = l.next().unwrap().parse::<f64>()?;
            let sigx = l.next().unwrap().parse::<f64>()?;
            let sigy = l.next().unwrap().parse::<f64>()?;
            let sigth = l.next().unwrap().parse::<f64>()?;

            let dse2 = SE2::<f64>::new(dx, dy, dth);
            let prev_pose = factor_graph
                .get_variable::<SE2>(prev_pose_id)
                .unwrap()
                .origin;
            let curr_pose = factor_graph.get_variable_mut::<SE2>(curr_pose_id).unwrap();
            curr_pose.origin = prev_pose.multiply(&dse2.origin);
            // pose1.origin = pose0;
            factor_graph.add_factor(BetweenFactor::new(
                prev_pose_id,
                curr_pose_id,
                dx,
                dy,
                dth,
                Some(DiagonalLoss::sigmas(&dvector![sigx, sigy, sigth].as_view())),
            ));
            prev_pose_id = curr_pose_id;
        }
        {
            let mut l = gps_lines[id].split_whitespace();
            let gpsx = l.next().unwrap().parse::<f64>()?;
            let gpsy = l.next().unwrap().parse::<f64>()?;
            let sigx = l.next().unwrap().parse::<f64>()?;
            let sigy = l.next().unwrap().parse::<f64>()?;
            if args.use_gps {
                factor_graph.add_factor(GPSPositionFactor::new(
                    curr_pose_id,
                    vector![gpsx, gpsy],
                    vector![sigx, sigy],
                ));
            }
        }

        poses_keys.push_back(curr_pose_id);
        let last_pose_key = *poses_keys.front().unwrap();
        let last_pose: &SE2 = factor_graph.get_variable(last_pose_key).unwrap();
        let lx = last_pose.origin.params()[0];
        let ly = last_pose.origin.params()[1];
        let lth = last_pose.origin.log()[2];
        if args.marginalize {
            if id == 0 {
                factor_graph.add_factor(PriorFactor::new(
                    *poses_keys.front().unwrap(),
                    lx,
                    ly,
                    lth,
                    Some(ScaleLoss::scale(1e5)),
                ));
            }
        } else {
            factor_graph.add_factor(PriorFactor::new(
                *poses_keys.front().unwrap(),
                lx,
                ly,
                lth,
                Some(ScaleLoss::scale(1e5)),
            ));
        }
        let _sA = Matrix2::<f64>::zeros();
        let _sb = Vector2::<f64>::zeros();
        let rays_cnt = l.next().unwrap().parse::<usize>()?;

        let _R = matrix![th.cos(), -th.sin(); th.sin(), th.cos() ];
        for _ in 0..rays_cnt {
            let id = l.next().unwrap().parse::<usize>()?;
            let rx = l.next().unwrap().parse::<f64>()?;
            let ry = l.next().unwrap().parse::<f64>()?;
            let sx = l.next().unwrap().parse::<f64>()?;
            let sy = l.next().unwrap().parse::<f64>()?;
            let sxy = l.next().unwrap().parse::<f64>()?;
            let landmark_id = factor_graph.map_key(Vkey(id));
            if args.use_vision {
                landmarks.add_observation(
                    &mut factor_graph,
                    curr_pose_id,
                    landmark_id,
                    rx,
                    ry,
                    sx,
                    sy,
                    sxy,
                );
            }

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

        landmarks.triangulate(&mut factor_graph);
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

        let mut ren_tar: Option<RenderTarget> = None;
        if args.do_viz {
            ren_tar = Some(render_target(img_w as u32, img_h as u32));
            ren_tar
                .as_mut()
                .unwrap()
                .texture
                .set_filter(FilterMode::Nearest);
        }
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
                            win_size,
                            id,
                            &gt_poses,
                            &poses_keys,
                            &poses_history,
                            ren_tar.as_ref().unwrap().clone(),
                            args.write_gif,
                        )
                    },
                )
                .build();
            println!("opt");
            let res = factor_graph.optimize(opt_params);
            next_frame().await;
            if args.write_gif {
                let mut frame = ren_tar.as_ref().unwrap().texture.get_texture_data();
                let frame = gif::Frame::from_rgba(frame.width, frame.height, &mut frame.bytes);
                encoder.as_mut().unwrap().write_frame(&frame).unwrap();
            }

            res
        } else {
            let opt_params = <OptParams<_, _, _>>::builder().build();
            factor_graph.optimize(opt_params)
        };
        // if poses_keys.len() > 3 {
        //     return Ok(());
        // }
        // if step == 6 {
        //     return Ok(());
        // }

        if poses_keys.len() >= win_size {
            let first_pose_id = poses_keys.pop_front().unwrap();
            if write_pdf {
                factor_graph_viz.add_page(&factor_graph, None, None, &format!("Step {}", step));
            }
            let pose: &SE2 = factor_graph.get_variable(first_pose_id).unwrap();
            let pose = pose.origin.params();
            poses_history.push(vector![pose[0], pose[1]]);

            landmarks.proc_pose_remove(&mut factor_graph, first_pose_id, args.marginalize);
            factor_graph.remove_variable(first_pose_id, args.marginalize);
            println!("pose id {:?}", first_pose_id);
        }
        let duration = start.elapsed();
        println!("optimize time: {:?}", duration);
        println!("opt_res {:?}", opt_res);
    }
    Ok(())
}
