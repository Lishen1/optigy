use std::error::Error;
use std::time::Instant;
use std::{env::current_dir, fs::read_to_string};

use clap::Parser;
use nalgebra::Matrix3;
use optigy::core::loss_function::ScaleLoss;

use optigy::nonlinear::levenberg_marquardt_optimizer::{
    LevenbergMarquardtOptimizer, LevenbergMarquardtOptimizerParams,
};
use optigy::prelude::{
    Factors, FactorsContainer, GaussianLoss, NonlinearOptimizer, Variables, VariablesContainer,
    Vkey,
};
use plotters::coord::types::RangedCoordf64;
use plotters::prelude::*;
use plotters::style::full_palette::BLACK;

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
}
#[allow(non_snake_case)]
fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    let container = ().and_variable::<SE2>();
    let mut variables = Variables::new(container);

    let container =
        ().and_factor::<BetweenFactor<GaussianLoss>>()
            .and_factor::<PriorFactor<ScaleLoss>>();
    let mut factors = Factors::new(container);
    // println!("current dir {:?}", current_dir().unwrap());
    let filename = current_dir()
        .unwrap()
        .join("data")
        .join("input_M3500_g2o.g2o");
    for line in read_to_string(filename).unwrap().lines() {
        let mut l = line.split_whitespace();
        let strhead = l.next().unwrap();
        match strhead {
            "VERTEX_SE2" => {
                let id = l.next().unwrap().parse::<usize>()?;
                let x = l.next().unwrap().parse::<f64>()?;
                let y = l.next().unwrap().parse::<f64>()?;
                let th = l.next().unwrap().parse::<f64>()?;
                // println!("id: {}, x: {} y: {} th: {}", id, x, y, th);
                variables.add(Vkey(id), SE2::new(x, y, th));
            }
            "EDGE_SE2" => {
                let ido = l.next().unwrap().parse::<usize>()?;
                let idi = l.next().unwrap().parse::<usize>()?;
                let dx = l.next().unwrap().parse::<f64>()?;
                let dy = l.next().unwrap().parse::<f64>()?;
                let dth = l.next().unwrap().parse::<f64>()?;
                let i11 = l.next().unwrap().parse::<f64>()?;
                let i12 = l.next().unwrap().parse::<f64>()?;
                let i13 = l.next().unwrap().parse::<f64>()?;
                let i22 = l.next().unwrap().parse::<f64>()?;
                let i23 = l.next().unwrap().parse::<f64>()?;
                let i33 = l.next().unwrap().parse::<f64>()?;
                let I = Matrix3::new(i11, i12, i13, i12, i22, i23, i13, i23, i33);
                // println!(
                //     "ido: {} idi: {} dx: {} dy: {} dth: {}",
                //     ido, idi, dx, dy, dth
                // );
                // println!("I: {}", I);
                factors.add(BetweenFactor::new(
                    Vkey(ido),
                    Vkey(idi),
                    dx,
                    dy,
                    dth,
                    Some(GaussianLoss::information(I.as_view())),
                ));
                // factors.add(BetweenFactor::new(Key(ido), Key(idi), dx, dy, dth, None));
            }
            &_ => (),
        }
        // println!("line: {}", line);
    }
    let v0: &SE2 = variables.get(Vkey(0)).unwrap();
    factors.add(PriorFactor::from_se2(
        Vkey(0),
        v0.origin,
        Some(ScaleLoss::scale(1.0)),
    ));

    const OUTPUT_GIF: &str = "pose_graph.gif";

    let params = LevenbergMarquardtOptimizerParams::default();
    // params.base.verbosity_level = NonlinearOptimizerVerbosityLevel::Subiteration;
    let mut optimizer = NonlinearOptimizer::new(LevenbergMarquardtOptimizer::with_params(params));
    // let mut optimizer = NonlinearOptimizer::new(LevenbergMarquardtOptimizer::default());
    let start = Instant::now();
    let opt_res = if args.do_viz {
        let img_w = 1024_i32;
        let img_h = 768_i32;
        let root_screen = BitMapBackend::gif(OUTPUT_GIF, (img_w as u32, img_h as u32), 1000)
            .unwrap()
            .into_drawing_area();
        let res = optimizer.optimize_with_callback(
            &factors,
            &mut variables,
            Some(
                |iteration, error, factors2: &Factors<_, _>, variables2: &Variables<_, _>| {
                    println!("iteration: {} error: {}", iteration, error);

                    let mut min_x = f64::MAX;
                    let mut max_x = f64::MIN;
                    let mut min_y = f64::MAX;
                    let mut max_y = f64::MIN;
                    for key in variables2.default_variable_ordering().keys() {
                        let v: &SE2 = variables2.get(*key).unwrap();
                        min_x = min_x.min(v.origin.params()[0]);
                        max_x = max_x.max(v.origin.params()[0]);
                        min_y = min_y.min(v.origin.params()[1]);
                        max_y = max_y.max(v.origin.params()[1]);
                    }
                    let root = root_screen.apply_coord_spec(Cartesian2d::<
                        RangedCoordf64,
                        RangedCoordf64,
                    >::new(
                        min_x..max_x,
                        min_y..max_y,
                        (0..img_w, 0..img_h),
                    ));
                    root.fill(&BLACK).unwrap();
                    // println!("iter {}", iteration);
                    for key in variables2.default_variable_ordering().keys() {
                        let v: &SE2 = variables2.get(*key).unwrap();
                        // Draw an circle on the drawing area
                        root.draw(&Circle::new(
                            (v.origin.params()[0], v.origin.params()[1]),
                            3,
                            Into::<ShapeStyle>::into(GREEN).filled(),
                        ))
                        .unwrap();
                    }
                    for f_idx in 0..factors2.len() {
                        let keys = factors2.keys_at(f_idx).unwrap();
                        if keys.len() == 1 {
                            continue;
                        }

                        let v0: &SE2 = variables2.get(keys[0]).unwrap();
                        let v1: &SE2 = variables2.get(keys[1]).unwrap();
                        root.draw(&PathElement::new(
                            vec![
                                (v0.origin.params()[0], v0.origin.params()[1]),
                                (v1.origin.params()[0], v1.origin.params()[1]),
                            ],
                            RED,
                        ))
                        .unwrap();
                    }
                    root_screen
                        .draw(&Rectangle::new(
                            [(6, 3), (300, 25)],
                            ShapeStyle {
                                color: RGBAColor(255, 165, 0, 1.0),
                                filled: true,
                                stroke_width: 2,
                            },
                        ))
                        .unwrap();
                    root_screen
                        .draw(&Text::new(
                            format!("iteration: {} error: {:.2}", iteration, error),
                            (10, 5),
                            ("sans-serif", 25.0).into_font(),
                        ))
                        .unwrap();
                    root.present().unwrap();
                },
            )
            .as_ref(),
        );
        root_screen
            .present()
            .expect("Unable to write result to file");
        println!("{} saved!", OUTPUT_GIF);

        res
    } else {
        optimizer.optimize(&factors, &mut variables)
    };
    // let mut optimizer = NonlinearOptimizer::new(GaussNewtonOptimizer::default());
    // let start = Instant::now();
    let duration = start.elapsed();
    println!("optimize time: {:?}", duration);
    println!("opt_res {:?}", opt_res);
    Ok(())
}
