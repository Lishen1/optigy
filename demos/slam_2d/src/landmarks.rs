use std::collections::HashMap;

use nalgebra::{matrix, vector, Matrix2, Vector2};
use num::Float;
use optigy::{
    prelude::{Factor, FactorGraph, FactorsContainer, OptIterate, Real, VariablesContainer, Vkey},
    slam::se3::SE2,
};

use crate::{vision_factor::VisionFactor, E2};

pub struct Landmark<R>
where
    R: Real,
{
    id: Vkey,
    obs_cnt: usize,
    poses_keys: Vec<Vkey>,
    triangulated: bool,
    factors: Vec<VisionFactor<R>>,
}
#[allow(non_snake_case)]
impl<R> Landmark<R>
where
    R: Real,
{
    pub fn new(id: Vkey) -> Self {
        Landmark {
            id,
            obs_cnt: 0,
            poses_keys: Vec::new(),
            triangulated: false,
            factors: Vec::new(),
        }
    }
    pub fn add_observation<FC, VC, O>(
        &mut self,
        fg: &mut FactorGraph<FC, VC, O, R>,
        pose_id: Vkey,
        ray: Vector2<R>,
        cov: &Matrix2<R>,
    ) where
        FC: FactorsContainer<R>,
        VC: VariablesContainer<R>,
        O: OptIterate<R>,
    {
        // return;

        let vf = VisionFactor::<R>::new(self.id, pose_id, ray, cov.as_view());
        if self.triangulated {
            fg.add_factor(vf);
        } else {
            self.factors.push(vf);
        }
        self.obs_cnt += 1;
        self.poses_keys.push(pose_id);
        // self.triangulated = false;
    }
    fn need_to_remove(&self, pose_key: Vkey) -> bool {
        let mut rem_poses = 0_usize;
        if let Some(_idx) = self.poses_keys.iter().position(|v| *v == pose_key) {
            rem_poses += 1;
        }
        self.poses_keys.len() == rem_poses
    }
    pub fn remove_pose<FC, VC, O>(
        &mut self,
        pose_key: Vkey,
        fg: &mut FactorGraph<FC, VC, O, R>,
        marginalize: bool,
    ) -> bool
    where
        FC: FactorsContainer<R>,
        VC: VariablesContainer<R>,
        O: OptIterate<R>,
    {
        if let Some(idx) = self.poses_keys.iter().position(|v| *v == pose_key) {
            self.poses_keys.remove(idx);
            self.factors.retain(|f| {
                !(f.keys()[VisionFactor::<R>::POSE_KEY] == pose_key
                    && f.keys()[VisionFactor::<R>::LANDMARK_KEY] == self.id)
            });
        }
        if self.need_to_remove(pose_key) {
            if self.triangulated {
                fg.remove_variable(self.id, marginalize);
            }
            // self.triangulated = false;
            return true;
        }
        false
    }
    pub fn triangulate<FC, VC, O>(&mut self, fg: &mut FactorGraph<FC, VC, O, R>)
    where
        FC: FactorsContainer<R>,
        VC: VariablesContainer<R>,
        O: OptIterate<R>,
    {
        if self.obs_cnt < 2 || self.triangulated {
            return;
        }
        let mut rays = Vec::<Vector2<R>>::new();
        for vf in &self.factors {
            let r = vf.ray();
            rays.push(*r);
        }
        let mut max_ang = R::zero();
        for i in 0..rays.len() {
            for j in 0..rays.len() {
                if i == j {
                    continue;
                }
                let ri = &rays[i];
                let rj = &rays[j];
                let ang = Float::abs(Float::acos(ri.dot(rj)).to_degrees());

                if ang > max_ang {
                    max_ang = ang;
                }
            }
        }
        // println!("max_ang {}", max_ang);
        if max_ang < R::from_f64(5.0).unwrap() {
            return;
        }

        let mut A = Matrix2::<R>::zeros();
        let mut b = Vector2::<R>::zeros();
        let mut rays = Vec::<Vector2<R>>::new();
        let mut poses = Vec::<Vector2<R>>::new();
        for p_key in &self.poses_keys {
            let pose: &SE2<R> = fg.get_variable(*p_key).unwrap();
            let th = R::from_f64(pose.origin.log()[2]).unwrap();
            let R_cam = matrix![Float::cos(th), -Float::sin(th); Float::sin(th), Float::cos(th) ];
            for f_idx in 0..self.factors.len() {
                let vf = self.factors.get(f_idx).unwrap();
                if vf.keys()[1] == *p_key {
                    let p = pose.origin.params().fixed_rows::<2>(0);
                    let p = vector![R::from_f64(p[0]).unwrap(), R::from_f64(p[1]).unwrap()];
                    let r = R_cam * vf.ray();
                    let Ai = Matrix2::<R>::identity() - r * r.transpose();
                    A += Ai;
                    b += Ai * p;
                    rays.push(r);
                    poses.push(Vector2::<R>::new(p[0], p[1]));
                }
            }
        }
        if let Some(chol) = A.cholesky() {
            let coord = chol.solve(&b);
            for i in 0..poses.len() {
                let p = &poses[i];
                let r = &rays[i];
                let nr = (coord - p).normalize();
                let ang = Float::abs(Float::acos(r.dot(&nr)).to_degrees());
                if ang > R::from_f64(0.1).unwrap() {
                    return;
                }
            }
            //TODO: move whole vector
            for f in &self.factors {
                fg.add_factor(f.clone());
            }
            self.factors.clear();
            // self.factors.extend()
            fg.add_variable_with_key(
                self.id,
                E2::<R>::new(coord[0].to_f64().unwrap(), coord[1].to_f64().unwrap()),
            );
            self.triangulated = true;
        }
    }
}

#[derive(Default)]
pub struct Landmarks<R = f64>
where
    R: Real,
{
    landmarks: HashMap<Vkey, Landmark<R>>,
}

impl<R> Landmarks<R>
where
    R: Real,
{
    pub fn add_observation<FC, VC, O>(
        &mut self,
        fg: &mut FactorGraph<FC, VC, O, R>,
        pose_id: Vkey,
        landmark_id: Vkey,
        rx: R,
        ry: R,
        sx: R,
        sy: R,
        sxy: R,
    ) where
        FC: FactorsContainer<R>,
        VC: VariablesContainer<R>,
        O: OptIterate<R>,
    {
        self.landmarks
            .entry(landmark_id)
            .or_insert_with(|| Landmark::new(landmark_id));

        self.landmarks
            .get_mut(&landmark_id)
            .unwrap()
            .add_observation(
                fg,
                pose_id,
                Vector2::<R>::new(rx, ry),
                &Matrix2::new(sx, sxy, sxy, sy),
            );
    }
    pub fn triangulate<FC, VC, O>(&mut self, fg: &mut FactorGraph<FC, VC, O, R>)
    where
        FC: FactorsContainer<R>,
        VC: VariablesContainer<R>,
        O: OptIterate<R>,
    {
        for l in self.landmarks.values_mut() {
            l.triangulate(fg);
        }
    }
    pub fn proc_pose_remove<FC, VC, O>(
        &mut self,
        fg: &mut FactorGraph<FC, VC, O, R>,
        pose_id: Vkey,
        marginalize: bool,
    ) where
        FC: FactorsContainer<R>,
        VC: VariablesContainer<R>,
        O: OptIterate<R>,
    {
        self.landmarks
            .retain(|_, l| !l.remove_pose(pose_id, fg, marginalize));
    }
}
