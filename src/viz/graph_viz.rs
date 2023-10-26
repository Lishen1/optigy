use crate::prelude::{FactorGraph, FactorsContainer, OptIterate, VariablesContainer, Vkey};
use angular_units::Deg;
use dot_graph::{Edge, Graph, Kind, Node, Style, Subgraph};
use hashbrown::{HashMap, HashSet};
use nalgebra::RealField;
use num::Float;
use prisma::{Hsv, Rgb};
use rand::seq::SliceRandom;
use rand::thread_rng;

fn generate_colors(count: usize, saturation: f64) -> Vec<Hsv<f64>> {
    (0..count)
        .into_iter()
        .map(|i| Hsv::new(Deg((i as f64 / count as f64) * 360.0), saturation, 1.0))
        .collect()
}

fn color_to_hexcode(color: Hsv<f64>) -> String {
    let color = Rgb::from(color);
    format!(
        "#{:02x}{:02x}{:02x}",
        (color.red() * 255.0) as u32,
        (color.green() * 255.0) as u32,
        (color.blue() * 255.0) as u32
    )
}
fn color_luminace(color: Hsv<f64>) -> f64 {
    let rgb = Rgb::from(color);
    rgb.red().powf(2.2) * 0.2126 + rgb.green().powf(2.2) * 0.7151 + rgb.blue().powf(2.2) * 0.0721
}
//https://stackoverflow.com/questions/3116260/given-a-background-color-how-to-get-a-foreground-color-that-makes-it-readable-o
fn fit_font_color(color: Hsv<f64>) -> Hsv<f64> {
    let y0 = color_luminace(color);
    let c0 = (y0 + 0.05) / (0.0 + 0.05); // contrast of black font
    let c1 = (1.0 + 0.05) / (y0 + 0.05); // contrast of white font
    if c0 < c1 {
        Hsv::<f64>::new(Deg(0.0), 0.0, 1.0)
    } else {
        Hsv::<f64>::new(Deg(0.0), 0.0, 0.0)
    }
}

fn dimm_color(color: Hsv<f64>) -> Hsv<f64> {
    let mut color = color;
    color.set_value(0.25);
    color.set_saturation(0.8);
    color
}

fn quote_string(s: String) -> String {
    format!("\"{}\"", s)
}

pub struct HighlightVariablesGroup {
    keys: Vec<Vkey>,
    title: String,
}
impl HighlightVariablesGroup {
    pub fn new(keys: Vec<Vkey>, title: &str) -> Self {
        HighlightVariablesGroup {
            keys,
            title: title.to_owned(),
        }
    }
}

pub struct HighlightFactorsGroup {
    indexes: Vec<usize>,
    title: String,
}
impl HighlightFactorsGroup {
    pub fn new(indexes: Vec<usize>, title: &str) -> Self {
        HighlightFactorsGroup {
            indexes,
            title: title.to_owned(),
        }
    }
}

#[derive(Default, Debug)]
pub struct FactorGraphViz {}
impl FactorGraphViz {
    pub fn generate_dot<FC, VC, O, R>(
        &self,
        factor_graph: &FactorGraph<FC, VC, O, R>,
        variables_group: Option<Vec<HighlightVariablesGroup>>,
        factors_group: Option<Vec<HighlightFactorsGroup>>,
    ) where
        FC: FactorsContainer<R>,
        VC: VariablesContainer<R>,
        O: OptIterate<R>,
        R: RealField + Float,
    {
        let mut variables_types = HashMap::<Vkey, String>::default();
        let mut factors_types = HashMap::<usize, String>::default();
        let mut unique_variables_types = HashSet::<String>::default();
        let mut unique_factors_types = HashSet::<String>::default();

        for vk in factor_graph.variables.default_variable_ordering().keys() {
            let mut type_name = factor_graph.variables.type_name_at(*vk).unwrap();
            let s = type_name.split_once('<');
            if let Some(s) = s {
                type_name = s.0.to_owned();
            }
            variables_types.insert(*vk, type_name.clone());
            unique_variables_types.insert(type_name);
        }
        for fi in 0..factor_graph.factors.len() {
            let mut type_name = factor_graph.factors.type_name_at(fi).unwrap();
            let s = type_name.split_once('<');
            if let Some(s) = s {
                type_name = s.0.to_owned();
            }
            factors_types.insert(fi, type_name.clone());
            unique_factors_types.insert(type_name);
        }
        let mut type_to_color = HashMap::<String, Hsv<f64>>::default();
        let mut types: Vec<String> = unique_factors_types.into_iter().collect();
        types.append(&mut unique_variables_types.into_iter().collect::<Vec<String>>());

        let mut colors = generate_colors(types.len(), 1.0);
        colors.shuffle(&mut thread_rng());

        let get_color = |idx: usize| -> Hsv<f64> { colors[idx % colors.len()].clone() };

        let mut highlight_colors = generate_colors(30, 0.5);
        highlight_colors.shuffle(&mut thread_rng());

        let get_highlight_color =
            |idx: usize| -> Hsv<f64> { highlight_colors[idx % highlight_colors.len()].clone() };

        for (color_idx, t) in types.iter().enumerate() {
            type_to_color.insert(t.to_string(), get_color(color_idx));
        }

        let mut graph = Graph::new("factor_graph", Kind::Graph)
            .attrib("layout", "fdp")
            .attrib("splines", "true")
            .attrib("bgcolor", "black")
            .attrib("fontcolor", "white");

        for (vk, vt) in &variables_types {
            let color = if variables_group.is_some() || factors_group.is_some() {
                if variables_group.is_some() {
                    let vgs = variables_group.as_ref().unwrap();
                    let vg = vgs.iter().position(|g| g.keys.contains(vk));
                    if vg.is_some() {
                        get_highlight_color(vg.unwrap())
                    } else {
                        dimm_color(type_to_color[vt])
                    }
                } else {
                    dimm_color(type_to_color[vt])
                }
            } else {
                type_to_color[vt]
            };
            graph.add_node(
                Node::new(format!("x{}", vk.0).as_str())
                    .shape(Some("circle"))
                    .style(Style::Filled)
                    .attrib(
                        "fontcolor",
                        &quote_string(color_to_hexcode(fit_font_color(color))),
                    )
                    .color(Some(&color_to_hexcode(color))),
            );
        }
        for (fi, ft) in &factors_types {
            let color = if variables_group.is_some() || factors_group.is_some() {
                if factors_group.is_some() {
                    let fgs = factors_group.as_ref().unwrap();
                    let fg = fgs.iter().position(|g| g.indexes.contains(fi));
                    if fg.is_some() {
                        get_highlight_color(fg.unwrap())
                    } else {
                        dimm_color(type_to_color[ft])
                    }
                } else {
                    dimm_color(type_to_color[ft])
                }
            } else {
                type_to_color[ft]
            };
            graph.add_node(
                Node::new(format!("f{}", fi).as_str())
                    .shape(Some("square"))
                    .style(Style::Filled)
                    .attrib(
                        "fontcolor",
                        &quote_string(color_to_hexcode(fit_font_color(color))),
                    )
                    .color(Some(&color_to_hexcode(color))),
            );
        }
        for f_idx in 0..factor_graph.factors.len() {
            let f_keys = factor_graph.factors.keys_at(f_idx).unwrap();
            let mut f_type = factor_graph.factors.type_name_at(f_idx).unwrap();
            let s = f_type.split_once('<');
            if let Some(s) = s {
                f_type = s.0.to_owned();
            }
            let color = if variables_group.is_some() || factors_group.is_some() {
                if factors_group.is_some() {
                    let fgs = factors_group.as_ref().unwrap();
                    let fg = fgs.iter().position(|g| g.indexes.contains(&f_idx));
                    if fg.is_some() {
                        get_highlight_color(fg.unwrap())
                    } else {
                        dimm_color(type_to_color[&f_type])
                    }
                } else {
                    dimm_color(type_to_color[&f_type])
                }
            } else {
                type_to_color[&f_type]
            };
            for vk in f_keys {
                graph.add_edge(
                    Edge::new(
                        format!("f{}", f_idx).as_str(),
                        format!("x{}", vk.0).as_str(),
                        "",
                    )
                    .color(Some(&color_to_hexcode(color))),
                );
            }
        }
        let mut legend =
            String::from(r#"<table border="0" cellborder="1" cellspacing="0" cellpadding="4">"#);
        legend = format!(
            "{}\n{}\n",
            legend, r#"<tr><td colspan="2"><b>Legend</b></td></tr>"#
        );
        for (color_idx, t) in types.iter().enumerate() {
            let mut color = get_color(color_idx);
            if variables_group.is_some() || factors_group.is_some() {
                color = dimm_color(color);
            }
            legend = format!(
                "{}<tr>\n\t<td>{}</td>\n\t<td bgcolor=\"{}\" width=\"40%\"></td>\n</tr>\n",
                legend,
                t.replace("<", "[").replace(">", "]"),
                color_to_hexcode(color)
            );
        }

        if let Some(variables_group) = variables_group {
            for (color_idx, g) in variables_group.iter().enumerate() {
                let color = get_highlight_color(color_idx);
                legend = format!(
                    "{}<tr>\n\t<td>{}</td>\n\t<td bgcolor=\"{}\" width=\"40%\"></td>\n</tr>\n",
                    legend,
                    g.title,
                    color_to_hexcode(color)
                );
            }
        }
        if let Some(factors_group) = factors_group {
            for (color_idx, g) in factors_group.iter().enumerate() {
                let color = get_highlight_color(color_idx);
                legend = format!(
                    "{}<tr>\n\t<td>{}</td>\n\t<td bgcolor=\"{}\" width=\"40%\"></td>\n</tr>\n",
                    legend,
                    g.title,
                    color_to_hexcode(color)
                );
            }
        }
        legend = format!("{}</table>", legend);
        let mut sg = Subgraph::new("cluster_Legend");
        sg.add_node(
            Node::new("Legend")
                .attrib("label", &format!("<{}>", legend))
                .attrib("color", "white")
                .attrib("fontcolor", "white")
                .shape(Some("none")),
        );
        graph.add_subgraph(sg);
        println!("{}", graph.to_dot_string().unwrap());
    }
}
