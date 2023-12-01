use std::cell::RefCell;

use std::collections::HashSet;
use std::iter::zip;

use crate::core::{HashMap, Real};
use crate::prelude::{FactorGraph, FactorsContainer, OptIterate, VariablesContainer, Vkey};
use angular_units::Deg;
use dot_graph::{Edge, Graph, Kind, Node, Style, Subgraph};
use graphviz_rust::cmd::CommandArg;

use graphviz_rust::{cmd::Format, exec, parse, printer::PrinterContext};

use prisma::{Hsv, Rgb};
use rand::seq::SliceRandom;
use rand::thread_rng;
fn generate_colors(count: usize, saturation: f64) -> Vec<Hsv<f64>> {
    (0..count)
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
const PDF_PAGE_W: f32 = 210.0;
const PDF_PAGE_H: f32 = 297.0;

#[derive(Default, Debug)]
pub struct FactorGraphViz {
    svgs: Vec<String>,
    titles: Vec<String>,
    nodes_colors: RefCell<Option<Vec<Hsv<f64>>>>,
    groups_colors: RefCell<Option<Vec<Hsv<f64>>>>,
}
impl FactorGraphViz {
    pub fn add_page<FC, VC, O, R>(
        &mut self,
        factor_graph: &FactorGraph<FC, VC, O, R>,
        variables_group: Option<Vec<HighlightVariablesGroup>>,
        factors_group: Option<Vec<HighlightFactorsGroup>>,
        title: &str,
    ) where
        FC: FactorsContainer<R>,
        VC: VariablesContainer<R>,
        O: OptIterate<R>,
        R: Real,
    {
        // for svg convert
        // sudo apt install ttf-mscorefonts-installer
        let dot_content = self.generate_dot(factor_graph, variables_group, factors_group);
        {
            let g = parse(&dot_content).unwrap();
            let graph_svg = exec(
                g,
                &mut PrinterContext::default(),
                vec![
                    Format::Svg.into(),
                    // CommandArg::Custom("-Nfontname=Helvetica".to_string()),
                    CommandArg::Custom(
                        "-Nfontname=Liberation Serif,Nimbus Roman, Times, Times New Roman"
                            .to_string(),
                    ),
                ],
            )
            .expect("probaly you should install graphviz");
            // let mut output = File::create("graph.svg").unwrap();
            // write!(output, "{}", graph_svg).unwrap();
            self.titles.push(title.to_owned());
            self.svgs.push(graph_svg);
        }
    }

    pub fn save_pdf(&self, path: &str) {
        use printpdf::*;
        let page_w = Mm(PDF_PAGE_W);
        let page_h = Mm(PDF_PAGE_H);

        let mut doc: Option<PdfDocumentReference> = None;
        let mut page: Option<PdfPageIndex>;
        let mut layer: Option<PdfLayerIndex>;
        let mut font: Option<IndirectFontRef> = None;
        for (svg, title) in zip(&self.svgs, &self.titles) {
            if doc.is_none() {
                let (d, p, l) = PdfDocument::new("factor graph debug viz", page_w, page_h, "layer");
                doc = Some(d);
                page = Some(p);
                layer = Some(l);
                font = Some(
                    doc.as_mut()
                        .unwrap()
                        // .add_builtin_font(BuiltinFont::TimesRoman)
                        .add_builtin_font(BuiltinFont::Helvetica)
                        .unwrap(),
                );
            } else {
                let (p, l) = doc.as_mut().unwrap().add_page(page_w, page_h, "layer");
                page = Some(p);
                layer = Some(l);
            }
            let current_layer = doc
                .as_ref()
                .unwrap()
                .get_page(page.unwrap())
                .get_layer(layer.unwrap());
            // let svg = fs::read_to_string("graph.svg").unwrap();

            let svg = Svg::parse(svg).unwrap();
            let reference = svg.into_xobject(&current_layer);
            let svg_w = reference.width;
            let svg_h = reference.height;
            // current_layer.set_outline_color(Color::Rgb(printpdf::Rgb::new(1.0, 0.0, 0.0, None)));
            let page_background_poly = Polygon {
                rings: vec![vec![
                    (Point::new(Mm(0.0), Mm(0.0)), false),
                    (Point::new(Mm(0.0), page_h), false),
                    (Point::new(page_w, page_h), false),
                    (Point::new(page_w, Mm(0.0)), false),
                ]],
                mode: PolygonMode::FillStroke,
                winding_order: WindingOrder::NonZero,
            };

            let page_color = Color::Rgb(Rgb::new(0.0, 0.0, 0.0, None));

            // More advanced graphical options
            current_layer.set_overprint_stroke(true);
            current_layer.set_fill_color(page_color);
            current_layer.add_polygon(page_background_poly);

            current_layer.set_fill_color(Color::Rgb(printpdf::Rgb::new(1.0, 1.0, 1.0, None)));
            current_layer.use_text(title, 16.0, Mm(20.0), Mm(285.0), font.as_ref().unwrap());

            // println!(
            //     "svg_w {:?} svg_h {:?}",
            //     svg_w.clone().into_pt(300.0),
            //     svg_h.clone().into_pt(300.0)
            // );
            // println!("svg_w {:?} svg_h {:?}", svg_w, svg_h);
            // println!(
            //     "svg_w {:?} svg_h {:?}",
            //     svg_w.0 as f32 / 1.333f32,
            //     svg_h.0 as f32 / 1.333f32
            // );
            // println!(
            //     "page_w {:?} page_h {:?}",
            //     page_w.clone().into_pt(),
            //     page_h.clone().into_pt()
            // );
            let wscale = page_w.into_pt() / svg_w.into_pt(300.0);
            let hscale = page_h.into_pt() / svg_h.into_pt(300.0);

            let nw = svg_w.into_pt(300.0) * hscale;

            let scale = if nw <= page_w.into_pt() {
                hscale
            } else {
                wscale
            };
            // println!("scale {}", scale);
            let y_offset = Pt(30f32);
            reference.add_to_layer(
                &current_layer,
                SvgTransform {
                    scale_x: Some(scale),
                    scale_y: Some(scale),
                    translate_y: Some(y_offset),
                    // dpi: Some(96.0),
                    ..SvgTransform::default()
                },
            );
        }
        let pdf_bytes = doc.unwrap().save_to_bytes().unwrap();
        std::fs::write(path, pdf_bytes)
            .map_err(|_| "Failed to write PDF file")
            .unwrap();
    }

    pub fn generate_dot<FC, VC, O, R>(
        &self,
        factor_graph: &FactorGraph<FC, VC, O, R>,
        variables_group: Option<Vec<HighlightVariablesGroup>>,
        factors_group: Option<Vec<HighlightFactorsGroup>>,
    ) -> String
    where
        FC: FactorsContainer<R>,
        VC: VariablesContainer<R>,
        O: OptIterate<R>,
        R: Real,
    {
        let mut variables_types = HashMap::<Vkey, String>::default();
        let mut factors_types = HashMap::<usize, String>::default();
        let mut unique_variables_types = HashSet::<String>::default();
        let mut unique_factors_types = HashSet::<String>::default();

        for vk in factor_graph.variables().default_variable_ordering().keys() {
            let mut type_name = factor_graph.variables().type_name_at(*vk).unwrap();
            let s = type_name.split_once('<');
            if let Some(s) = s {
                type_name = s.0.to_owned();
            }
            variables_types.insert(*vk, type_name.clone());
            unique_variables_types.insert(type_name);
        }
        for fi in 0..factor_graph.factors().len() {
            let mut type_name = factor_graph.factors().type_name_at(fi).unwrap();
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
        types.sort();

        if self.nodes_colors.borrow().is_none() {
            let mut colors = generate_colors(types.len(), 1.0);
            colors.shuffle(&mut thread_rng());
            *self.nodes_colors.borrow_mut() = Some(colors);
        }

        let binding = self.nodes_colors.borrow();
        let colors = binding.as_ref().unwrap();

        let get_color = |idx: usize| -> Hsv<f64> { colors[idx % colors.len()] };

        if self.groups_colors.borrow().is_none() {
            let mut highlight_colors = generate_colors(30, 0.5);
            highlight_colors.shuffle(&mut thread_rng());
            *self.groups_colors.borrow_mut() = Some(highlight_colors);
        }

        let binding = self.groups_colors.borrow();
        let highlight_colors = binding.as_ref().unwrap();

        let get_highlight_color =
            |idx: usize| -> Hsv<f64> { highlight_colors[idx % highlight_colors.len()] };

        for (color_idx, t) in types.iter().enumerate() {
            type_to_color.insert(t.to_string(), get_color(color_idx));
        }

        let mut graph = Graph::new("factor_graph", Kind::Graph)
            .attrib("layout", "fdp")
            .attrib("splines", "true")
            .attrib("bgcolor", "black")
            // .attrib("bgcolor", "blue")
            .attrib("fontcolor", "white");
        // .attrib("margin", "0");
        // .attrib("size", &quote_string("50,50!".to_owned()))
        // .attrib("ratio", &quote_string("fill".to_owned()));

        for (vk, vt) in &variables_types {
            let color = if variables_group.is_some() || factors_group.is_some() {
                if variables_group.is_some() {
                    let vgs = variables_group.as_ref().unwrap();
                    let vg = vgs.iter().position(|g| g.keys.contains(vk));
                    if let Some(vg) = vg {
                        get_highlight_color(vg)
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
                    .attrib(
                        "fontcolor",
                        &quote_string(color_to_hexcode(fit_font_color(color))),
                    )
                    .shape("circle")
                    .style(Style::Filled)
                    .color(&color_to_hexcode(color))
                    .attrib("fillcolor", &quote_string(color_to_hexcode(color))),
            );
        }
        for (fi, ft) in &factors_types {
            let color = if variables_group.is_some() || factors_group.is_some() {
                if factors_group.is_some() {
                    let fgs = factors_group.as_ref().unwrap();
                    let fg = fgs.iter().position(|g| g.indexes.contains(fi));
                    if let Some(fg) = fg {
                        get_highlight_color(fg)
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
                    .attrib(
                        "fontcolor",
                        &quote_string(color_to_hexcode(fit_font_color(color))),
                    )
                    .shape("square")
                    .style(Style::Filled)
                    .color(&color_to_hexcode(color))
                    .attrib("fillcolor", &quote_string(color_to_hexcode(color))),
            );
        }
        for f_idx in 0..factor_graph.factors().len() {
            let f_keys = factor_graph.factors().keys_at(f_idx).unwrap();
            let mut f_type = factor_graph.factors().type_name_at(f_idx).unwrap();
            let s = f_type.split_once('<');
            if let Some(s) = s {
                f_type = s.0.to_owned();
            }
            let color = if variables_group.is_some() || factors_group.is_some() {
                if factors_group.is_some() {
                    let fgs = factors_group.as_ref().unwrap();
                    let fg = fgs.iter().position(|g| g.indexes.contains(&f_idx));
                    if let Some(fg) = fg {
                        get_highlight_color(fg)
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
                    )
                    .color(&color_to_hexcode(color)),
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
                t.replace('<', "[").replace('>', "]"),
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
                .color("white")
                .attrib("fontcolor", "white")
                .shape("none"),
        );
        graph.add_subgraph(sg);
        graph.to_dot_string().unwrap()
    }
}
