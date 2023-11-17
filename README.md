# OptiGy
## _General purpose factor graph optimizer framework_

OptiGy general purpose [factor graph] optimizer mostly suitable for SLAM tasks ~~copied~~ inspired by [miniSAM]
>**Warning**
>**WIP**
## Features
- As fast as [miniSAM]
- Static polymorphism
- Rust only
- Numerical differentiation
- Visualization

## Installation

```sh
git clone https://github.com/Lishen1/optigy.git
cd optigy
cargo build
```
## Run examples
```sh
cd optigy/demos/pose_graph_g2o
cargo run -- --do-viz
```
![pose graph optimization](https://github.com/Lishen1/optigy/blob/master/static/pose_graph.gif)

```sh
cd optigy/demos/slam_2d
cargo run --release -- --do-viz --use-gps --use-vision --marginalize
````
## Build doc
```sh
cargo rustdoc -- --html-in-header katex-header.html
```
![2d slam](https://github.com/Lishen1/optigy/blob/master/static/2d-slam.gif)
## Factor graph visualization
![2d slam](https://github.com/Lishen1/optigy/blob/master/static/graph_viz.png)
## License
GNU GPLv3

[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)

   [factor graph]: <https://en.wikipedia.org/wiki/Factor_graph>
   [miniSAM]: <https://github.com/dongjing3309/minisam>

