# genBoxes

This program generates synthetic 3D point clouds, sampled from different simple object types (boxes, simplified cars (2 boxes side-by-side), spheres, cylinders) and with custom sampling parameters. The aim is to simulate simplified 3D Lidar scenes, aimed more towards autonomous driving scenes.

> [!TIP]
> The generated output can be used directly in [LearnableEarthParser](https://github.com/romainloiseau/LearnableEarthParser) through [EarthParserDataset](https://github.com/romainloiseau/EarthParserDataset), **and more specifically in the corresponding forks specifically dedicated to it: [LearnableEarthParser](https://github.com/cnstt/LearnableEarthParser) and [EarthParserDataset](https://github.com/cnstt/EarthParserDataset).**

## Set up environment and activate it

```
conda env create -f genboxes.yml
conda activate genboxes
```

## How to use

### Without visualisation
```
python main.py
```

### With visualisation
```
python main.py settings.visu=True
```

### Configuration options
Custom generation configurations can be created, using [Hydra](https://hydra.cc/) as a backend.
To understand how it works, you can have a look at the `configs` folder containing:
  - a `default.yaml` file with the default configuration;
  - and all the other config files that already contain advanced settings.

You can launch them by using the following command:
```
python main.py --config-name config_name
```
