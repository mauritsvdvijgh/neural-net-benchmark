# Neural network benchmarks

## Requirements
- docker
- docker-compose
- md5sum

## How to run
```
./run.sh
````
The results will be written to the results folder.

## Post process
```
./post-process.sh
```
Outputs prediction performance table data and throughput vs latency scatter plots based on the .json results in the results folder.

## Implement other models
The models are implemented in `benchmark/models.py` they can be modified there.

## Project layout

```
├── benchmark <- code for running the benchmarks
│   ├── main.py
│   ├── models.py <- "equivalent" model implementation for every framework
├── data
│   └── data.md5 <- MD5 hashes of datasets to verify dataset download
├── docker <- contains files for building docker images
│   ├── cntk
│   │   └── requirements.txt <- version pinned python dependencies per framework
│   ├── mxnet
│   │   └── requirements.txt
│   ├── python-dockerfile <- base python image for all frameworks and tools
│   ...
│   └── util
│       └── requirements.txt <- dependencies for the utility scripts
├── docker-compose.yml <- all container definitions and config
├── docs
│   └── frameworks.txt <- list of frameworks to compare
├── post_process.sh <- post process the results to obtain tables and plots
├── run.sh <- download the datasets and run the benchmarks!
└── util
    ├── download.py <- download all the necessary datasets
    ├── plot_throughput_latency.py <- create a throughput latency plot
    └── prediction_time.py <- output prediction stats table data
```
