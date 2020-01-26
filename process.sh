cat results/predict/*.json | python util/prediction_time.py > results/prediction-table.txt
cat results/batch/*.json | python util/plot_throughput_latency.py > results/batch_throughput_latency.pdf
cat results/single/*.json | python util/plot_throughput_latency.py > results/single_throughput_latency.pdf
