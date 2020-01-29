echo "Booting up the orca"
docker-compose up -d orca
sleep 2
echo "Generating prediction table data"
cat results/predict/*.json | docker-compose run post-process prediction_time.py > results/prediction-table.txt
echo "Generating batch throughput latency scatter plot"
cat results/batch/*.json | docker-compose run post-process plot_throughput_latency.py > results/batch_throughput_latency.pdf
echo "Generating batchsize throughput bar chart"
cat results/batch/*.json | docker-compose run post-process plot_batchsize_throughput.py > results/batch_batchsize_throughput.pdf
echo "Generating batchsize accuracy bar chart"
cat results/batch/*.json | docker-compose run post-process plot_batchsize_accuracy.py > results/batch_batchsize_accuracy.pdf
echo "Generating single throughput latency scatter plot"
cat results/single/*.json | docker-compose run post-process plot_throughput_latency.py > results/single_throughput_latency.pdf
echo "Cleaning up the orca"
docker-compose rm -f orca
