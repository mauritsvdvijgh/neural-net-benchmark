echo "Booting up the orca"
docker-compose up -d orca
sleep 2
echo "Building post-process container image"
docker-compose build -q post-process
echo "Generating prediction table data"
cat results/predict/*.json | docker-compose run --no-deps --rm  post-process prediction_time.py > results/prediction-table.txt
echo "Generating batch throughput latency scatter plot"
cat results/batch/*.json | docker-compose run --rm post-process plot_throughput_latency.py > results/batch_throughput_latency.pdf
echo "Generating single throughput latency scatter plot"
cat results/single/*.json | docker-compose run --rm post-process plot_throughput_latency.py > results/single_throughput_latency.pdf
echo "Killing the orca"
docker-compose rm -f orca
