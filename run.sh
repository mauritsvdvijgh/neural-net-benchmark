OUTPUT_DIR="results"
FRAMEWORKS="scikit tensorflow cntk mxnet mlpack pytorch theano"
BATCH_SIZES="10000 5000 1000 500 100 50 10"
EPOCHS=10

rm -rf $OUTPUT_DIR
mkdir $OUTPUT_DIR

for FRAMEWORK in $FRAMEWORKS
do
    for BATCH_SIZE in $BATCH_SIZES
    do
        docker-compose run -T "$FRAMEWORK" \
            python3 main.py \
                --batch_sizes="$BATCH_SIZE" \
                --epochs="$EPOCHS" \
            > $OUTPUT_DIR/"$FRAMEWORK"_"$EPOCHS"_"$BATCH_SIZE".txt
    done
done
