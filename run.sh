OUTPUT_DIR="results"
FRAMEWORKS="scikit tensorflow cntk mxnet pytorch theano"
MULTI_BATCH_SIZES="8 16 32 64 128 256 512 1024 2048 4096 8192"
MULTI_EPOCHS=3

if md5sum --quiet -c data/data.md5; then
    echo "datasets are downloaded and valid"
else
    echo "datasets are missing or corrupted, downloading..."
    docker-compose run download
fi

rm -rf $OUTPUT_DIR
mkdir $OUTPUT_DIR
mkdir $OUTPUT_DIR/predict
mkdir $OUTPUT_DIR/batch
mkdir $OUTPUT_DIR/single


echo "running single datapoint prediction benchmark"
for FRAMEWORK in $FRAMEWORKS
do
    echo "$FRAMEWORK | epochs: 3 | batch size: 128"
    docker-compose run -T "$FRAMEWORK" \
        --batch_size=128 \
        --epochs=3 \
        --predict \
        > $OUTPUT_DIR/predict/"$FRAMEWORK"_3_128.json
done

echo "running multi batch size benchmark"
for BATCH_SIZE in $MULTI_BATCH_SIZES
do
    for FRAMEWORK in $FRAMEWORKS
    do
        echo "$FRAMEWORK | epochs: $MULTI_EPOCHS | batch size: $BATCH_SIZE"
        docker-compose run -T "$FRAMEWORK" \
                --batch_size="$BATCH_SIZE" \
                --epochs="$MULTI_EPOCHS" \
            > $OUTPUT_DIR/batch/"$FRAMEWORK"_"$MULTI_EPOCHS"_"$BATCH_SIZE".json
    done
done

echo "running single datapoint benchmark"
for FRAMEWORK in $FRAMEWORKS
do
    echo "$FRAMEWORK | epochs: 1 | batch size: 1"
    docker-compose run -T "$FRAMEWORK" \
        --batch_size=1 \
        --epochs=1 \
        > $OUTPUT_DIR/single/"$FRAMEWORK"_1_1.json
done
