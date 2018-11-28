


DATASET_DIR=/data3/bdd100k/
OUTPUT_DIR=/data3/dyq/tfrecordsbdd_train2
python tf_convert_data.py \
    --dataset_name=bdd100k \
        --dataset_dir=${DATASET_DIR} \
            --output_name=bdd100k_train \
                --output_dir=${OUTPUT_DIR}
