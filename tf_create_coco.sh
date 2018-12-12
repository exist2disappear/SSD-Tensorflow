


DATASET_DIR=/data1/dataset/coco/
OUTPUT_DIR=/data2/dongyq/tfrecordscoco
python tf_convert_data.py \
    --dataset_name=coco \
        --dataset_dir=${DATASET_DIR} \
            --output_name=coco_train \
                --output_dir=${OUTPUT_DIR}
