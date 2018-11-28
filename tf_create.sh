


DATASET_DIR=/data1/dataset/voc/VOCdevkit/VOC2007/
OUTPUT_DIR=/data3/dyq/tfrecords
python tf_convert_data.py \
    --dataset_name=pascalvoc \
        --dataset_dir=${DATASET_DIR} \
            --output_name=voc_2007_train \
                --output_dir=${OUTPUT_DIR}
