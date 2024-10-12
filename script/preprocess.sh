DIR=`pwd`
input_file=${DIR}/data/public_test/vimmsd-public-test.json
output_file_1=${DIR}/data/public_test/vimmsd-public-test_01.json
output_file_2=${DIR}/data/public_test/vimmsd-public-test_02.json

python ${DIR}/src/preprocess/split_file.py --input_file $input_file \
                                           --output_file_1 $output_file_1 \
                                           --output_file_2 $output_file_2