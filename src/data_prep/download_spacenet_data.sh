#!/bin/bash
# download_spacenet.sh
# Downloads SpaceNet AOI_8 (Mumbai) and AOI_5 (Khartoum) from AWS S3

# Download Khartoum
echo "Downloading AOI_2 Vegas..."
aws s3 cp s3://spacenet-dataset/spacenet/SN3_roads/tarballs/SN3_roads_train_AOI_2_Vegas.tar.gz data/raw/ --no-sign-request
tar -xzvf data/raw/SN3_roads_train_AOI_2_Vegas.tar.gz -C data/raw/

# Download Khartoum
echo "Downloading AOI_3 Paris..."
aws s3 cp s3://spacenet-dataset/spacenet/SN3_roads/tarballs/SN3_roads_train_AOI_3_Paris.tar.gz data/raw/ --no-sign-request
tar -xzvf data/raw/SN3_roads_train_AOI_3_Paris.tar.gz -C data/raw/

# Download Khartoum
echo "Downloading AOI_4 Shanghai..."
aws s3 cp s3://spacenet-dataset/spacenet/SN3_roads/tarballs/SN3_roads_train_AOI_4_Shanghai.tar.gz data/raw/ --no-sign-request
tar -xzvf data/raw/SN3_roads_train_AOI_4_Shanghai.tar.gz -C data/raw/

# Download Khartoum
echo "Downloading AOI_5 Khartoum..."
aws s3 cp s3://spacenet-dataset/spacenet/SN3_roads/tarballs/SN3_roads_train_AOI_5_Khartoum.tar.gz data/raw/ --no-sign-request
tar -xzvf data/raw/SN3_roads_train_AOI_5_Khartoum.tar.gz -C data/raw/

echo "Download and extraction complete"