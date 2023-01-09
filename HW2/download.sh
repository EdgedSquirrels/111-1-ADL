gdown https://drive.google.com/uc?id=1fOlUrEhQvxqcJSrz-FN4kSeucoR-o30W&export=download

while [ ! -f "download.zip" ]
do
    sleep 1
done

unzip download.zip


# bash ./download.sh
# bash ./run.sh /path/to/context.json /path/to/test.json /path/to/prediction.csv