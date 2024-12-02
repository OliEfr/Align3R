mkdir -p Tartanair
cd Tartanair
git clone https://github.com/castacks/tartanair_tools.git
python tartanair_tools/download_training.py --output-dir ./ --rgb --depth 

DIR="./"
if [ ! -d "$DIR" ]; then
  echo "Directory $DIR does not exist."
  exit 1
fi
for zipfile in "$DIR"/*.zip; do
  if [ -e "$zipfile" ]; then
    filename=$(basename "$zipfile")
    case "$filename" in
      abandonedfactory* | carwelding* | amusement* | abandonedfactory_night* |endofworld* |gascola* )
        echo "Skipping $zipfile ..."
        continue
        ;;
      *)
        echo "Unzipping $zipfile ..."
        unzip -o "$zipfile" -d "$DIR"
        if [ $? -eq 0 ]; then
          echo "$zipfile unzipped successfully."
        else
          echo "Failed to unzip $zipfile."
        fi
        ;;
    esac
  else
    echo "No zip files found in $DIR."
    exit 1
  fi
done
