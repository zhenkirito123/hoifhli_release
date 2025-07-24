cd ./data
rm -rf ./processed_data

gdown --fuzzy https://drive.google.com/file/d/1vtx8rjTGwy6hAxnB1QsMPzZGBJNr1Ffa/view?usp=drive_link
gdown --fuzzy https://drive.google.com/file/d/18JGdLLZwAIYNud6wBx8JZhHnAnSPXJmM/view?usp=drive_link
gdown --fuzzy https://drive.google.com/file/d/1npkrQd2SM14wsFxKDrIE5fNMxzE0GWo-/view?usp=sharing
gdown --fuzzy https://drive.google.com/file/d/1BYl1VUeeC-Vf_V9Sq1gw6Ft2Bcj_LyB_/view?usp=sharing
gdown --fuzzy https://drive.google.com/file/d/1dxhsEQkJzio52jUSsUkCc5VCV-FrgSRx/view?usp=sharing
gdown --fuzzy https://drive.google.com/file/d/17HjKySkDaZ-eJQ0jb3tuGtNC2ZuHnRRN/view?usp=sharing
gdown --fuzzy https://drive.google.com/file/d/1CUOEbO8dPwIY8BrovszmzU5NqVfvcggo/view?usp=sharing

cat processed_data.tar.gz.part* | tar -xzvf -

rm -f processed_data.tar.gz.partaa
rm -f processed_data.tar.gz.partab
rm -f processed_data.tar.gz.partac
rm -f processed_data.tar.gz.partad
rm -f processed_data.tar.gz.partae
rm -f processed_data.tar.gz.partaf
rm -f processed_data.tar.gz.partag

