wget -O demo_data.zip https://github.com/HanGuo97/soft-Q-learning-for-text-generation/files/6764327/pplm-gpt2.zip

unzip demo_data.zip
rm demo_data.zip
mv pplm-gpt2 demo_data
cd demo_data
sed -i 's/Ġ//g' train.targets*
wget https://raw.githubusercontent.com/HanGuo97/soft-Q-learning-for-text-generation/main/experiments/pplm-inputs.txt
mkdir wordlists
cd wordlists

for i in legal computers politics science space military religion
do
    wget https://raw.githubusercontent.com/HanGuo97/soft-Q-learning-for-text-generation/main/experiments/wordlists/$i.txt
done

cd ../../