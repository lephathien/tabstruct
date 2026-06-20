
#!/bin/bash
set -e

mkdir -p data/train
mkdir -p data/wikisql
mkdir -p data/test

echo "📦 Downloading synthetic training data..."
curl -L -o data/synthetic_train.zip https://github.com/RaphaelMouravieff/TabStruct/releases/download/v1.0/synthetic_train.zip
unzip -o data/synthetic_train.zip -d data/train/

echo "📦 Downloading preprocessed WikiSQL..."
curl -L -o data/wikisql_preprocessed.zip https://github.com/RaphaelMouravieff/TabStruct/releases/download/v1.0/wikisql_preprocessed.zip
unzip -o data/wikisql_preprocessed.zip -d data/

echo "📦 Downloading generalization evaluation datasets..."
curl -L -o data/synthetic_generalization.zip https://github.com/RaphaelMouravieff/TabStruct/releases/download/v1.0/synthetic_generalization.zip
unzip -o data/synthetic_generalization.zip -d data/

echo "✅ All datasets downloaded and extracted."


rm -rf data/synthetic_generalization.zip
rm -rf data/synthetic_train.zip
rm -rf data/wikisql_preprocessed.zip