#!/usr/bin/env bash

# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh
# Adapted from https://github.com/pytorch/fairseq/blob/master/examples/translation/prepare-iwslt14.sh

git clone https://github.com/moses-smt/mosesdecoder.git

MOSES=`pwd`/mosesdecoder

SCRIPTS=${MOSES}/scripts
TOKENIZER=${SCRIPTS}/tokenizer/tokenizer.perl
LC=${SCRIPTS}/tokenizer/lowercase.perl
CLEAN=${SCRIPTS}/training/clean-corpus-n.perl

merge_ops=32000
src=$1
tgt=$2
lang=$3

if [ -z "$1" ]; then
    echo "Missing source language argument. Script has three arguments: src tgt src-tgt."
fi

if [ -z "$2" ]; then
    echo "Missing target language argument. Script has three arguments: src tgt src-tgt."
fi

if [ -z "$3" ]; then
    echo "Missing language pair argument. Script has three arguments: src tgt src-tgt."
fi

URL="https://wit3.fbk.eu/archive/2015-01/texts/${src}/${tgt}/${lang}.tgz"
prep="../test/data/iwslt15/${lang}"
tmp=${prep}/tmp
GZ=${lang}.tgz
orig=orig

mkdir -p ${orig} ${tmp} ${prep}

echo "Downloading data from ${URL}..."
cd ${orig}
curl -O "${URL}"

if [ -f ${GZ} ]; then
    echo "Data successfully downloaded."
else
    echo "Data not successfully downloaded."
    exit
fi

tar zxvf ${GZ}
cd ..

echo "pre-processing train data..."
for l in ${src} ${tgt}; do
    f=train.tags.$lang.$l
    tok=train.tags.$lang.tok.$l

    cat ${orig}/${lang}/${f} | \
    grep -v '<url>' | \
    grep -v '<talkid>' | \
    grep -v '<keywords>' | \
    sed -e 's/<title>//g' | \
    sed -e 's/<\/title>//g' | \
    sed -e 's/<description>//g' | \
    sed -e 's/<\/description>//g' | \
    perl ${TOKENIZER} -threads 8 -l $l > ${tmp}/${tok}
    echo ""
done
perl ${CLEAN} -ratio 1.5 ${tmp}/train.tags.${lang}.tok ${src} ${tgt} ${tmp}/train.tags.${lang}.clean 1 80
for l in ${src} ${tgt}; do
    perl ${LC} < ${tmp}/train.tags.${lang}.clean.${l} > ${tmp}/train.tags.${lang}.${l}
done

echo "pre-processing valid/test data..."
for l in ${src} ${tgt}; do
    for o in `ls ${orig}/${lang}/IWSLT15.TED*.${l}.xml`; do
    fname=${o##*/}
    f=${tmp}/${fname%.*}
    echo $o $f
    grep '<seg id' $o | \
        sed -e 's/<seg id="[0-9]*">\s*//g' | \
        sed -e 's/\s*<\/seg>\s*//g' | \
        sed -e "s/\’/\'/g" | \
    perl ${TOKENIZER} -threads 8 -l ${l} | \
    perl ${LC} > ${f}
    echo ""
    done
done

echo "creating train, valid, test..."
for l in ${src} ${tgt}; do
    awk '{if (NR%23 == 0)  print $0; }' ${tmp}/train.tags.${lang}.${l} > ${tmp}/valid.${l}
    awk '{if (NR%23 != 0)  print $0; }' ${tmp}/train.tags.${lang}.${l} > ${tmp}/train.${l}
    
    cat ${tmp}/*.dev*.${lang}.${l} \
        ${tmp}/*.tst*.${lang}.${l} 
        > ${tmp}/test.${l}
done

echo "learning * joint * BPE..."
codes_file="${tmp}/bpe.${merge_ops}"
cat "${tmp}/train.${src}" "${tmp}/train.${tgt}" > ${tmp}/train.tmp
python3 -m subword_nmt.learn_bpe -s "${merge_ops}" -i "${tmp}/train.tmp" -o "${codes_file}"
rm "${tmp}/train.tmp"

echo "applying BPE..."
for l in ${src} ${tgt}; do
    for p in train valid test; do
        python3 -m subword_nmt.apply_bpe -c "${codes_file}" -i "${tmp}/${p}.${l}" -o "${prep}/${p}.bpe.${merge_ops}.${l}"
    done
done

for l in ${src} ${tgt}; do
    for p in train valid test; do
        mv ${tmp}/${p}.${l} ${prep}/
    done
done

mv "${codes_file}" "${prep}/"
rm -rf ${MOSES}
rm -rf ${tmp}

echo "Done pre-processing small corpus."