#!/usr/bin/env bash

# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh
# Adapted from https://github.com/pytorch/fairseq/blob/master/examples/translation/prepare-iwslt14.sh

echo "Setting up tokenizers"
git clone https://github.com/moses-smt/mosesdecoder.git
git clone https://github.com/vncorenlp/VNCoreNLP.git

bashloc=`pwd`
MOSES=`pwd`/mosesdecoder
VNTOK=`pwd`/VNCoreNLP
vntokjar=(${VNTOK}/*.jar)

SCRIPTS=${MOSES}/scripts
TOKENIZER_src=${SCRIPTS}/tokenizer/tokenizer.perl
LC=${SCRIPTS}/tokenizer/lowercase.perl
CLEAN=${SCRIPTS}/training/clean-corpus-n.perl

merge_ops=32000
src=$1
tgt=vi
lang=${src}-vi

TOKENIZER_tgt=${vntokjar[-1]}

if [ -z "$1" ]; then
    echo "Missing source language argument."
fi

URL="https://wit3.fbk.eu/archive/2015-01/texts/${src}/${tgt}/${lang}.tgz"
prep="${bashloc}/test/data/iwslt15/${lang}"
tmp=${prep}/tmp
GZ=${lang}.tgz
orig=`pwd`/orig

mkdir -p ${orig} ${tmp} ${prep}

yes '' | sed 5q
echo "Downloading data from ${URL}..."
curl -O "${URL}"

if [ -f ${GZ} ]; then
    echo "Data successfully downloaded."
else
    echo "Data not successfully downloaded."
    exit
fi

tar zxvf ${GZ}
mv ${bashloc}/${lang} ${orig}/${lang}

yes '' | sed 5q
echo "pre-processing train data..."
f=train.tags.$lang.$src
tok=train.tags.$lang.tok.$src

cat ${orig}/${lang}/${f} | \
grep -v '<url>' | \
grep -v '<talkid>' | \
grep -v '<keywords>' | \
sed -e 's/<title>//g' | \
sed -e 's/<\/title>//g' | \
sed -e 's/<description>//g' | \
sed -e 's/<\/description>//g' | \
perl ${TOKENIZER_src} -threads 8 -l $src > ${tmp}/${tok}

f=train.tags.$lang.$tgt
pretok=train.tags.$lang.pretok.$tgt
tok=train.tags.$lang.tok.$tgt

yes '' | sed 2q

cd ${bashloc}/VNCoreNLP

cat ${orig}/${lang}/${f} | \
grep -v '<url>' | \
grep -v '<talkid>' | \
grep -v '<keywords>' | \
sed -e 's/<title>//g' | \
sed -e 's/<\/title>//g' | \
sed -e 's/<description>//g' | \
sed -e 's/<\/description>//g' | \
sed -e G | \
sed -e 's/^$/Obamabarack\./' \
> ${tmp}/${pretok}
java -Xmx2g -jar ${TOKENIZER_tgt} -fin ${tmp}/${pretok} -fout ${tmp}/${tok} -annotators wseg

cat ${tmp}/${tok} | \
sed -e 's/^[0-9]*[[:blank:]]//g' -e 's/[[[:blank:]]_]*//g' -e 's/</\&lt;/' -e 's/>/\&gt;/' | \
sed -e ':a;N;/\n$/!s/[[:blank:]]*\n/ /;ta;P;d' | \
sed -e ':a;N;$!ba;s/\n//g' | \
sed -e 's/[[:blank:]]Obamabarack[[:blank:]]\.[[:blank:]]/\n/g' | \
sed -e '${/^$/d;}' \
> ${tmp}/${pretok}
mv ${tmp}/${pretok} ${tmp}/${tok}

cd ..

perl ${CLEAN} -ratio 1.5 ${tmp}/train.tags.${lang}.tok ${src} ${tgt} ${tmp}/train.tags.${lang}.clean 1 80
#perl ${CLEAN} -ratio ${clean_ratio} ${tmp}/train.tags.${lang}.tok ${src} ${tgt} ${tmp}/train.tags.${lang}.clean 1 80
for l in ${src} ${tgt}; do
    perl ${LC} < ${tmp}/train.tags.${lang}.clean.${l} > ${tmp}/train.tags.${lang}.${l}
done

yes '' | sed 5q
echo "pre-processing valid/test data..."
for o in `ls ${orig}/${lang}/IWSLT15.TED*.${src}.xml`; do
    fname=${o##*/}
    f=${tmp}/${fname%.*}
    echo $o $f
    grep '<seg id' $o | \
        sed -e 's/<seg id="[0-9]*">\s*//g' | \
        sed -e 's/\s*<\/seg>\s*//g' | \
        sed -e "s/\’/\'/g" | \
    perl ${TOKENIZER_src} -threads 8 -l ${src} | \
    perl ${LC} > ${f}
    echo ""
done

yes '' | sed 2q

cd ${bashloc}/VNCoreNLP

for o in `ls ${orig}/${lang}/IWSLT15.TED*.${tgt}.xml`; do
    fname=${o##*/}
    pre_f=${tmp}/${fname%.*}pre
    f=${tmp}/${fname%.*}
    echo $o $f
    grep '<seg id' $o | \
        sed -e 's/<seg id="[0-9]*">\s*//g' | \
        sed -e 's/\s*<\/seg>\s*//g' | \
        sed -e "s/\’/\'/g" | \
        sed -e G | \
        sed -e 's/^$/Obamabarack\./' \
    > ${pre_f}
    
    java -Xmx2g -jar ${TOKENIZER_tgt} -fin ${pre_f} -fout ${f} -annotators wseg

    cat ${f} | \
    sed -e 's/^[0-9]*[[:blank:]]//g' -e 's/[[[:blank:]]_]*//g' -e 's/</\&lt;/' -e 's/>/\&gt;/' | \
    sed -e ':a;N;/\n$/!s/[[:blank:]]*\n/ /;ta;P;d' | \
    sed -e ':a;N;$!ba;s/\n//g' | \
    sed -e 's/[[:blank:]]Obamabarack[[:blank:]]\.[[:blank:]]/\n/g' | \
    sed -e '${/^$/d;}' | \
    perl ${LC} \
    > ${pre_f}
    mv ${pre_f} ${f}

    echo ""
done

cd ..

yes '' | sed 5q

echo "creating train, valid, test..."
for l in ${src} ${tgt}; do
    awk '{if (NR%23 == 0)  print $0; }' ${tmp}/train.tags.${lang}.${l} > ${tmp}/valid.${l}
    awk '{if (NR%23 != 0)  print $0; }' ${tmp}/train.tags.${lang}.${l} > ${tmp}/train.${l}
    
    cat ${tmp}/*.dev*.${lang}.${l} \
        ${tmp}/*.tst*.${lang}.${l} \
        > ${tmp}/test.${l}
done


yes '' | sed 5q

echo "learning * separate * BPEs..."
codes_file_src="${tmp}/bpe.${merge_ops}.${src}"
codes_file_tgt="${tmp}/bpe.${merge_ops}.${tgt}"

python3 -m subword_nmt.learn_bpe -s "${merge_ops}" -i "${tmp}/train.${src}" -o "${codes_file_src}"
python3 -m subword_nmt.learn_bpe -s "${merge_ops}" -i "${tmp}/train.${tgt}" -o "${codes_file_tgt}"

yes '' | sed 5q

echo "applying BPE..."
for p in train valid test; do
        python3 -m subword_nmt.apply_bpe -c "${codes_file_src}" -i "${tmp}/${p}.${src}" -o "${prep}/${p}.bpe.${merge_ops}.${src}"
        python3 -m subword_nmt.apply_bpe -c "${codes_file_tgt}" -i "${tmp}/${p}.${tgt}" -o "${prep}/${p}.bpe.${merge_ops}.${tgt}"
done

for l in ${src} ${tgt}; do
    for p in train valid test; do
        mv ${tmp}/${p}.${l} ${prep}/
    done
done

mv "${codes_file_src}" "${prep}/"
mv "${codes_file_tgt}" "${prep}/"
rm -rf ${MOSES}
rm -rf ${VNTOK}
rm -rf ${tmp}

echo "Done pre-processing small corpus."