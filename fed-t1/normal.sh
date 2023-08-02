#!/bin/sh

now="$(date +'%Y%m%d%s')"
echo "INFO. : Test Time : $now"

dirname="test-log-"$now

echo "INFO. : The directory name for storing log files is $dirname."

if [ -d $dirname ]; then
    # 目錄存在
    echo "INFO. : Directory exists."
else
    # 目錄不存在
    echo "INFO. : Create directory automatically because directory does not exist."
    mkdir $dirname
fi

# python XXXXXX.py >> ${dirname}/${dirname}_log.txt

python example_code_mnist.py > ${dirname}/log_ex_code_mnist_${now}.txt

python example_code_cifar10.py > ${dirname}/log_ex_code_cifar10_${now}.txt

python example_code_cifar100.py > ${dirname}/log_ex_code_cifar100_${now}.txt

python example_code_synthetic.py > ${dirname}/log_ex_code_synthetic_${now}.txt



