git clone https://github.com/microsoft/sccl.git
cd sccl
pip install -e .
cd ..
git clone https://github.com/microsoft/msccl.git
cd msccl
make -j src.build
cd ..
git clone https://github.com/parasailteam/nccl-tests.git
cd nccl-tests
make MPI=1 NCCL_HOME=~/msccl/build -j
