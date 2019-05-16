make;
cd lib_kernel/lib_psroi_pooling
sh make.sh
cd ../..
cd lib_kernel/lib_roi_pooling
sh make.sh
cd ../..
cd lib_kernel/lib_roi_align
sh make.sh
cd ../..
cd datasets/lib_coco/PythonAPI
make install
cd ../..
