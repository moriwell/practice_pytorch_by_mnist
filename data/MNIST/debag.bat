@echo off

set root_path=J:\git_works\python\step3\AE
set exec_path=C:\Users\hara\Anaconda3

set output_path=%root_path%\results2\test
set model=%root_path%\results2\epoch_69.npz
set gpu=0
set type=test2

python %root_path%\data2\add_noise.py -n gaussian -s Folse
pause