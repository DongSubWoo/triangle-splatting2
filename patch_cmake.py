path = 'C:/Users/dwoo/AppData/Local/miniforge3/envs/triangle-splatting2/Lib/site-packages/torch/share/cmake/Caffe2/Modules_CUDA_fix/upstream/FindCUDA.cmake'
f = open(path, 'r')
c = f.read()
f.close()
old = 'if (EXISTS "${CUDA_TOOLKIT_ROOT}/targets/${CUDA_TOOLKIT_TARGET_NAME}")'
new = 'if (FALSE)'
c = c.replace(old, new)
f = open(path, 'w')
f.write(c)
f.close()
print('Done! Replaced:', c.count('if (FALSE)'), 'occurrence(s)')
