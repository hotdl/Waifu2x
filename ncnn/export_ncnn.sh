styles=(anime photo)

for i in {1..2}; do
  style=${styles[$i]}
  mkdir -p ncnn_model/"$style"
  for i in {0..4}; do
    if [ $i -eq 4 ]; then
      model_name=scale
    else
      model_name=noise${i}
    fi
    echo $model_name

    # onnx2ncnn
    onnx_path=onnx_model/$style/${model_name}-sim.onnx
    param_path=ncnn_model/$style/waifu2x.param
    bin_path=ncnn_model/$style/${model_name}.bin
    param_opt_path=ncnn_model/$style/waifu2x_opt.param
    bin_opt_path=ncnn_model/$style/${model_name}_opt.bin
    idc_opt_path=ncnn_model/$style/waifu2x_opt.idc.h
    mem_opt_path=ncnn_model/$style/${model_name}_opt.mem.h
    param_opt_fp16_path=ncnn_model/$style/waifu2x_opt_fp16.param
    bin_opt_fp16_path=ncnn_model/$style/${model_name}_opt_fp16.bin
    idc_opt_fp16_path=ncnn_model/$style/waifu2x_opt_fp16.idc.h
    mem_opt_fp16_path=ncnn_model/$style/${model_name}_opt_fp16.mem.h

    onnx2ncnn $onnx_path $param_path $bin_path

    #ncnnopt
    ncnnoptimize $param_path $bin_path $param_opt_path $bin_opt_path 0
    ncnnoptimize $param_path $bin_path $param_opt_fp16_path $bin_opt_fp16_path 1

    #ncnn2mem
    ncnn2mem $param_opt_path $bin_opt_path $idc_opt_path $mem_opt_path
    ncnn2mem $param_opt_fp16_path $bin_opt_fp16_path $idc_opt_fp16_path $mem_opt_fp16_path
  done
done
